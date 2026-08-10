#!/usr/bin/env python3
"""KID and reference-subset confidence intervals (camera-ready analysis).

Added in response to a reviewer request for a metric that is unbiased at small
sample sizes, plus confidence intervals across sampling repetitions.

Inception-V3 features are extracted once, with the same extractor torch-fidelity
uses internally, and every metric is recomputed from them. That keeps the
resampling experiments cheap and keeps the numbers bit-comparable with the
published FID values.

The generated set stays exactly as published (the same 200 images per run); only
the 200-image real reference is redrawn, N_REDRAW times, from the full class
pool. No published FID value is modified by this script.

  python analysis/kid_and_resampling.py --verify   # reproduce published FIDs
  python analysis/kid_and_resampling.py --full     # KID + intervals, all runs
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg

from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lacci_paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, FASTGAN_ROOT  # noqa: E402


CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

DATA_BASE = DATA_ROOT
RUNS = RUNS_ROOT
ET07 = Path(str(OUT_ROOT / "et07_synthetic"))
OUT_DIR = OUT_ROOT / "camera_ready"
CACHE = OUT_DIR / "feat_cache"

N_SAMPLE = 200
PROTOCOL_SEED = 42
N_REDRAW = 20          # repeticoes do subconjunto real
REDRAW_SEED0 = 2000
KID_SUBSET_SIZE = 100  # <= N_SAMPLE (torch_fidelity's default of 1000 does not fit)
KID_N_SUBSETS = 100
KID_DEGREE = 3
KID_COEF0 = 1
BATCH = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fastgan_fakes(cls: str, seed: int) -> Path:
    return RUNS / "fastgan_p3" / "train_results" / f"{cls}_s{seed}" / "fakes_for_fid"


def stylegan_fakes(cls: str, seed: int) -> Path:
    if seed == 0:
        base = (RUNS / "stylegan_basophil_64" if cls == "basophil"
                else RUNS / "stylegan_a4" / cls)
        return base / "fakes_kimg300_for_fid"
    return RUNS / "stylegan_3seeds" / f"{cls}_s{seed}" / "fakes_kimg300_for_fid"


# order: "rf" = real images drawn BEFORE the fakes (FastGAN script);
#        "fr" = fakes drawn first (OrGANiCells and StyleGAN2-Ada scripts).
SYSTEMS = {
    "OrGANiCells": dict(
        seeds=[0, 1, 2, 3, 4], order="fr",
        lister=lambda cls, s: sorted((ET07 / cls).glob(f"g_s{s}_*.png")),
    ),
    "StyleGAN2-Ada": dict(
        seeds=[0, 1, 2], order="fr",
        lister=lambda cls, s: sorted(stylegan_fakes(cls, s).glob("*.png")),
    ),
    "FastGAN-64": dict(
        seeds=[0, 1, 2], order="rf",
        lister=lambda cls, s: sorted(fastgan_fakes(cls, s).glob("*.png")),
    ),
}

_EXTRACTOR = None


def extractor() -> FeatureExtractorInceptionV3:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = FeatureExtractorInceptionV3(
            "inception-v3-compat", ["2048"]
        ).to(DEVICE).eval()
    return _EXTRACTOR


@torch.no_grad()
def features_for(paths: list[Path]) -> np.ndarray:
    """2048-d features, same pipeline torch_fidelity uses (uint8 NCHW -> 299)."""
    fe = extractor()
    out = []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in chunk])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(DEVICE)
        feats = fe(t)[0]
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def cached_features(key: str, paths: list[Path]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{key}.npy"
    if f.exists():
        cached = np.load(f)
        if cached.shape[0] == len(paths):
            return cached
    feats = features_for(paths)
    np.save(f, feats)
    return feats


def fid_from_features(f1: np.ndarray, f2: np.ndarray) -> float:
    """Reference FID (scipy.sqrtm on 2048x2048). Slow; used only for validation."""
    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid_fast(f1: np.ndarray, f2: np.ndarray) -> float:
    """Identical FID, exploiting rank <= n-1 with n=200 << d=2048.

    With S1 = A A^T/(m-1) and S2 = B B^T/(n-1) (A, B centred, d x m and d x n),
    the non-zero eigenvalues of S1 S2 are those of (A^T B B^T A)/((m-1)(n-1)),
    an m x m matrix. So tr(sqrtm(S1 S2)) = sum(sqrt(eigenvalues)) is computed on
    200x200 instead of 2048x2048: milliseconds rather than tens of seconds.

    Verified against the scipy path above to within 1e-4 FID.
    """
    x = f1.astype(np.float64)
    y = f2.astype(np.float64)
    mu1, mu2 = x.mean(axis=0), y.mean(axis=0)
    m, n = x.shape[0], y.shape[0]
    a = (x - mu1).T                      # d x m
    b = (y - mu2).T                      # d x n
    atb = a.T.dot(b)                     # m x n
    mprod = atb.dot(atb.T) / ((m - 1) * (n - 1))   # m x m ~ A^T B B^T A escalado
    ev = np.linalg.eigvals(mprod).real
    tr_sqrt = np.sqrt(np.clip(ev, 0.0, None)).sum()
    tr_s1 = (a * a).sum() / (m - 1)
    tr_s2 = (b * b).sum() / (n - 1)
    diff = mu1 - mu2
    return float(diff.dot(diff) + tr_s1 + tr_s2 - 2 * tr_sqrt)


def _mmd2_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    m = x.shape[0]
    kxx = (x.dot(x.T) / d + KID_COEF0) ** KID_DEGREE
    kyy = (y.dot(y.T) / d + KID_COEF0) ** KID_DEGREE
    kxy = (x.dot(y.T) / d + KID_COEF0) ** KID_DEGREE
    sxx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(sxx + syy - 2 * kxy.mean())


def kid_from_features(f1: np.ndarray, f2: np.ndarray, rng: np.random.Generator) -> float:
    m = min(KID_SUBSET_SIZE, f1.shape[0], f2.shape[0])
    vals = [
        _mmd2_unbiased(f1[rng.choice(f1.shape[0], m, replace=False)],
                       f2[rng.choice(f2.shape[0], m, replace=False)])
        for _ in range(KID_N_SUBSETS)
    ]
    return float(np.mean(vals))


def protocol_indices(reals_all: list[Path], fakes_all: list[Path], order: str):
    """Reproduce exactly the draw performed by each system's original script."""
    rng = random.Random(PROTOCOL_SEED)
    if order == "rf":
        rl = rng.sample(reals_all, N_SAMPLE)
        fk = rng.sample(fakes_all, N_SAMPLE)
    else:
        fk = rng.sample(fakes_all, N_SAMPLE)
        rl = rng.sample(reals_all, N_SAMPLE)
    return rl, fk


PUBLISHED = {
    ("FastGAN-64", "basophil", 0): 49.18645145127428,
    ("FastGAN-64", "basophil", 1): 48.90634327433031,
    ("FastGAN-64", "basophil", 2): 48.702407418564576,
    ("FastGAN-64", "lymphocyte", 0): None,
    ("StyleGAN2-Ada", "basophil", 1): 54.26534921801294,
    ("StyleGAN2-Ada", "basophil", 2): 53.51930019092711,
    ("StyleGAN2-Ada", "eosinophil", 1): 44.927823155140004,
    ("OrGANiCells", "basophil", 0): 326.0491212608353,
    ("OrGANiCells", "basophil", 1): 263.5082459283723,
    ("OrGANiCells", "basophil", 2): 317.7658413468987,
}


def run(verify_only: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    targets = [k for k, v in PUBLISHED.items() if v is not None] if verify_only else None

    for sys_name, cfg in SYSTEMS.items():
        for cls in CLASSES:
            reals_all = sorted((DATA_BASE / f"bloodmnist_{cls}_64" / "raw").glob("*.png"))
            real_feats_all = None
            for seed in cfg["seeds"]:
                if verify_only and (sys_name, cls, seed) not in targets:
                    continue
                fakes_all = cfg["lister"](cls, seed)
                if len(fakes_all) < N_SAMPLE or len(reals_all) < N_SAMPLE:
                    print(f"[skip] {sys_name} {cls} s{seed}: "
                          f"fakes={len(fakes_all)} reals={len(reals_all)}")
                    continue

                rl, fk = protocol_indices(reals_all, fakes_all, cfg["order"])
                if real_feats_all is None:
                    real_feats_all = cached_features(f"real_{cls}", reals_all)
                idx_map = {p: i for i, p in enumerate(reals_all)}
                rl_idx = np.array([idx_map[p] for p in rl])

                tag = f"{sys_name.replace('/', '_')}_{cls}_s{seed}"
                fake_feats = cached_features(f"fake_{tag}", fk)
                fid_proto = fid_fast(fake_feats, real_feats_all[rl_idx])

                pub = PUBLISHED.get((sys_name, cls, seed))
                flag = ""
                if pub is not None:
                    delta = abs(fid_proto - pub)
                    flag = f"  publicado={pub:.4f}  |delta|={delta:.4f}  " \
                           f"{'OK' if delta < 0.05 else 'DIVERGENTE'}"
                if verify_only:
                    ref = fid_from_features(fake_feats, real_feats_all[rl_idx])
                    flag += f"  [ref sqrtm={ref:.4f} |d|={abs(ref - fid_proto):.5f}]"
                print(f"[{sys_name:14s}] {cls:13s} s{seed}  FID={fid_proto:8.4f}{flag}",
                      flush=True)

                if verify_only:
                    continue

                rng_np = np.random.default_rng(PROTOCOL_SEED)
                kid_proto = kid_from_features(fake_feats, real_feats_all[rl_idx], rng_np)

                fids, kids = [], []
                for r in range(N_REDRAW):
                    rr = random.Random(REDRAW_SEED0 + r)
                    sub = np.array([idx_map[p] for p in rr.sample(reals_all, N_SAMPLE)])
                    fids.append(fid_fast(fake_feats, real_feats_all[sub]))
                    kids.append(kid_from_features(fake_feats, real_feats_all[sub],
                                                  np.random.default_rng(REDRAW_SEED0 + r)))
                fids, kids = np.array(fids), np.array(kids)
                rows.append({
                    "system": sys_name, "class": cls, "seed": seed,
                    "fid_protocol": fid_proto, "fid_published": pub if pub else "",
                    "kid_protocol": kid_proto,
                    "fid_redraw_mean": fids.mean(), "fid_redraw_std": fids.std(ddof=1),
                    "fid_ci_lo": np.percentile(fids, 2.5),
                    "fid_ci_hi": np.percentile(fids, 97.5),
                    "kid_redraw_mean": kids.mean(), "kid_redraw_std": kids.std(ddof=1),
                    "kid_ci_lo": np.percentile(kids, 2.5),
                    "kid_ci_hi": np.percentile(kids, 97.5),
                    "n_redraw": N_REDRAW, "kid_subset_size": KID_SUBSET_SIZE,
                    "kid_n_subsets": KID_N_SUBSETS,
                })

    if rows:
        out = OUT_DIR / "cr_kid_ci_byseed.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[ok] {len(rows)} linhas -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="only recompute published FIDs and compare")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run(verify_only=a.verify and not a.full)
