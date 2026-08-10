#!/usr/bin/env python3
"""Nearest-neighbor memorization audit (camera-ready analysis).

Added in response to a reviewer request for a memorization check.

For every generated image, the distance to its nearest real neighbor is measured
in the same Inception-V3 feature space used for FID, against the *full* class
pool rather than the 200-image evaluation subset: a copied training image need
not fall inside that subset, so the full pool is the only conservative reading.

The reference is the real-to-real (leave-one-out) distance distribution of the
same class. A generator that copies its training data produces distances below
the 5th percentile of that distribution; when the two distributions coincide,
5% of generated images fall below it by construction.

Reuses the feature cache written by analysis/kid_and_resampling.py.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lacci_paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, FASTGAN_ROOT  # noqa: E402


CACHE = Path(str(OUT_ROOT / "camera_ready" / "feat_cache"))
OUT = Path(str(OUT_ROOT / "camera_ready" / "cr_memorization.csv"))
DATA = DATA_ROOT

CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
SYSTEMS = {"OrGANiCells": ([0, 1, 2, 3, 4], "fr"),
           "StyleGAN2-Ada": ([0, 1, 2], "fr"),
           "FastGAN-64": ([0, 1, 2], "rf")}
N_SAMPLE = 200
PROTOCOL_SEED = 42


def nn_dists(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Smallest Euclidean distance from each row of `a` to any row of `b`."""
    d2 = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :]
          - 2.0 * a.dot(b.T))
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    return np.sqrt(np.clip(d2.min(axis=1), 0.0, None))


def main():
    rows = []
    for cls in CLASSES:
        reals_all = sorted((DATA / f"bloodmnist_{cls}_64" / "raw").glob("*.png"))
        real_feats = np.load(CACHE / f"real_{cls}.npy").astype(np.float64)
        idx_map = {p: i for i, p in enumerate(reals_all)}

        for sys_name, (seeds, order) in SYSTEMS.items():
            for seed in seeds:
                fake_file = CACHE / f"fake_{sys_name}_{cls}_s{seed}.npy"
                if not fake_file.exists():
                    print(f"[skip] {fake_file.name}")
                    continue
                fake_feats = np.load(fake_file).astype(np.float64)

                # Nearest neighbor against the FULL class pool, not only the 200
                # images of the protocol: a copied training image need not fall
                # inside that subset, so this is the conservative reading.
                ref = real_feats
                d_synth = nn_dists(fake_feats, ref)
                d_real = nn_dists(ref, ref, exclude_self=True)
                p5 = float(np.percentile(d_real, 5))
                frac = float((d_synth < p5).mean())
                rows.append({
                    "system": sys_name, "class": cls, "seed": seed,
                    "p5_intra_real": p5,
                    "nn_dist_median_synth": float(np.median(d_synth)),
                    "nn_dist_min_synth": float(d_synth.min()),
                    "frac_below_p5": frac,
                    "passes": bool(frac <= 0.05),
                })
                print(f"[{sys_name:14s}] {cls:13s} s{seed}  "
                      f"mediana={np.median(d_synth):7.2f}  p5_real={p5:7.2f}  "
                      f"frac<p5={frac:.3f}  {'OK' if frac <= 0.05 else 'ALERTA'}",
                      flush=True)

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["passes"] for r in rows)
    print(f"\n[ok] {n_ok}/{len(rows)} runs sem indicio de memorizacao -> {OUT}")


if __name__ == "__main__":
    main()
