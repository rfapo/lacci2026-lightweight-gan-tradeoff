"""FID 200v200 for the 24 FastGAN-64 generators (8 classes x 3 seeds @ 25k iter).

For every run under $RUNS_ROOT/fastgan_p3/train_results/<class>_s<seed>/:
  1. load the checkpoint models/all_25000.pth and extract `g_ema`
  2. build Generator64 (im_size=64) and apply the EMA weights
  3. generate 200 deterministic samples (PROTOCOL_SEED=42)
  4. write the PNGs to <run>/fakes_for_fid/
  5. compute FID 200v200 against the real images of $BLOODMNIST_ROOT

The same 200-image budget, protocol seed and Inception-V3 extractor are used by
the OrGANiCells and StyleGAN2-Ada scripts in this directory, which is what makes
the cross-system comparison direct.

Output: $RUNS_ROOT/fastgan_p3/p4_fastgan_fid_match200.csv
        columns: class, seed, fid_200v200, n_fakes, n_reals, ckpt_iter

Run it from a checkout of FastGAN-pytorch with the patch of patch/ applied:

  cd $FASTGAN_ROOT && python /path/to/protocol/fid_match200_fastgan.py
"""
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Local imports (run with cwd set to the FastGAN-pytorch checkout)
from models_64 import Generator
from operation import load_params

import torch_fidelity

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lacci_paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, FASTGAN_ROOT  # noqa: E402


CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
SEEDS = [0, 1, 2]

RUNS_BASE = Path(str(RUNS_ROOT / "fastgan_p3/train_results"))
DATA_BASE = DATA_ROOT
OUT_CSV = Path(str(RUNS_ROOT / "fastgan_p3/p4_fastgan_fid_match200.csv"))
TMP_BASE = Path(str(RUNS_ROOT / "fastgan_p3/tmp_fid"))

CKPT_ITER = 25000
N_SAMPLE = 200
PROTOCOL_SEED = 42
NZ = 256
NGF = 64
BATCH = 8


def generate_fakes(ckpt_path: Path, fakes_dir: Path, device: str = "cuda:0"):
    """Load the checkpoint, apply EMA weights, write N_SAMPLE PNGs to fakes_dir."""
    fakes_dir.mkdir(parents=True, exist_ok=True)
    # drop stale PNGs so the sample set is fresh
    for f in fakes_dir.glob("*.png"):
        f.unlink()

    ckpt = torch.load(str(ckpt_path), map_location=device)
    G = Generator(ngf=NGF, nz=NZ, im_size=64).to(device)
    G.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["g"].items()})
    load_params(G, ckpt["g_ema"])
    G.eval()

    torch.manual_seed(PROTOCOL_SEED)
    np.random.seed(PROTOCOL_SEED)

    n_done = 0
    with torch.no_grad():
        while n_done < N_SAMPLE:
            cur_b = min(BATCH, N_SAMPLE - n_done)
            z = torch.randn(cur_b, NZ, device=device)
            out = G(z)[0]  # [B, 3, 64, 64] in [-1, 1]
            out = (out.add(1).mul(0.5).clamp(0, 1) * 255).byte().cpu().numpy()
            for i in range(cur_b):
                img = out[i].transpose(1, 2, 0)  # HWC
                Image.fromarray(img).save(fakes_dir / f"fake_{n_done + i:04d}.png")
            n_done += cur_b
    del G, ckpt
    torch.cuda.empty_cache()


def fid_against_reals(fakes_dir: Path, reals_dir: Path, cls: str, name: str) -> tuple:
    """Sample N_SAMPLE real images with PROTOCOL_SEED, compute FID 200v200."""
    reals_all = sorted(reals_dir.glob("*.png"))
    if len(reals_all) < N_SAMPLE:
        return None, len(reals_all)

    fakes_all = sorted(fakes_dir.glob("*.png"))
    if len(fakes_all) < N_SAMPLE:
        return None, len(reals_all)

    rng = random.Random(PROTOCOL_SEED)
    rl = rng.sample(reals_all, N_SAMPLE)
    fk = rng.sample(fakes_all, N_SAMPLE)

    ftmp = TMP_BASE / f"f_{name}"
    rtmp = TMP_BASE / f"r_{name}"
    for d in (ftmp, rtmp):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for src in fk:
        (ftmp / src.name).symlink_to(src)
    for src in rl:
        (rtmp / src.name).symlink_to(src)

    res = torch_fidelity.calculate_metrics(
        input1=str(ftmp), input2=str(rtmp),
        fid=True, isc=False, kid=False, prc=False, verbose=False,
    )
    fid = float(res["frechet_inception_distance"])

    shutil.rmtree(ftmp, ignore_errors=True)
    shutil.rmtree(rtmp, ignore_errors=True)

    return fid, len(reals_all)


def main():
    if TMP_BASE.exists():
        shutil.rmtree(TMP_BASE)
    TMP_BASE.mkdir(parents=True)

    rows = []
    total = len(CLASSES) * len(SEEDS)
    idx = 0
    for cls in CLASSES:
        reals_dir = DATA_BASE / f"bloodmnist_{cls}_64" / "raw"
        if not reals_dir.exists():
            print(f"[!] {cls}: reals dir absent ({reals_dir}) — SKIP all seeds")
            continue

        for seed in SEEDS:
            idx += 1
            name = f"{cls}_s{seed}"
            run_dir = RUNS_BASE / name
            ckpt = run_dir / "models" / f"all_{CKPT_ITER}.pth"
            if not ckpt.exists():
                print(f"[{idx}/{total}] {name}: ckpt absent ({ckpt}) — SKIP")
                continue

            fakes_dir = run_dir / "fakes_for_fid"
            print(f"\n[{idx}/{total}] {name}  -> gerando {N_SAMPLE} samples...")
            generate_fakes(ckpt, fakes_dir)

            print(f"          -> FID 200v200 ...")
            fid, n_reals = fid_against_reals(fakes_dir, reals_dir, cls, name)
            if fid is None:
                print(f"          [!] reals insuficientes")
                continue
            print(f"          FID = {fid:.4f}")

            rows.append({
                "class": cls,
                "seed": seed,
                "fid_200v200": fid,
                "n_fakes": N_SAMPLE,
                "n_reals_avail": n_reals,
                "ckpt_iter": CKPT_ITER,
                "protocol_seed": PROTOCOL_SEED,
            })

    if not rows:
        print("[!] no FID computed — exiting")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] {len(rows)} rows -> {OUT_CSV}")

    # quick aggregation: mean ± std per class
    print("\nClass               s0       s1       s2       mean ± std")
    print("-" * 65)
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], {})[r["seed"]] = r["fid_200v200"]
    for cls in CLASSES:
        if cls in by_class:
            d = by_class[cls]
            vals = [d.get(s) for s in SEEDS if d.get(s) is not None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            s0 = f"{d.get(0, float('nan')):>7.3f}"
            s1 = f"{d.get(1, float('nan')):>7.3f}"
            s2 = f"{d.get(2, float('nan')):>7.3f}"
            print(f"  {cls:<18s} {s0} {s1} {s2}   {mean:6.2f} ± {std:5.2f}")

    grand = [r["fid_200v200"] for r in rows]
    print("-" * 65)
    print(f"  Grand mean ± std (N={len(grand)}): {np.mean(grand):.2f} ± {np.std(grand, ddof=1):.2f}")


if __name__ == "__main__":
    main()
