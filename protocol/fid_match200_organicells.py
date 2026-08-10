"""FID 200v200 for the OrGANiCells generators (8 classes x 5 seeds).

Reuses the synthetic images already written under $OUT_ROOT/et07_synthetic/ and
scores them against 200 real images per class drawn with PROTOCOL_SEED=42, the
same tuple used for StyleGAN2-Ada and FastGAN-64.

Output: $OUT_ROOT/et07_synthetic/p4_organicells_fid_match200.csv
"""
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch_fidelity

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lacci_paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, FASTGAN_ROOT  # noqa: E402


CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
SEEDS = [0, 1, 2, 3, 4]

FAKES_BASE = Path(str(OUT_ROOT / "et07_synthetic"))
DATA_BASE = DATA_ROOT
OUT_CSV = FAKES_BASE / "p4_organicells_fid_match200.csv"
TMP_BASE = Path(str(OUT_ROOT.parent / "outputs/et07_synthetic/tmp_fid"))

N_SAMPLE = 200
PROTOCOL_SEED = 42


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
            print(f"[!] {cls}: reals dir absent — SKIP")
            continue
        reals_all = sorted(reals_dir.glob("*.png"))
        if len(reals_all) < N_SAMPLE:
            print(f"[!] {cls}: reals={len(reals_all)} < 200 — SKIP")
            continue

        for seed in SEEDS:
            idx += 1
            name = f"{cls}_s{seed}"
            fakes_all = sorted((FAKES_BASE / cls).glob(f"g_s{seed}_*.png"))
            if len(fakes_all) < N_SAMPLE:
                print(f"[{idx}/{total}] {name}: fakes={len(fakes_all)} < 200 — SKIP")
                continue

            print(f"[{idx}/{total}] {name}  fakes={len(fakes_all)}  reals={len(reals_all)}")
            rng = random.Random(PROTOCOL_SEED)
            fk = rng.sample(fakes_all, N_SAMPLE)
            rl = rng.sample(reals_all, N_SAMPLE)

            ftmp = TMP_BASE / f"f_{name}"
            rtmp = TMP_BASE / f"r_{name}"
            for d in (ftmp, rtmp):
                d.mkdir(parents=True, exist_ok=True)
            for src in fk:
                (ftmp / src.name).symlink_to(src)
            for src in rl:
                (rtmp / src.name).symlink_to(src)

            res = torch_fidelity.calculate_metrics(
                input1=str(ftmp), input2=str(rtmp),
                fid=True, isc=False, kid=False, prc=False, verbose=False,
            )
            fid = float(res["frechet_inception_distance"])
            print(f"    FID 200v200 = {fid:.4f}")

            rows.append({
                "class": cls, "seed": seed, "fid_200v200": fid,
                "n_fakes": N_SAMPLE, "n_reals_avail": len(reals_all),
                "protocol_seed": PROTOCOL_SEED,
            })

            shutil.rmtree(ftmp, ignore_errors=True)
            shutil.rmtree(rtmp, ignore_errors=True)

    if not rows:
        sys.exit(1)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] {len(rows)} rows -> {OUT_CSV}")

    print("\nClass               s0       s1       s2       s3       s4       mean ± std")
    print("-" * 80)
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
            cells = " ".join(f"{d.get(s, float('nan')):>7.2f}" for s in SEEDS)
            print(f"  {cls:<18s} {cells}   {mean:6.2f} ± {std:5.2f}")
    grand = [r["fid_200v200"] for r in rows]
    print("-" * 80)
    print(f"  Grand mean ± std (N={len(grand)}): {np.mean(grand):.2f} ± {np.std(grand, ddof=1):.2f}")


if __name__ == "__main__":
    main()
