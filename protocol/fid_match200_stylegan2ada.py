"""FID 200v200 for the StyleGAN2-Ada generators, seeds 1 and 2.

Iterates the 8 classes under $RUNS_ROOT/stylegan_3seeds/ and scores each run
against 200 real images per class drawn with PROTOCOL_SEED=42. Seed 0 was scored
by the same procedure in an earlier run; its results ship in
results/fid_stylegan2ada_seed0.csv.
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import torch_fidelity

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lacci_paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, FASTGAN_ROOT  # noqa: E402


CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
SEEDS = [1, 2]  # seed 0 ja em a4_consolidated_match200.csv
RUNS_BASE = Path(str(RUNS_ROOT / "stylegan_3seeds"))
DATA_BASE = DATA_ROOT
N_SAMPLE = 200
PROTOCOL_SEED = 42

OUT_CSV = RUNS_BASE / "a4_b5_3seeds_fid_match200.csv"
TMP_BASE = RUNS_BASE / "tmp_fid_match200"


def main():
    if TMP_BASE.exists():
        shutil.rmtree(TMP_BASE)
    TMP_BASE.mkdir(parents=True)

    rows = []
    for cls in CLASSES:
        reals_dir = DATA_BASE / f"bloodmnist_{cls}_64" / "raw"
        if not reals_dir.exists():
            print(f"[!] {cls}: real dir absent ({reals_dir}) — SKIP all seeds")
            continue
        reals_all = sorted(reals_dir.glob("*.png"))
        if len(reals_all) < N_SAMPLE:
            print(f"[!] {cls}: reals={len(reals_all)} < {N_SAMPLE} — SKIP")
            continue

        for seed in SEEDS:
            name = f"{cls}_s{seed}"
            fakes_dir = RUNS_BASE / name / "fakes_kimg300_for_fid"
            if not fakes_dir.exists():
                print(f"[!] {name}: fakes dir absent — SKIP")
                continue
            fakes_all = sorted(fakes_dir.glob("*.png"))
            if len(fakes_all) < N_SAMPLE:
                print(f"[!] {name}: fakes={len(fakes_all)} < {N_SAMPLE} — SKIP")
                continue

            print(f"\n[+] {name}  fakes={len(fakes_all)}  reals={len(reals_all)}")
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
                fid=True, isc=False, kid=False, prc=False, verbose=False)
            fid = float(res["frechet_inception_distance"])
            print(f"    FID 200v200 = {fid:.4f}")

            rows.append({
                "class": cls,
                "seed": seed,
                "fid_200v200": fid,
                "n_fakes_avail": len(fakes_all),
                "n_reals_avail": len(reals_all),
                "kimg": 300,
                "protocol_seed": PROTOCOL_SEED,
            })

            # cleanup symlinks immediately to keep tmp small
            shutil.rmtree(ftmp, ignore_errors=True)
            shutil.rmtree(rtmp, ignore_errors=True)

    if not rows:
        print("[!] no FID computed — exiting")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[+] {len(rows)} rows -> {OUT_CSV}")

    # quick aggregation table
    print("\nClass               s1 FID   s2 FID")
    print("-" * 40)
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class"], {})[r["seed"]] = r["fid_200v200"]
    for cls in CLASSES:
        if cls in by_class:
            d = by_class[cls]
            print(f"  {cls:<18s} {d.get(1, float('nan')):>7.3f} {d.get(2, float('nan')):>9.3f}")


if __name__ == "__main__":
    main()
