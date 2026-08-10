#!/usr/bin/env python3
"""Qualitative figure: real vs. generated samples at 64 x 64.

Added in response to a reviewer request for representative generated samples.

A 4 x 4 grid: rows are the real data and the three systems, columns are four
BloodMNIST classes chosen to span the morphological range described in the
paper -- round nucleus (lymphocyte), segmented nucleus (neutrophil), granular
cytoplasm (basophil) and anuclear fragment (platelet).

Samples are drawn deterministically (Random(42)) from the same image pools used
for the quantitative evaluation.
"""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lacci_paths import DATA_ROOT, RUNS_ROOT, OUT_ROOT, FASTGAN_ROOT  # noqa: E402


DATA = DATA_ROOT
RUNS = RUNS_ROOT
ET07 = Path(str(OUT_ROOT / "et07_synthetic"))
OUT = OUT_ROOT / "camera_ready"

CLASSES = [("basophil", "Basophil"), ("lymphocyte", "Lymphocyte"),
           ("neutrophil", "Neutrophil"), ("platelet", "Platelet")]
SEED = 0
PICK_SEED = 42


def real_dir(cls):
    return sorted((DATA / f"bloodmnist_{cls}_64" / "raw").glob("*.png"))


def organ_dir(cls):
    return sorted((ET07 / cls).glob(f"g_s{SEED}_*.png"))


def stylegan_dir(cls):
    base = (RUNS / "stylegan_basophil_64" if cls == "basophil"
            else RUNS / "stylegan_a4" / cls)
    return sorted((base / "fakes_kimg300_for_fid").glob("*.png"))


def fastgan_dir(cls):
    return sorted((RUNS / "fastgan_p3" / "train_results" / f"{cls}_s{SEED}"
                   / "fakes_for_fid").glob("*.png"))


ROWS = [("Real", real_dir), ("OrGANiCells", organ_dir),
        ("StyleGAN2-Ada", stylegan_dir), ("FastGAN-64", fastgan_dir)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(ROWS), len(CLASSES),
                             figsize=(3.4, 3.6), dpi=300)

    for r, (row_name, lister) in enumerate(ROWS):
        for c, (cls, disp) in enumerate(CLASSES):
            ax = axes[r][c]
            files = lister(cls)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if not files:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=6)
            else:
                pick = random.Random(PICK_SEED + r * 10 + c).choice(files)
                ax.imshow(Image.open(pick).convert("RGB"))
            if r == 0:
                ax.set_title(disp, fontsize=6.5, pad=2.5)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=6.5, rotation=90,
                              va="center", labelpad=3)

    plt.tight_layout(pad=0.25, h_pad=0.35, w_pad=0.25)
    for ext in ("pdf", "png"):
        p = OUT / f"cr_qualitative_grid.{ext}"
        plt.savefig(p, bbox_inches="tight", dpi=300)
        print("saved:", p)


if __name__ == "__main__":
    main()
