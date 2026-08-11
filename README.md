# FastGAN-64 — Cost–Quality Trade-off of Lightweight GANs for 64 × 64 Hematological Synthesis

Reference implementation, evaluation protocol and per-class results for the paper:

> **A Cost–Quality Trade-off Study of Lightweight Generative Adversarial Networks for 64 × 64 Hematological Synthesis**
> Rodrigo F. A. P. Oliveira and Carmelo J. A. Bastos Filho
> Escola Politécnica de Pernambuco, Universidade de Pernambuco, Recife, Brazil
>
> **Accepted for oral presentation at the 2026 IEEE Latin American Conference on Computational Intelligence (LA-CCI), Lima, Peru, November 2026.** To appear in the conference proceedings and on IEEE Xplore.

---

## What this repository is for

The paper compares three lightweight generative systems — OrGANiCells (a DCGAN-based baseline with R1 regularization), StyleGAN2-Ada, and a 64 × 64 adaptation of FastGAN — on BloodMNIST at 64 × 64, across 8 cell classes and 3 random seeds, under a single matched evaluation protocol. Its headline result is that StyleGAN2-Ada leaves the empirical cost–quality operating-point frontier once the third lightweight competitor is introduced.

Two artifacts of that study are released here:

1. **FastGAN-64** — a patch to [FastGAN-pytorch](https://github.com/odegeasslbc/FastGAN-pytorch) that makes the architecture trainable at `im_size=64`. The reference implementation hardcodes the `se_512` Generator block for `im_size >= 256`, so no public 64 × 64 instantiation existed.
2. **The matched evaluation protocol** — the FID 200v200 pipeline plus the robustness analyses added for the camera-ready version (KID, reference-subset resampling, mixed-effects model, nearest-neighbor memorization audit).

Everything needed to reproduce the numbers in the paper's tables is included.

## Layout

```
lacci_paths.py  root paths, driven by environment variables (no absolute paths anywhere)
patch/          models_64.py, train_64.py - Generator64/Discriminator64 and the 64 x 64 entry point
protocol/       fid_match200_{fastgan,organicells,stylegan2ada}.py - the matched FID 200v200 evaluation
analysis/       kid_and_resampling.py, hierarchical_model.py, memorization_nn.py
results/        per-class, per-seed CSVs behind every table in the paper
figures/        pareto_plot.py, qualitative_grid.py
```

## Setup

```bash
pip install -r requirements.txt

export BLOODMNIST_ROOT=/path/to/data    # $BLOODMNIST_ROOT/bloodmnist_<class>_64/raw/*.png
export RUNS_ROOT=/path/to/runs          # checkpoints and generated images
export OUT_ROOT=/path/to/outputs        # where analysis results are written
export FASTGAN_ROOT=/path/to/FastGAN-pytorch
```

Apply the patch by copying `patch/models_64.py` and `patch/train_64.py` into your
FastGAN-pytorch checkout; they sit alongside the upstream `models.py` and
`train.py` rather than replacing them.

## Reproducing the published numbers

The protocol is deterministic. For each `(class, seed)` pair, 200 synthetic images are generated from the EMA generator weights (`g_ema`) with a fixed noise seed, and compared against 200 real images sampled from the same class under `PROTOCOL_SEED = 42`. FID is computed with [`torch-fidelity`](https://github.com/toshas/torch-fidelity) on Inception-V3 features.

```bash
# 1. per-class FID, all seeds, one script per system
python protocol/fid_match200_fastgan.py
python protocol/fid_match200_organicells.py
python protocol/fid_match200_stylegan2ada.py

# 2. camera-ready robustness analyses (they reuse the same Inception features)
python analysis/kid_and_resampling.py --full
python analysis/hierarchical_model.py
python analysis/memorization_nn.py
```

The analysis scripts cache Inception-V3 features once and recompute the metrics from them, which keeps the resampling experiments cheap and guarantees that the reported FID values are bit-comparable with the published ones. A verification mode recomputes published FID values and reports the deviation:

```bash
python analysis/kid_and_resampling.py --verify
```

## Headline results

| System | Params | GFLOPs/img | Lat. (ms, b=1) | Lat. (ms, b=64) | VRAM (GB) | FID 200v200 | KID |
|---|---|---|---|---|---|---|---|
| OrGANiCells | 2.65 M | 0.77 | 0.58 | 0.31 | 0.43 | 254.21 ± 31.82 | 0.290 |
| StyleGAN2-Ada | 22.24 M | ~7.5 | 8.66 | 3.78 | 1.57 | 59.55 ± 15.85 | 0.0238 |
| **FastGAN-64** | 28.21 M | 7.57 | 5.07 | 1.22 | 0.84 | **45.29 ± 4.32** | **0.0080** |

All measurements on a single NVIDIA RTX 3050 8 GB. FID and KID are lower-is-better. The per-class ordering is identical under both metrics in 8/8 classes.

Parameters, FLOPs, latency and VRAM are generator-only, since the discriminator is not executed at inference. The FID tolerance is the standard deviation over all training runs of the system (40 for OrGANiCells, 24 for the other two).

Three entries supersede the submitted version of the paper, and the camera-ready response to the reviewers documents why: OrGANiCells latency (0.31, not 0.68 ms/img) and FastGAN-64 peak VRAM (0.84, not 0.78 GB) were re-measured under the matched protocol, and the OrGANiCells parameter count (2.65 M, not 4.2 M) is now generator-only like the other two rows. Re-run `analysis/latency_b1_b64.py` to reproduce the first two.

## Scope and caveats

These results are specific to one operating point: BloodMNIST at 64 × 64, 8 classes, 88 training runs (3 seeds per system, 5 for OrGANiCells), one consumer GPU. Each system is trained under its own reference recipe rather than a shared compute budget, so the reported gap is a property of published configurations, not of architectures at matched compute. FID and KID are computed on Inception-V3 features trained on natural images and therefore measure distributional similarity in a generic feature space, not clinical realism. Downstream classifier utility was not measured. See the Limitations section of the paper.

## Dataset

BloodMNIST, the 8-class peripheral blood cell subset of [MedMNIST v2](https://medmnist.com/), derived from the dataset of Acevedo et al. No images are redistributed here; the scripts expect the publicly released raw PNGs.

## Citation

```bibtex
@inproceedings{oliveira2026costquality,
  author    = {Oliveira, Rodrigo F. A. P. and Bastos Filho, Carmelo J. A.},
  title     = {A Cost--Quality Trade-off Study of Lightweight Generative
               Adversarial Networks for $64 \times 64$ Hematological Synthesis},
  booktitle = {2026 IEEE Latin American Conference on Computational Intelligence (LA-CCI)},
  year      = {2026},
  address   = {Lima, Peru},
  note      = {To appear}
}
```

Please also cite the original FastGAN paper (Liu et al., ICLR 2021), whose implementation this work patches, and MedMNIST v2 (Yang et al., 2023) for the dataset.

## License

The patch inherits the license of the upstream FastGAN-pytorch repository. The protocol, analysis and figure scripts are released under the MIT License.

## Contact

Rodrigo F. A. P. Oliveira — rfapo@ecomp.poli.br
