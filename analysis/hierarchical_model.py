#!/usr/bin/env python3
"""Mixed-effects model and paired tests on KID (camera-ready analysis).

Added in response to a reviewer observation that the 8 BloodMNIST classes are
levels of one dataset rather than independent replications, so paired tests over
n=8 classes cannot speak to run-to-run reliability.

Model: FID ~ system, with a random intercept per class. Seed-to-seed variation
enters as residual, since every row is one training run (class, seed).

Inputs are the published per-run CSVs in results/; KID comes from the output of
analysis/kid_and_resampling.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "camera_ready_hierarchical.md"


def load_fid() -> pd.DataFrame:
    frames = []
    o = pd.read_csv(ROOT / "outputs/et07_synthetic/p4_organicells_fid_match200.csv")
    frames.append(o.assign(system="OrGANiCells")[["system", "class", "seed", "fid_200v200"]])

    s0 = pd.read_csv(ROOT / "outputs/a4_b5/a4_consolidated_match200.csv")
    s0 = s0.rename(columns={"fid_external_200v200": "fid_200v200"})
    s0["seed"] = 0
    frames.append(s0.assign(system="StyleGAN2-Ada")[["system", "class", "seed", "fid_200v200"]])

    s12 = pd.read_csv(ROOT / "outputs/a4_b5/a4_b5_3seeds_fid_match200.csv")
    frames.append(s12.assign(system="StyleGAN2-Ada")[["system", "class", "seed", "fid_200v200"]])

    f = pd.read_csv(ROOT / "outputs/fastgan_p3/p4_fastgan_fid_match200.csv")
    frames.append(f.assign(system="FastGAN-64")[["system", "class", "seed", "fid_200v200"]])

    return pd.concat(frames, ignore_index=True).rename(columns={"fid_200v200": "fid"})


def main():
    df = load_fid()
    lines = ["# Hierarchical analysis", ""]
    lines.append(f"Observations: {len(df)} training runs (class x seed x system).")
    counts = df.groupby("system").size().to_dict()
    lines.append(f"Per system: {counts}")
    lines.append("")

    # ---- mixed model: random intercept per class -------------------------
    df["system"] = pd.Categorical(
        df["system"], categories=["FastGAN-64", "StyleGAN2-Ada", "OrGANiCells"])
    model = smf.mixedlm("fid ~ C(system)", df, groups=df["class"])
    res = model.fit(reml=True, method="lbfgs")

    lines.append("## Mixed model  fid ~ system + (1|class)")
    lines.append("")
    lines.append("| term | coef | SE | z | p | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    ci = res.conf_int()
    for name in res.params.index:
        if name == "Group Var":
            continue
        lo, hi = ci.loc[name]
        lines.append(f"| {name} | {res.params[name]:.2f} | {res.bse[name]:.2f} | "
                     f"{res.tvalues[name]:.2f} | {res.pvalues[name]:.3g} | "
                     f"[{lo:.2f}, {hi:.2f}] |")
    gv = float(res.cov_re.iloc[0, 0])
    lines.append("")
    lines.append(f"- Class intercept variance: {gv:.2f} "
                 f"(SD {np.sqrt(gv):.2f} FID)")
    lines.append(f"- Residual (seed-to-seed) variance: {res.scale:.2f} "
                 f"(SD {np.sqrt(res.scale):.2f} FID)")
    icc = gv / (gv + res.scale)
    lines.append(f"- ICC (share of variance attributable to class): {icc:.3f}")
    lines.append("")

    # contrast StyleGAN2-Ada vs FastGAN-64 (reference level = FastGAN-64)
    key = [n for n in res.params.index if "StyleGAN2-Ada" in n][0]
    lo, hi = ci.loc[key]
    lines.append(f"Contrast StyleGAN2-Ada - FastGAN-64: "
                 f"{res.params[key]:.2f} FID [IC95% {lo:.2f}, {hi:.2f}], "
                 f"p = {res.pvalues[key]:.3g}.")
    lines.append("")

    # ---- KID: same paired tests, per class ------------------------------
    kid = pd.read_csv(ROOT / "outputs/camera_ready_kid_ci_byseed.csv")
    piv = kid.groupby(["class", "system"]).kid_protocol.mean().unstack()
    lines.append("## KID per class (mean over seeds)")
    lines.append("")
    lines.append("| pair | mean delta KID | Wilcoxon p | wins |")
    lines.append("|---|---|---|---|")
    pairs = [("OrGANiCells", "StyleGAN2-Ada"), ("OrGANiCells", "FastGAN-64"),
             ("StyleGAN2-Ada", "FastGAN-64")]
    for a, b in pairs:
        d = piv[a] - piv[b]
        w = stats.wilcoxon(piv[a], piv[b], alternative="two-sided",
                           method="exact")
        lines.append(f"| {a} vs {b} | {d.mean():+.5f} | {w.pvalue:.4f} | "
                     f"{int((d > 0).sum())}/8 |")
    lines.append("")

    fidpiv = kid.groupby(["class", "system"]).fid_protocol.mean().unstack()
    agree = (fidpiv.rank(axis=1).values == piv.rank(axis=1).values).all(axis=1)
    tau = stats.kendalltau(fidpiv.values.ravel(), piv.values.ravel())
    lines.append(f"- Per-class ordering identical under FID and KID: "
                 f"{int(agree.sum())}/{len(agree)} classes.")
    lines.append(f"- Kendall tau between FID and KID (24 class x system cells): "
                 f"{tau.statistic:.3f} (p = {tau.pvalue:.2e}).")
    lines.append("")

    # ---- sensitivity to the real reference subset -----------------------
    lines.append("## Sensitivity to the real reference subset (20 redraws)")
    lines.append("")
    lines.append("| system | FID SD across redraws | mean 95% interval width |")
    lines.append("|---|---|---|")
    for s, g in kid.groupby("system", sort=False):
        width = (g.fid_ci_hi - g.fid_ci_lo).mean()
        lines.append(f"| {s} | {g.fid_redraw_std.mean():.2f} | {width:.2f} |")

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[ok] -> {OUT}")


if __name__ == "__main__":
    main()
