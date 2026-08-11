"""Cost-quality trade-off figure (FID vs. inference latency).

Axes: latency (ms/img, log scale) against FID 200v200. The dashed segment joins
the two non-dominated systems (OrGANiCells and FastGAN-64); StyleGAN2-Ada is
marked as dominated.

Colours have well-separated luminances and each system has a distinct marker, so
the figure survives black-and-white printing.

Output: $P4_OUT_DIR/p4_pareto_chart.{png,pdf}
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Camera-ready IEEE: embed TrueType fonts (fonttype 42). The matplotlib
# default emits Type 3, which IEEE PDF eXpress rejects.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Output directory can be overridden with P4_OUT_DIR
if "P4_OUT_DIR" in os.environ:
    OUT_DIR = Path(os.environ["P4_OUT_DIR"])
else:
    ROOT = Path(__file__).resolve().parents[2]
    OUT_DIR = ROOT / "outputs" / "fastgan_p3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Well-separated luminances (~25%, ~55%, ~5% grey) plus distinct markers,
# so the figure stays readable when printed in black and white.
SYSTEMS = {
    "OrGANiCells":      {"lat": 0.309, "fid": 254.21, "std": 31.82, "color": "#2E7D32", "marker": "o", "pareto": True},
    "StyleGAN2-Ada":    {"lat": 3.780, "fid": 59.55, "std": 15.85, "color": "#9E9E9E", "marker": "X", "pareto": False},
    "FastGAN-64 (adapted)": {"lat": 1.215, "fid": 45.29, "std": 4.32, "color": "#0D2C54", "marker": "s", "pareto": True},
}

fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)

pareto_pts = sorted([(s["lat"], s["fid"]) for s in SYSTEMS.values() if s["pareto"]])
xs, ys = zip(*pareto_pts)
ax.plot(xs, ys, "--", color="#E65100", linewidth=1.5, alpha=0.7,
        label="Non-dominated segment", zorder=2)

for name, s in SYSTEMS.items():
    label = name + ("" if s["pareto"] else " (dominated)")
    ax.errorbar(
        s["lat"], s["fid"], yerr=s["std"],
        fmt=s["marker"], color=s["color"], markersize=11,
        markeredgecolor="black", markeredgewidth=0.6,
        ecolor=s["color"], elinewidth=1.2, capsize=4,
        label=label, zorder=3,
    )

ax.annotate("OrGANiCells\n2.65 M params\n0.77 GFLOPs",
            xy=(0.309, 254.21), xytext=(0.30, 205),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="-", color="#2E7D32", lw=0.6, alpha=0.6))
ax.annotate("FastGAN-64\n28.21 M params\n7.57 GFLOPs",
            xy=(1.215, 45.29), xytext=(1.35, 8),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="-", color="#0D2C54", lw=0.6, alpha=0.6))
ax.annotate("StyleGAN2-Ada\n22.24 M params\n~7.5 GFLOPs",
            xy=(3.79, 59.6), xytext=(5.2, 105),
            fontsize=8, ha="left", color="#424242",
            arrowprops=dict(arrowstyle="-", color="#9E9E9E", lw=0.6, alpha=0.6))

ax.set_xscale("log")
ax.set_xlabel("Latency (ms/img, b=64 amortized, log scale)", fontsize=10)
ax.set_ylabel("FID 200v200 (lower is better)", fontsize=10)
ax.set_title("Cost-quality trade-off @ BloodMNIST 64x64\n(8 classes; lower-left = better)",
             fontsize=11)

ax.set_xlim(0.3, 20)
ax.set_ylim(0, 320)
ax.grid(True, which="both", linestyle=":", alpha=0.35)
ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

plt.tight_layout()

png = OUT_DIR / "p4_pareto_chart.png"
pdf = OUT_DIR / "p4_pareto_chart.pdf"
plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
print("saved:", png)
print("saved:", pdf)
