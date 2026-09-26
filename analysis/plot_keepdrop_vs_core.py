"""Figure 1(b): keep-drop accuracy vs. 1B DCLM Core score.

Keep-drop accuracy is the mean of the share of worth-keeping pages kept and the share of
junk pages removed, over the 5,000 held-out pages labeled keep or remove by the gpt-oss-120b
keep-or-drop judge. A junk page counts as removed when the pipeline drops it or keeps it only as
text that the output judge (evaluation/judges/output_judge.py) rates worth keeping; the same rule
applies to every pipeline. Values: the "keepdrop_fig1b" block of evaluation/judges/rule_motivation.py
run with the output judge, rounded to one decimal as in the paper.
Core scores are the 1B rows of the main results table (Table 2).

Save target: $FIG_DIR/keepdrop_vs_core.pdf (FIG_DIR defaults to analysis/figures).
"""

import os
from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"

MEDIUM_SIZE = 46
SMALLER_SIZE = 42
plt.rc("font", size=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALLER_SIZE)
plt.rc("ytick", labelsize=SMALLER_SIZE)
plt.rc("figure", titlesize=MEDIUM_SIZE)
plt.rc("legend", fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 7
FIG_WIDTH = 7

palette = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = palette

# (keep-drop accuracy %, 1B Core)
points = {
    "ReScraper": (85.7, 0.27348),
    "UltraX": (76.8, 0.26152),
    "ProX-C": (71.9, 0.23391),
    "RefinedWeb-rule": (67.0, 0.25340),
    "FineWeb-rule": (63.9, 0.23022),
}
# name: (color, marker, size, name_dx, name_dy, name_ha, value_dy, value_va)
style = {
    "ReScraper": (RED, "*", 1400, -1.6, 0.0, "right", 0.0036, "bottom"),
    "UltraX": (TEAL, "^", 800, 1.3, 0.0, "left", 0.0032, "bottom"),
    "RefinedWeb-rule": (NAVY, "o", 800, 1.3, 0.0, "left", 0.0032, "bottom"),
    "ProX-C": (GREEN, "s", 600, 1.3, 0.0, "left", 0.0032, "bottom"),
    "FineWeb-rule": (YELLOW, "D", 600, 1.3, -0.0009, "left", -0.0032, "top"),
}

# Square 5.6 in plot box, same size in both Figure 1 panels.
fig = plt.figure(figsize=(8.20, 7.40))
ax = fig.add_axes([0.2439, 0.2095, 0.6829, 0.7568])
for name, (acc, core) in points.items():
    color, marker, size, ndx, ndy, nha, vdy, vva = style[name]
    r, g, b = matplotlib.colors.to_rgb(color)
    ax.scatter(
        acc, core, marker=marker, s=size,
        color=(r, g, b, 0.3), edgecolor=(r, g, b, 1.0), linewidths=2, zorder=3,
    )
    ax.text(
        acc, core + vdy, f"{core:.3f}", horizontalalignment="center", verticalalignment=vva,
        color=color, fontsize=35, fontweight="bold" if name == "ReScraper" else "normal", zorder=4,
    )
    ax.text(
        acc + ndx, core + ndy, name, horizontalalignment=nha, verticalalignment="center",
        color=color, fontsize=35, fontweight="bold" if name == "ReScraper" else "normal", zorder=4,
    )

ax.set_xlabel("Keep–drop accuracy (%)")
ax.set_ylabel("Avg. Downstream Acc.")
ax.set_xlim(61, 89)
ax.set_ylim(0.2205, 0.2845)
ax.set_xticks([65, 70, 75, 80, 85])
ax.set_yticks([0.23, 0.24, 0.25, 0.26, 0.27])
ax.grid(alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)

FIG_DIR = Path(os.environ.get("FIG_DIR", Path(__file__).resolve().parent / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "keepdrop_vs_core.pdf")
for name, (acc, core) in points.items():
    print(f"{name:16s} keep-drop accuracy {acc:.1f}  core {core:.5f}")
