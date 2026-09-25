#!/usr/bin/env python3
"""
Figure 5 (fig:operation-mix). Operation distribution of the four model-based pipelines on the same pages, one horizontal
stacked bar per system: ProX-C, UltraX, DataOrchestra, ReScraper. Pages a system could not process are drawn as Delete. Pages are classified by what changes in the output
text, with one rule for all systems; Edit is split by the share of the page's words it removes
(below 10%, 10-50%, above 50%).

Data: $FIG_DATA_DIR/operation_mix.json (FIG_DATA_DIR defaults to analysis/data); another file can
be given with --data.
Drawn at print size (the figure sits at half the text width, next to the operation-scores figure), so
font sizes are the printed sizes.
Style mirrors plot_scraper_comparison.py: shared palette, serif fonts,
transparent fill (face_alpha=0.3) + colored edge, bold in-bar shares. Operation colors follow
the method overview figure and plot_decision_sankey.py: keep green, edit gold, delete red, rewrite blue. The three Edit classes share the gold and darken
with the share removed.

Save target: $FIG_DIR/operation_mix.pdf (FIG_DIR defaults to analysis/figures)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches  # noqa: F401  (mpl.patches.Patch)
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
DATA_PATH = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else DATA_DIR / "operation_mix.json"
FIG_PATH = FIG_DIR / "operation_mix.pdf"

# ---------- Font sizes ----------
MEDIUM_SIZE = 9
SMALLER_SIZE = 8

plt.rc("font", size=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALLER_SIZE)
plt.rc("ytick", labelsize=MEDIUM_SIZE)
plt.rc("legend", fontsize=SMALLER_SIZE)

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# ---------- Palette ----------
PALETTE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = PALETTE
OP_COLOR = {"keep": GREEN, "edit_lt10": YELLOW, "edit_gt10": YELLOW, "delete": RED, "rewrite": BLUE}
OP_ALPHA = {"edit_lt10": 0.15, "edit_gt10": 0.6}  # sequential: more removed, darker
OP_LABEL = {"keep": "Keep", "edit_lt10": "Edit <10%", "edit_gt10": ">10%",
            "delete": "Delete", "rewrite": "Rewrite"}
face_alpha = 0.3
MIN_LABEL = 5.0  # segments thinner than this share (%) are not labelled; all others carry their label inside
LW = 0.8
# Printed sizes of the earlier column version (12 x 12 in canvas at half the text width).
BAR_LABEL, TICK_SYS, TICK_SHARE, LEGEND = 7.3, 8.2, 9.5, 7.3

data = json.loads(DATA_PATH.read_text())
# Pages a system could not process (ReScraper: HTML that fails to parse or a rendering longer than
# its context) produce no output and never reach the corpus, so they are drawn as Delete; every
# system keeps the same 7,000-page denominator. The data file keeps them as their own count.
# The data file keeps three Edit bins; the figure merges 10-50% and >50% into one Edit >10% class.
ops = ["keep", "edit_lt10", "edit_gt10", "delete", "rewrite"]
systems = ["ProX-C", "UltraX", "DataOrchestra", "ReScraper"]


def counts(name):
    c = dict(data["systems"][name]["counts"])
    c["delete"] += c.pop("not_processed", 0)
    c["edit_gt10"] = c.pop("edit_10_50") + c.pop("edit_gt50")
    return c


# One bar per system, top to bottom in reading order.
# Same canvas and top/bottom margins as plot_operation_scores.py, so the two figures align side by side.
CANVAS = (2.8, 1.55)
FRAME = dict(top=1 - 0.14 / 1.55, bottom=0.34 / 1.55, left=0.71 / 2.8, right=1 - 0.13 / 2.8)  # left fits "DataOrchestra"
LEGEND_FS = 7.4  # largest size at which the one-row legend, centered over the frame, fits the canvas
fig, ax = plt.subplots(figsize=CANVAS)
BH = 0.56  # bar height; the gap above each bar holds the labels of thin segments
for y, name in enumerate(systems):
    c = counts(name)
    denom = sum(c.values())
    left = 0.0
    for op in ops:  # stacked left to right in the legend's reading order
        share = 100.0 * c[op] / denom
        if share == 0:
            continue
        ax.barh(y, share, left=left, height=BH,
                facecolor=(*mpl.colors.to_rgb(OP_COLOR[op]), OP_ALPHA.get(op, face_alpha)),
                edgecolor=OP_COLOR[op], linewidth=LW, zorder=3)
        if share >= MIN_LABEL:
            ax.text(left + share / 2, y, f"{share:.1f}", ha="center", va="center", color="#222222",
                    fontsize=BAR_LABEL, fontweight="bold", zorder=4)
        left += share

ax.set_yticks(range(len(systems)), systems)
ax.set_ylim(len(systems) - 0.55, -0.75)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of pages (%)", labelpad=1.5, fontsize=TICK_SHARE)
ax.tick_params(axis="x", labelsize=TICK_SHARE, length=2, pad=1.5)
ax.tick_params(axis="y", labelsize=TICK_SYS, length=0, pad=2)
ax.grid(axis="y", visible=False)
handles = {op: mpl.patches.Patch(facecolor=(*mpl.colors.to_rgb(OP_COLOR[op]), OP_ALPHA.get(op, face_alpha)),
                                 edgecolor=OP_COLOR[op], linewidth=LW, label=OP_LABEL[op]) for op in ops}
# One row centered over the plot frame, in the bars' left-to-right order.
ax_c = (FRAME["left"] + FRAME["right"]) / 2
fig.legend(handles=[handles[op] for op in ops], loc="lower center", bbox_to_anchor=(ax_c, FRAME["top"]),
           ncol=len(ops), frameon=False, handlelength=0.6, handleheight=0.55, columnspacing=0.4,
           handletextpad=0.2, fontsize=LEGEND_FS, borderaxespad=0.1)
for spine in ax.spines.values():
    spine.set_color("#666666")
    spine.set_linewidth(0.6)
ax.set_axisbelow(True)
ax.tick_params(colors="#333333")

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(**FRAME)
mpl.rcParams["savefig.bbox"] = "standard"  # keep the fixed canvas (no tight crop)
plt.savefig(FIG_PATH, bbox_inches=None)
if "--png" in sys.argv:
    plt.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=300, bbox_inches=None)
print(f"Saved to {FIG_PATH}")
plt.close()
