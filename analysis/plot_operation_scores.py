#!/usr/bin/env python3
"""
Figure 4 (fig:operation-scores). Mean DataMan and FineWeb-Edu score of the held-out pages, grouped by the operation ReScraper
chose. One frame: DataMan on the left half (left axis), FineWeb-Edu on the right half (right
axis). Hatched outline = before the operation (the page after extraction), filled bar = after;
the two are overlaid at one position. Delete: before = the pages ReScraper deletes, after = every
page it keeps, as extracted (keep + edit before + rewrite before).

Data: $FIG_DATA_DIR/operation_scores.json (per-document scores; FIG_DATA_DIR defaults to
analysis/data); another file can be given with --data.
Drawn at print size (the figure sits at half the text width, next to the operation-mix figure), so
font sizes are the printed sizes. Operation colors follow the method overview figure.

Save target: $FIG_DIR/operation_scores.pdf (FIG_DIR defaults to analysis/figures)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
DATA_PATH = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else DATA_DIR / "operation_scores.json"
FIG_PATH = FIG_DIR / "operation_scores.pdf"

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

PALETTE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = PALETTE
face_alpha = 0.3
LW = 0.8

G = json.loads(DATA_PATH.read_text())["groups"]
mean = lambda x: sum(x) / len(x)
kept = {m: G["keep"][m] + G["edit_before"][m] + G["rewrite_before"][m] for m in ("dataman", "fineweb_edu")}
groups = [("Keep", GREEN, None, G["keep"]),
          ("Delete", RED, G["delete"], kept),
          ("Edit", YELLOW, G["edit_before"], G["edit_after"]),
          ("Rewrite", BLUE, G["rewrite_before"], G["rewrite_after"])]

# Same canvas and frame margins as plot_operation_mix.py, so the two figures align side by side.
CANVAS = (2.8, 1.55)
FRAME = dict(top=1 - 0.14 / 1.55, bottom=0.34 / 1.55, left=0.36 / 2.8, right=1 - 0.40 / 2.8)
fig, axl = plt.subplots(figsize=CANVAS)
axr = axl.twinx()
W = 0.7
GAP = 0.3  # empty slot between the two halves (narrow, so the 6.8 pt operation labels keep a gap)
halves = ((axl, "dataman", "DataMan", 3.0, 4.8, 0.0),  # tops leave room for the three-line labels
          (axr, "fineweb_edu", "FineWeb-Edu", 0.0, 1.8, len(groups) + GAP))
for ax, m, name, lo, top, x0 in halves:
    for i, (lab, c, before, after) in enumerate(groups):
        x = x0 + i
        va = mean(after[m])
        ax.bar(x, va - lo, bottom=lo, width=W, facecolor=(*mpl.colors.to_rgb(c), face_alpha),
               edgecolor=c, linewidth=LW, zorder=3)
        txt, hi = f"{va:.2f}", va
        if before is not None:
            vb = mean(before[m])
            ax.bar(x, vb - lo, bottom=lo, width=W, facecolor="none", edgecolor=c, hatch="/////",
                   linewidth=LW, linestyle=(0, (3, 1.5)), zorder=4)
            # Higher value on top, arrow pointing from before to after.
            txt = (f"{va:.2f}\n$\\uparrow$\n{vb:.2f}" if va >= vb else f"{vb:.2f}\n$\\downarrow$\n{va:.2f}")
            hi = max(va, vb)
        ax.text(x, hi + (top - lo) * 0.015, txt, ha="center", va="bottom", fontsize=6.5,
                color="#222222", linespacing=0.85)
    ax.set_ylim(lo, top)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="y", labelsize=7.5, length=2, pad=1.5)
axl.set_ylabel("DataMan", labelpad=2)
axr.set_ylabel("FineWeb-Edu", labelpad=2)
axr.grid(False)
axl.grid(axis="x", visible=False)
axl.set_axisbelow(True)
ends = (-0.5, 2 * len(groups) + GAP - 0.5)
axl.set_xlim(*ends)
axl.axvline(len(groups) - 0.5 + GAP / 2, color="#999999", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
xt = [i for i in range(len(groups))] + [len(groups) + GAP + i for i in range(len(groups))]
axl.set_xticks(xt, [g[0] for g in groups] * 2, fontsize=6.8)
axl.tick_params(axis="x", length=0, pad=2)
axl.set_xlabel("Operation chosen by ReScraper", fontsize=9.5, labelpad=2)
for ax in (axl, axr):
    for sp in ax.spines.values():
        sp.set_color("#666666")
        sp.set_linewidth(0.6)
axl.legend([Patch(facecolor="none", edgecolor="#555555", hatch="/////", linewidth=LW, linestyle=(0, (3, 1.5))),
            Patch(facecolor=(0.4, 0.4, 0.4, face_alpha), edgecolor="#555555", linewidth=LW)],
           ["Before", "After"], loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.0), fontsize=7.4, handlelength=1.0, handleheight=0.55,
           columnspacing=0.9, handletextpad=0.3, borderaxespad=0.1)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.subplots_adjust(**FRAME)
mpl.rcParams["savefig.bbox"] = "standard"  # keep the fixed canvas (no tight crop)
plt.savefig(FIG_PATH, bbox_inches=None)
if "--png" in sys.argv:
    plt.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=300, bbox_inches=None)
print(f"Saved to {FIG_PATH}")
