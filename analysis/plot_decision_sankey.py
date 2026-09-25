#!/usr/bin/env python3
"""
Operation flow: teacher operation (left) -> ReScraper operation (right) on the 5,000 held-out
pages (Figure 7, fig:decision-flow). Each band is the share of pages the teacher assigned to the left operation
and the student routed to the right one; bands take the teacher's color.

Data: $FIG_DATA_DIR/decision_flow.json (the held-out confusion matrix; FIG_DATA_DIR defaults to
analysis/data); another file can be given with --data.
Style mirrors plot_scraper_comparison.py: shared palette, serif fonts, transparent fill
(face_alpha=0.3) + colored edge, dark ink for all text. Operation colors follow the method overview
figure: keep green, edit gold, delete red, rewrite blue.

Save target: $FIG_DIR/decision_sankey_final.pdf (FIG_DIR defaults to analysis/figures)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
DATA_PATH = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else DATA_DIR / "decision_flow.json"
FIG_PATH = FIG_DIR / "decision_sankey_final.pdf"

# ---------- Font sizes ----------
MEDIUM_SIZE = 34
SMALLER_SIZE = 29

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": MEDIUM_SIZE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ---------- Palette ----------
PALETTE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = PALETTE
OP_COLOR = {"keep": GREEN, "edit": YELLOW, "delete": RED, "rewrite": BLUE}
OP_LABEL = {"keep": "Keep", "edit": "Edit", "delete": "Delete", "rewrite": "Rewrite"}
INK = "#222222"

face_alpha = 0.3   # node fill, as in the bar figures
band_alpha = 0.25  # flow bands
edge_width = 2.0

GAP_FRAC = 0.035    # vertical gap between nodes, as a share of all pages
NODE_W = 0.03       # node width, in axis units
X_L, X_R = 0.25, 0.75
MIN_LABEL_FRAC = 0.025  # bands below this share of all pages carry no label


def band(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color):
    """S-shaped band from the left interval to the right interval (y grows downward)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0_top), (xm, y0_top), (xm, y1_top), (x1, y1_top),
             (x1, y1_bot), (xm, y1_bot), (xm, y0_bot), (x0, y0_bot), (x0, y0_top)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=band_alpha,
                           edgecolor="none", zorder=1))


data = json.loads(DATA_PATH.read_text())
ops = data["operations"]
flows = data["teacher_to_student"]
n = data["pages"]
GAP = GAP_FRAC * n
MIN_LABEL = MIN_LABEL_FRAC * n
left_tot = {s: sum(flows[s].values()) for s in ops}
right_tot = {t: sum(flows[s][t] for s in ops) for t in ops}
assert sum(left_tot.values()) == sum(right_tot.values()) == n


def pct(v):
    # every number drawn is a share of all held-out pages, so labels match band widths
    return f"{100 * v / n:.1f}%"


def stack(tot):
    y, pos = 0, {}
    for op in ops:
        pos[op] = y
        y += tot[op] + GAP
    return pos


lpos, rpos = stack(left_tot), stack(right_tot)
height = n + GAP * (len(ops) - 1)

fig, ax = plt.subplots(figsize=(10.0, 5.9))  # flatter, so the figure is shorter at its column width
ax.set_xlim(0, 1)
ax.set_ylim(height + 0.008 * n, -0.008 * n)
ax.axis("off")

# Bands leave each teacher node in target order and enter each student node in source order,
# which minimises crossings. One label per band, at its source end.
lcur, rcur = dict(lpos), dict(rpos)
for s in ops:
    for t in ops:
        v = flows[s][t]
        if not v:
            continue
        band(ax, X_L + NODE_W, lcur[s], lcur[s] + v, X_R, rcur[t], rcur[t] + v, OP_COLOR[s])
        if v >= MIN_LABEL:
            ax.text(X_L + NODE_W + 0.008, lcur[s] + v / 2, pct(v), ha="left", va="center",
                    fontsize=SMALLER_SIZE - 5, color=INK,
                    fontweight="bold" if s == t else "normal", zorder=3)
        lcur[s] += v
        rcur[t] += v

for op in ops:
    for x, pos, tot, left in ((X_L, lpos, left_tot, True), (X_R, rpos, right_tot, False)):
        ax.add_patch(Rectangle((x, pos[op]), NODE_W, tot[op],
                               facecolor=(*mpl.colors.to_rgb(OP_COLOR[op]), face_alpha),
                               edgecolor=OP_COLOR[op], linewidth=edge_width, zorder=2))
        # the right column shares the left column's colours, so it carries only the shares
        label = f"{OP_LABEL[op]}  {pct(tot[op])}" if left else pct(tot[op])
        if left:
            ax.text(x - 0.012, pos[op] + tot[op] / 2, label, ha="right", va="center",
                    fontsize=SMALLER_SIZE, color=INK)
        else:
            ax.text(x + NODE_W + 0.012, pos[op] + tot[op] / 2, label, ha="left", va="center",
                    fontsize=SMALLER_SIZE, color=INK)

ax.text(X_L + NODE_W / 2, -0.017 * n, "Teacher", ha="center", va="bottom", fontsize=MEDIUM_SIZE, color=INK)
ax.text(X_R + NODE_W / 2, -0.017 * n, "ReScraper", ha="center", va="bottom", fontsize=MEDIUM_SIZE, color=INK)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, bbox_inches="tight", pad_inches=0.05)
if "--png" in sys.argv:
    plt.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=80)
agree = sum(flows[o][o] for o in ops)
print(f"Saved to {FIG_PATH}  agreement {agree}/{n} = {100 * agree / n:.2f}%")
plt.close()
