#!/usr/bin/env python3
"""
Figure 6 (fig:quality-diversity): quality by bucket (left 2 x 3 grid) and diversity (right column)
in one figure.

Right column, top to bottom: distinct 3-grams and 5-grams, as the share (%) of n-gram occurrences that
are distinct, against cumulative tokens (log x, up to --x-max M tokens, default 20), on each
pipeline's output for the same 376 source shards ($FIG_DATA_DIR/diversity_ngrams_pool376.json).
Left: per-bucket panels (mean score after cleaning of the pages each pipeline keeps, own y range,
dashed = the bucket's mean before cleaning), DataMan buckets 1-2 / 3 / 4-5 in the top row and
FineWeb-Edu <0.5 / 0.5-1 / >=1 in the bottom row, on the 5,000 held-out pages
($FIG_DATA_DIR/quality_buckets.json). FIG_DATA_DIR defaults to analysis/data.
Drawn at print size (\\linewidth = 5.5 in). Save target: $FIG_DIR/quality_diversity.pdf (FIG_DIR
defaults to analysis/figures)
usage: plot_quality_diversity.py [--x-min M] [--x-max M] [--with-fineweb] [--out pdf] [--png file]   (defaults 3 and 20)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
DIV = json.loads((DATA_DIR / "diversity_ngrams_pool376.json").read_text())["corpora"]
QB = json.loads((DATA_DIR / "quality_buckets.json").read_text())
# FineWeb-rule is left out of the paper figure; --with-fineweb draws it in both halves.
Q_SYS = [k for k in QB["order"] if k != "fineweb_rule" or "--with-fineweb" in sys.argv]
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else FIG_DIR / "quality_diversity.pdf"
X_MAX = float(sys.argv[sys.argv.index("--x-max") + 1]) if "--x-max" in sys.argv else 20.0
X_MIN = float(sys.argv[sys.argv.index("--x-min") + 1]) if "--x-min" in sys.argv else 3.0  # M tokens

mpl.rcParams.update({"text.usetex": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "xtick.major.size": 1.5, "ytick.major.size": 1.5,
    "xtick.major.pad": 1.2, "ytick.major.pad": 1.0, "xtick.minor.size": 0.8})
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
COLOR = {"refinedweb_rule": NAVY, "fineweb_rule": TEAL, "proxc": YELLOW, "ultrax": BLUE, "rescraper": RED}
LABEL = {"refinedweb_rule": "RefinedWeb-rule", "fineweb_rule": "FineWeb-rule", "proxc": "ProX-C",
         "ultrax": "UltraX", "rescraper": "ReScraper (Ours)"}
MARKER = {"proxc": "^", "refinedweb_rule": "s", "ultrax": "o", "rescraper": "D"}
if "--with-fineweb" in sys.argv:  # variant: FineWeb-rule also in the diversity panels (not in the paper)
    MARKER = {"fineweb_rule": "v", **MARKER}
DIV_SYS = ["proxc", "refinedweb_rule", "ultrax", "rescraper"]  # as in the previous diversity figure; ours on top
if "--with-fineweb" in sys.argv:
    DIV_SYS = ["fineweb_rule"] + DIV_SYS


def style(ax, lw=0.5):
    for sp in ax.spines.values():
        sp.set_color("#666666"); sp.set_linewidth(lw)
    ax.tick_params(colors="#333333", length=1.5)
    ax.set_axisbelow(True)


# ---------- diversity panels (distinct n-gram rate vs cumulative tokens)
X_LO = max(max(DIV[k]["tokens"][0] for k in DIV_SYS) / 1e6, X_MIN)
X_HI = min(min(DIV[k]["tokens"][-1] for k in DIV_SYS) / 1e6, X_MAX)
MARKS = [m for m in (1, 2, 3, 5, 10, 20, 30) if X_LO <= m <= X_HI]


def div_panel(ax, n, bottom):
    for k in DIV_SYS:
        c = DIV[k]
        x = np.asarray(c["tokens"], float) / 1e6
        y = 100 * np.asarray(c["distinct"][str(n)], float) / np.asarray(c["ngram_occurrences"][str(n)], float)
        xs = np.unique(np.r_[X_LO, x[(x > X_LO) & (x < X_HI)], X_HI, MARKS])
        ys = np.interp(np.log(xs), np.log(x), y)
        ax.plot(xs, ys, "-", color=COLOR[k], lw=0.7, marker=MARKER[k], ms=3.2,
                markevery=[int(np.searchsorted(xs, m)) for m in MARKS], markerfacecolor=COLOR[k],
                markeredgecolor="white", markeredgewidth=0.3, zorder=4 if k == "rescraper" else 3)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(MARKS))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter([f"{t:g}" for t in MARKS]) if bottom
                                 else mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlim(X_LO / 1.05, X_HI * 1.05)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.text(0.05, 0.06, f"{n}-grams", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    ax.set_ylabel("Distinct (%)", fontsize=7, labelpad=1.5)
    if bottom:
        ax.set_xlabel("Tokens (M)", fontsize=7, labelpad=1)
    ax.grid(axis="x", visible=False)
    style(ax)


# ---------- quality panels (one panel per bucket of the score before cleaning)
def buckets(metric, pre):
    if metric == "fineweb_edu":
        e = [0.0, 0.5, 1.0]
        return [None if x is None else sum(1 for t in e[1:] if x >= t) for x in pre], ["<0.5", "0.5–1", "≥1"]
    edges = [(1, 2, "1–2"), (3, 3, "3"), (4, 5, "4–5")]
    v = [None if x is None else int(round(x)) for x in pre]
    return ([None if x is None else next((i for i, (lo, hi, _) in enumerate(edges) if lo <= x <= hi), None)
             for x in v], [e[2] for e in edges])


def q_panel(ax, metric, members, n_all, tick, first):
    pre_mean = float(np.mean([p["pre"][metric] for p in members]))
    vals = []
    for j, k in enumerate(Q_SYS):
        kept = [p["post"][k][metric] for p in members if p["post"].get(k) is not None]
        v = float(np.mean(kept)) if kept else np.nan
        vals.append(v)
        ax.bar(j, v, width=0.78, facecolor=(*mpl.colors.to_rgb(COLOR[k]), 0.3), edgecolor=COLOR[k],
               linewidth=0.5, zorder=3)
    lo, hi = min(vals + [pre_mean]), max(vals + [pre_mean])
    span = max(hi - lo, 1e-3)
    ax.axhline(pre_mean, color="#555555", linestyle=(0, (2.5, 1.5)), linewidth=0.7, zorder=5)
    ax.set_ylim(lo - span * 0.35, hi + span * 0.55)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f" if span < 0.4 else "%.1f"))
    ax.set_xticks([])
    ax.set_xlabel(f"before {tick} ({100 * len(members) / n_all:.0f}%)", fontsize=6.6, labelpad=1.5)
    ax.tick_params(axis="y", labelsize=5.8, length=1.2, pad=0.5)
    ax.grid(axis="x", visible=False)
    style(ax)


fig = plt.figure(figsize=(5.5, 2.2))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.12, 1.45], hspace=0.2, wspace=0.42)  # tight row gap
div_axes = [fig.add_subplot(gs[r, 4]) for r in range(2)]
for r, n in enumerate((3, 5)):
    div_panel(div_axes[r], n, bottom=(r == 1))
pages = QB["pages"]
q_axes = []
for r, (metric, title) in enumerate((("dataman", "DataMan"), ("fineweb_edu", "FineWeb-Edu"))):
    bidx, ticks = buckets(metric, [p["pre"].get(metric) for p in pages])
    for b, tick in enumerate(ticks):
        ax = fig.add_subplot(gs[r, b])
        q_panel(ax, metric, [p for p, i in zip(pages, bidx) if i == b], len(pages), tick, b == 0)
        if b == 0:
            ax.set_ylabel(title, fontsize=7, labelpad=1.5)
        q_axes.append(ax)
fig.canvas.draw()
r_ = fig.canvas.get_renderer()
# column headers
top = max(ax.get_tightbbox(r_).y1 for ax in div_axes + q_axes) / fig.bbox.height
p0, p1 = q_axes[0].get_position(), q_axes[2].get_position()
fig.text(div_axes[0].get_position().x0 + div_axes[0].get_position().width / 2, top + 0.01, "Diversity",
         ha="center", va="bottom", fontsize=7.5, fontweight="bold")
fig.text((p0.x0 + p1.x1) / 2, top + 0.01, "Quality",
         ha="center", va="bottom", fontsize=7.5, fontweight="bold")
KEYS = [k for k in ["rescraper", "ultrax", "proxc", "refinedweb_rule", "fineweb_rule"] if k in Q_SYS]
box = lambda k: Patch(facecolor=(*mpl.colors.to_rgb(COLOR[k]), 0.3), edgecolor=COLOR[k], linewidth=0.5)
line = lambda k: Line2D([], [], color=COLOR[k], lw=0.7, marker=MARKER[k], ms=3.2, markerfacecolor=COLOR[k],
                        markeredgecolor="white", markeredgewidth=0.3)
handles = [Patch(facecolor=COLOR[k], edgecolor=COLOR[k], linewidth=0.5) for k in KEYS]  # plain color swatches
labels = [LABEL[k] for k in KEYS]
fig.legend(handles, labels, ncol=5, frameon=False, fontsize=6.8, handlelength=0.9, handleheight=0.7, columnspacing=0.9,
           handletextpad=0.3, borderaxespad=0.0, loc="lower center", bbox_to_anchor=(0.5, top + 0.075))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
if "--png" in sys.argv:
    fig.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=250)
for ax in q_axes[:1] + div_axes[:1]:
    bb = ax.get_window_extent()
    print("panel aspect (h/w): %.2f" % (bb.height / bb.width))
print(f"Saved to {OUT}")
