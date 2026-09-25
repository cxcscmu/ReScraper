#!/usr/bin/env python3
"""
Figure 10 (fig:length-dist). KDE distribution of per-document length in GPT-NeoX-20B tokens (full document,
no truncation) for the corpora compared in the paper.

Data file:  $FIG_DATA_DIR/dist_length.json (FIG_DATA_DIR defaults to analysis/data; --data overrides)
              {"lengths": {"rescraper": [...], "ultrax": [...], "refinedweb_rule": [...],
                       "resiliparse": [...], ("fineweb_rule": [...])},
               "sources": {...}, "chars": {...}, "summary": {...}, "pending": [...]}
              each list: per-document token counts of a 5,000-document sample.
              A corpus missing from "lengths" is skipped.

Style: shared palette, serif fonts, font sizes and spines of
plot_scraper_comparison.py; overlaid KDEs with translucent fill
and a solid outline of the same hue; legend inside the upper right.

Density estimate: a Gaussian KDE (Scott's
rule x bw_adjust=2, 2,000 grid points) is fit on log10 of all values of each
corpus (no trimming; log space keeps the long tail from inflating the
bandwidth and keeps all mass on positive values). It is drawn on a linear
x axis as a density in the original units, f(x) = f_log10(log10 x) / (x ln 10),
so areas under the curves are probabilities. The visible x range is where the
mass is: the pooled 1st to 97th percentile, rounded outward to a round tick
step; mass outside the range is simply not drawn.

Save target: $FIG_DIR/length_distribution.pdf (FIG_DIR defaults to analysis/figures)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import gaussian_kde

# ---------- Font sizes ----------
MEDIUM_SIZE = 56
SMALLER_SIZE = 46

plt.rc("font", size=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALLER_SIZE)
plt.rc("ytick", labelsize=SMALLER_SIZE)
plt.rc("figure", titlesize=MEDIUM_SIZE)
plt.rc("legend", fontsize=MEDIUM_SIZE)

# ---------- Global style ----------
mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "TeX Gyre Termes",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.2,
        "lines.markersize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.formatter.use_mathtext": True,
    }
)

# ---------- Palette ----------
PALETTE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = PALETTE
GRAY = "#8C8C8C"

# ---------- Data ----------
HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
DATA_PATH = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else DATA_DIR / "dist_length.json"
FIG_PATH = FIG_DIR / "length_distribution.pdf"
X_LABEL = "Length (tokens)"
BW_ADJUST = 2  # applied in log10 space
GRIDSIZE = 2000
TICK_STEPS = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000]

raw = json.load(open(DATA_PATH))["lengths"]

# Drawn back to front; ours last so its outline sits on top.
SPEC = [
    ("proxc", "ProX-C", YELLOW),
    ("refinedweb_rule", "RefinedWeb-rule", NAVY),
    ("ultrax", "UltraX", BLUE),
    ("rescraper", "ReScraper (Ours)", RED),
]


def clean(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x) & (x > 0)]


present = [(key, label, color) for key, label, color in SPEC if key in raw]
pooled = np.concatenate([clean(raw[k]) for k, _, _ in present])
p_lo, p_hi = np.percentile(pooled, [1, 97])
step = next(s for s in TICK_STEPS if (p_hi - p_lo) / s <= 6)
X_MIN = np.floor(p_lo / step) * step
X_MAX = np.ceil(p_hi / step) * step
X_MIN, X_MAX, step = 0.0, 2000.0, 500  # fixed range requested for the paper: 0-2000 tokens

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(10, 8))

grid = np.linspace(max(X_MIN, 1e-9), X_MAX, GRIDSIZE)
for key, label, color in present:
    logx = np.log10(clean(raw[key]))
    kde = gaussian_kde(logx, bw_method="scott")
    kde.set_bandwidth(kde.factor * BW_ADJUST)
    dens = kde(np.log10(grid)) / (grid * np.log(10))
    ax.fill_between(
        grid,
        dens,
        label=label,
        facecolor=mpl.colors.to_rgba(color, 0.18),
        edgecolor=color,
        linewidth=2.6,
    )

# ---------- Axes ----------
ax.set_xlabel(X_LABEL)
ax.set_ylabel("Prob. Density")
ax.set_xlim(X_MIN, X_MAX)
# First tick strictly right of the left edge, so no x label sits in the corner
# (it would collide with the y-axis 0 label).
xt = np.arange(np.floor(X_MIN / step) * step + step, X_MAX + step / 2, step)
ax.set_xticks(xt)
ax.set_xticklabels([f"{t:g}" for t in xt])
ax.set_ylim(bottom=0)
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
# Small densities (e.g. per-token length densities ~1e-3): show the power of
# ten once, above the top-left corner, instead of long decimals on every tick.
y_exp = int(np.floor(np.log10(ax.get_ylim()[1])))
if y_exp <= -3:
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{v / 10**y_exp:.1f}")
    )
    ax.text(0.0, 1.01, f"$\\times 10^{{{y_exp}}}$", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=36, color="#333333")
else:
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.set_axisbelow(True)

# ---------- Legend (inside, upper right; ours first) ----------
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    loc="upper right",
    frameon=False,
    ncol=1,
    handlelength=0.8,
    handleheight=0.6,
    handletextpad=0.4,
    borderpad=0.2,
    labelspacing=0.15,
    fontsize=36,
)

# ---------- Spine styling ----------
for spine in ax.spines.values():
    spine.set_color("#666666")
    spine.set_linewidth(1.2)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.tick_params(colors="#333333")

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, bbox_inches="tight", pad_inches=0.05)
print(f"Saved to {FIG_PATH}")
print(f"x range [{X_MIN:g}, {X_MAX:g}] step {step:g}; pooled p1={p_lo:.1f} p97={p_hi:.1f}")

# ---------- Console summary ----------
print("\nPer-corpus stats (raw, all documents):")
for key, label, _ in SPEC[::-1]:
    if key not in raw:
        print(f"  {label:18s}  (not in data file)")
        continue
    arr = clean(raw[key])
    print(
        f"  {label:18s}  n={arr.size:>6d}  median={np.median(arr):8.1f}  "
        f"mean={arr.mean():8.1f}  p10={np.percentile(arr, 10):7.1f}  "
        f"p90={np.percentile(arr, 90):8.1f}  shown={np.mean((arr >= X_MIN) & (arr <= X_MAX)):.1%}"
    )

plt.close()
