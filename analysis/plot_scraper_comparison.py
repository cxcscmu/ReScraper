#!/usr/bin/env python3
"""
Figure 3 (fig:scraper-comparison). Scraper comparison: DCLM Core score of pretraining on text from each heuristic
scraper (resiliparse, trafilatura, jusText, Dripper; same rule-based cleaning on
top) vs. ReScraper, in the 400M and 1B settings.

Style: shared six-color palette, serif fonts, transparent fill + colored edge,
#666666 spines. Two panels, one per setting, independent
y-ranges so within-setting gaps are legible; every bar is direct-labeled so the
truncated y-axis cannot mislead.

Core scores are the paper's numbers (Table 2 and the scraper runs), hard-coded below.
Save target: $FIG_DIR/scraper_comparison.pdf (FIG_DIR defaults to analysis/figures)
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

# ---------- Data ----------
HERE = Path(__file__).resolve().parent  # analysis/
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
FIG_PATH = FIG_DIR / "scraper_comparison.pdf"

scrapers = ["resiliparse", "trafilatura", "jusText", "Dripper", "Dripper+UltraX", "ReScraper"]
settings = [
    ("400M Setting", [0.12572, 0.13194, 0.13474, 0.14080, 0.144, 0.15345]),
    ("1B Setting", [0.25340, 0.25165, 0.25072, 0.25262, 0.265, 0.27348]),
]

# One palette hue per scraper, ours in RED. Order chosen so adjacent bars stay
# separable for normal and CVD vision (GREEN/TEAL and TEAL/BLUE are too close).
colors = [YELLOW, GREEN, NAVY, BLUE, TEAL, RED]

face_alpha = 0.3
edge_alpha = 1.0
bar_width = 0.62

# ---------- Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(24, 9.5), sharey=False)

for ax, (title, vals) in zip(axes, settings):
    x = np.arange(len(scrapers))
    for xi, v, c in zip(x, vals, colors):
        ax.bar(
            xi,
            v,
            width=bar_width,
            facecolor=(*mpl.colors.to_rgb(c), face_alpha),
            edgecolor=(*mpl.colors.to_rgb(c), edge_alpha),
            linewidth=1.8,
            zorder=3,
        )

    # Direct labels: every bar carries its value so the truncated axis is honest.
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = span * 0.05
    for xi, v in zip(x, vals):
        ax.text(
            xi,
            v + pad * 0.6,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=SMALLER_SIZE - 6,
            color="#333333",
            zorder=4,
        )

    ax.set_title(title, pad=14, fontsize=SMALLER_SIZE - 4)
    ax.set_xticks(x)
    ax.set_xticklabels(scrapers, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=SMALLER_SIZE - 8, pad=8)
    ax.set_ylim(lo - span * 0.6, hi + span * 0.45)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(1.2)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(colors="#333333")

axes[0].set_ylabel("DCLM Core", fontsize=SMALLER_SIZE - 4)

fig.tight_layout(w_pad=3.0)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, bbox_inches="tight", pad_inches=0.05)
print(f"Saved to {FIG_PATH}")
plt.close()
