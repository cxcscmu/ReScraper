#!/usr/bin/env python3
"""
Figure 9 (fig:token-waterfall). Token volume of the held-out pages after each cumulative stage of ReScraper: the input
rendering, then extraction, deletion, editing and rewriting (GPT-NeoX-20B tokens, % of input).

Data (default): $FIG_DATA_DIR/token_waterfall_5k.json (compute_token_waterfall_5k.py; 4,989 of the
5,000 held-out pages, release decoding; FIG_DATA_DIR defaults to analysis/data).
Style mirrors plot_scraper_comparison.py (shared palette, serif fonts, face_alpha=0.3 +
colored edge); operation colors follow the method overview figure.

Save target (default): $FIG_DIR/token_waterfall_5k.pdf (FIG_DIR defaults to analysis/figures)
usage: plot_token_waterfall_5k.py [--data FILE] [--out FILE.pdf] [--png FILE.png]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", type=Path, default=DATA_DIR / "token_waterfall_5k.json")
_ap.add_argument("--out", type=Path, default=FIG_DIR / "token_waterfall_5k.pdf")
_ap.add_argument("--png", default=None)
_args = _ap.parse_args()
DATA_PATH = _args.data
FIG_PATH = _args.out

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 40,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

PALETTE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = PALETTE
face_alpha = 0.3

share = json.loads(DATA_PATH.read_text())["share_of_input"]
bars = [("input", "Input\nrendering", NAVY), ("extract", "After\nextract", "#555555"),
        ("delete", "After\ndelete", RED), ("edit", "After\nedit", YELLOW), ("rewrite", "After\nrewrite", BLUE)]

fig, ax = plt.subplots(figsize=(14, 6.5))
for i, (k, lab, c) in enumerate(bars):
    v = 100 * share[k]
    ax.bar(i, v, width=0.66, facecolor=(*mpl.colors.to_rgb(c), face_alpha), edgecolor=c, linewidth=2, zorder=3)
    ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=36, color="#222222")
ax.set_xticks(range(len(bars)), [b[1] for b in bars], fontsize=36)
ax.set_ylabel("Tokens (% of input)")
ax.set_ylim(0, 115)
for sp in ax.spines.values():
    sp.set_color("#666666")
    sp.set_linewidth(1.2)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH)
if _args.png:
    plt.savefig(_args.png, dpi=55)
print(f"Saved to {FIG_PATH}")
