#!/usr/bin/env python3
"""
Extraction quality on the held-out pages (Figure 8, fig:extraction), one frame at half the text
width (next to the decision-flow Sankey):
  left axis:  token F1 of each extractor against Dripper, the teacher's extractor
              (ReScraper's own <extract> step, resiliparse, trafilatura, jusText);
  right axis: gpt-oss-120b score of each extraction against the rendered page, without any reference
              extraction (recall of the main content + precision against boilerplate + integrity,
              each 0-2, so 0-6); Dripper is judged too.
Token overlap is the lowercased \\w+ multiset used by analyze_fidelity_by_length_5k.py; values are
means over pages. Colors follow the scraper-comparison figure (plot_scraper_comparison.py).

Data: $FIG_DATA_DIR/extraction_quality.json = {"ext_vs_dripper": <token-F1 summary of evaluation/>,
                                               "judge": <extraction-judge summary of evaluation/>}
      (FIG_DATA_DIR defaults to analysis/data; --data <file> reads another combined file, and
      --ext <file> --judge <file> read the two evaluation outputs directly instead).

Save target: $FIG_DIR/extraction_quality.pdf (FIG_DIR defaults to analysis/figures; --out overrides)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # analysis/
DATA_DIR = Path(os.environ.get("FIG_DATA_DIR", HERE / "data"))
FIG_DIR = Path(os.environ.get("FIG_DIR", HERE / "figures"))
DATA_PATH = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else DATA_DIR / "extraction_quality.json"
FIG_PATH = FIG_DIR / "extraction_quality.pdf"


def arg(name):
    return Path(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else None


if arg("--ext"):
    EXT, JUD = json.loads(arg("--ext").read_text()), json.loads(arg("--judge").read_text())
else:
    D = json.loads(DATA_PATH.read_text())
    EXT, JUD = D["ext_vs_dripper"], D["judge"]

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

PALETTE = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = PALETTE
# (key in the data files, label, color as in the scraper-comparison figure)
SYSTEMS = [("student", "ReScraper", RED), ("dripper", "Dripper", BLUE),
           ("resiliparse", "resiliparse", YELLOW), ("trafilatura", "trafilatura", GREEN),
           ("justext", "jusText", NAVY)]
face_alpha = 0.3
LW = 0.7


def est(v):
    return v["est"] if isinstance(v, dict) else v


def style(ax):
    for sp in ax.spines.values():
        sp.set_color("#666666")
        sp.set_linewidth(0.6)
    ax.tick_params(axis="both", length=2, pad=1.5, colors="#333333")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def grouped(ax, groups, systems, value, top, fmt):
    ns = len(systems)
    W = 0.84 / ns
    for g, (gkey, glabel) in enumerate(groups):
        for j, (k, label, c) in enumerate(systems):
            v = value(k, gkey)
            x = g - 0.42 + W * (j + 0.5)
            ax.bar(x, v, width=W * 0.9, facecolor=(*mpl.colors.to_rgb(c), face_alpha), edgecolor=c,
                   linewidth=LW, zorder=3, label=label if g == 0 else None)
            ax.text(x, v + top * 0.01, fmt(v), ha="center", va="bottom", fontsize=5.5, rotation=90,
                    color="#222222")
    ax.set_xticks(range(len(groups)), [gl for _, gl in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, top)


def hug_legend(fig, leg, axes, pad_pt=2.0):
    """Put a figure-level legend right above the highest element of `axes` (titles and tick labels
    included), pad_pt points above it; the saved tight bbox then leaves no gap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = max(ax.get_tightbbox(r).y1 for ax in axes)  # display pixels
    y = (top + pad_pt * fig.dpi / 72) / fig.bbox.height
    leg.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
    leg._loc = 8  # lower center


fig, axl = plt.subplots(figsize=(2.9, 1.5))  # flatter, to match the shorter Sankey
axr = axl.twinx()
GAP = 0.6
ext_sys = [sp for sp in SYSTEMS if sp[0] in EXT["systems"]]
jud_sys = [sp for sp in SYSTEMS if sp[0] in JUD["systems"]]
halves = ((axl, ext_sys, lambda k: 100 * EXT["systems"][k]["tokF1_macro"]["est"], 0, 100, 0.0, "{:.0f}"),
          (axr, jud_sys, lambda k: est(JUD["systems"][k]["total_of_6"]), 0, 6, len(ext_sys) + GAP, "{:.1f}"))
for ax, syss, val, lo, top, x0, fmt in halves:
    for i, (k, label, c) in enumerate(syss):
        v = val(k)
        ax.bar(x0 + i, v, width=0.72, facecolor=(*mpl.colors.to_rgb(c), face_alpha), edgecolor=c,
               linewidth=LW, zorder=3, label=label if ax is axr else None)
        ax.text(x0 + i, v + (top - lo) * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=6.5,
                color="#222222")
    ax.set_ylim(lo, top * 1.12)
    ax.tick_params(axis="y", labelsize=7.5, length=2, pad=1.5)
axl.set_yticks([0, 25, 50, 75, 100])
axr.set_yticks([0, 2, 4, 6])
axl.set_ylabel("Token F1 (%)", labelpad=2, fontsize=8)
axr.set_ylabel("Judge score (0\u20136)", labelpad=2, fontsize=8)
axr.grid(False)
axl.grid(axis="x", visible=False)
axl.set_axisbelow(True)
n_l, n_r = len(ext_sys), len(jud_sys)
axl.set_xlim(-0.55, n_l + GAP + n_r - 0.45)
axl.axvline(n_l - 0.5 + GAP / 2, color="#999999", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
axl.set_xticks([(n_l - 1) / 2, n_l + GAP + (n_r - 1) / 2], ["F1 vs. Dripper", "LLM judge"], fontsize=8)
axl.tick_params(axis="x", length=0, pad=2)
for ax in (axl, axr):
    for sp in ax.spines.values():
        sp.set_color("#666666")
        sp.set_linewidth(0.6)
h, l = axr.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 1))
leg = fig.legend(h, l, ncol=len(l), frameon=False, handlelength=0.8, handleheight=0.6,
                 columnspacing=0.55, handletextpad=0.25, fontsize=6.8, borderaxespad=0.0)
hug_legend(fig, leg, [axl, axr])
out = arg("--out") or FIG_PATH
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
if "--png" in sys.argv:
    fig.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=300)
print(f"Saved to {out}")
for k, label, _ in SYSTEMS:
    e = EXT["systems"].get(k)
    j = JUD["systems"].get(k)
    print(f"  {label:20s}", "tokP/R/F1 = " + "/".join(f"{100 * e[m]['est']:.1f}" for m in ("tokP_macro", "tokR_macro", "tokF1_macro")) if e else " " * 28,
          f"judge total {est(j['total_of_6']):.2f}" if j else "")
