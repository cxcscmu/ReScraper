"""Figure 1(a): of the pages each type of rule drops, the share worth keeping.

Pages are the 5,000 held-out pages labeled keep or remove by gpt-oss-120b. Merged rule groups come
from analyze_rule_groups.py on rule_flags_5000.jsonl (released with the <HF_ORG>/<DATASET> dataset
upon acceptance, under analysis/keep_judge/): a page belongs to a group when either stack drops it
with a first-failing rule in that group. The numbers below are the printed output of that script.

Save target: $FIG_DIR/rule_worth_keeping.pdf (FIG_DIR defaults to analysis/figures).
"""

import os
from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"

MEDIUM_SIZE = 46
SMALLER_SIZE = 42
plt.rc("font", size=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALLER_SIZE)
plt.rc("ytick", labelsize=SMALLER_SIZE)
FIG_SIZE = 7  # square canvas, same as Figure 1(b)

palette = ["#EF3A47", "#FDB515", "#009647", "#008F91", "#043673", "#007BC0"]
RED, YELLOW, GREEN, TEAL, NAVY, BLUE = palette

# (rule group, n pages dropped, % of them judged worth keeping)
rows = [
    ("too many repetitions", 1568, 65.11),
    ("mostly numbers or symbols", 1134, 63.23),
    ("few lines end like sentences", 189, 59.26),
    ("too many noisy lines", 287, 52.96),
    ("page too short", 213, 31.46),
]
rows = sorted(rows, key=lambda r: -r[2])

r, g, b = matplotlib.colors.to_rgb(NAVY)
# Square 5.6 in plot box, same size in both Figure 1 panels.
fig = plt.figure(figsize=(6.20, 7.40))
ax = fig.add_axes([0.0484, 0.2095, 0.9032, 0.7568])
y = np.arange(len(rows))[::-1]
for yi, (name, n, val) in zip(y, rows):
    ax.barh(yi, val, height=0.42, color=(r, g, b, 0.3), edgecolor=NAVY, linewidth=2, zorder=3)
    ax.text(0.8, yi + 0.27, name, ha="left", va="bottom", fontsize=32, fontweight="bold", zorder=4)
    ax.text(val + 1.2, yi, f"{val:.0f}", ha="left", va="center", fontsize=38, zorder=4)

ax.set_yticks([])
ax.set_xlim(0, 75)
ax.set_xticks([0, 25, 50, 75])
ax.set_ylim(-0.5, len(rows) - 0.1)
ax.set_xlabel("Worth keeping (%)")
ax.grid(axis="x", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)

FIG_DIR = Path(os.environ.get("FIG_DIR", Path(__file__).resolve().parent / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_DIR / "rule_worth_keeping.pdf")
