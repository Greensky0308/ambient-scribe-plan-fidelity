import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE = "#0072B2"; ORANGE = "#D55E00"; SKY = "#56B4E9"; GREY = "#999999"

flow = pd.read_csv(os.path.join(DATA, "plan_flow.csv")).set_index("metric")["value"].to_dict()
nature = pd.read_csv(os.path.join(DATA, "severity_by_nature.csv"))
acc = pd.read_csv(os.path.join(DATA, "accuracy.csv"))
agree = pd.read_csv(os.path.join(DATA, "reviewer_agreement.csv"))

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

# a flow
ax = axes[0, 0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)


def box(x, y, w, h, text, fc="#EAF2F8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                facecolor=fc, edgecolor="#555555", lw=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2, linespacing=1.15)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.5, color="#555555"))


box(3.2, 8.6, 3.6, 0.9, f"{flow['total_encounters']} text encounters")
box(0.2, 5.6, 4.4, 1.5, f"Screen-negative\n{flow['screen_negative']} encounters\n"
    f"{flow['not_evaluable']} not evaluable · {flow['evaluable_negative']} evaluable")
box(5.4, 5.6, 4.4, 1.5, f"Screen-positive\n{flow['screen_positive']} encounters\n"
    f"{flow['flagged_items']} flagged items")
box(0.2, 2.4, 4.4, 1.6, f"Evaluable negatives\n{flow['fn']} high/medium\n"
    f"{flow['tn']} without high/medium")
box(5.4, 2.4, 4.4, 1.6, f"Flagged items\n{flow['tp']} encounters with high/medium\n"
    f"{flow['fp']} without high/medium")
box(3.0, 0.2, 4.0, 0.9, f"{flow['confirmed_cases']} confirmed high/medium\n"
    f"{flow['detected']} detected + {flow['missed']} missed", fc="#FDEBE3")
arrow(4.4, 8.6, 2.6, 7.1); arrow(5.6, 8.6, 7.4, 7.1)
arrow(2.4, 5.6, 2.4, 4.0); arrow(7.6, 5.6, 7.6, 4.0)
arrow(2.4, 2.4, 4.2, 1.1); arrow(7.6, 2.4, 5.8, 1.1)
ax.text(0.02, 0.98, "a", transform=ax.transAxes, va="top", fontsize=8, fontweight="bold")

# b confirmed items by nature and severity
ax = axes[0, 1]
high = nature["HIGH"].values
med = nature["MEDIUM"].values
x = np.arange(len(nature))
ax.bar(x, high, color=ORANGE, width=0.55, label="HIGH")
ax.bar(x, med, bottom=high, color=SKY, width=0.55, label="MEDIUM")
for i in range(len(nature)):
    ax.text(i, high[i] + med[i] + 0.4, str(int(high[i] + med[i])), ha="center", fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(nature["nature"].str.replace(" ", "\n"))
ax.set_ylabel("High/medium items")
ax.set_ylim(0, max(high + med) + 5)
ax.legend(frameon=False, loc="upper right", handlelength=1.0)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.25, 1.02, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

# c accuracy with Wilson CIs
ax = axes[1, 0]
vals = acc["estimate"].values
err = np.vstack([vals - acc["lo"].values, acc["hi"].values - vals])
colors = [BLUE if m != "PPV" else ORANGE for m in acc["metric"]]
ax.barh(np.arange(len(acc)), vals, color=colors, height=0.5)
ax.errorbar(vals, np.arange(len(acc)), xerr=err, fmt="none", ecolor="#333333", lw=0.8, capsize=2)
for i, v in enumerate(vals):
    ax.text(min(v + 4, 101), i, f"{v:.1f}%", va="center", fontsize=6)
ax.set_yticks(np.arange(len(acc))); ax.set_yticklabels(acc["metric"])
ax.set_xlim(0, 110); ax.set_xlabel("Percent (95% CI)")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.28, 1.02, "c", transform=ax.transAxes, fontsize=8, fontweight="bold")

# d reviewer agreement
ax = axes[1, 1]
labels = agree["set"].str.replace("_", " ").str.capitalize()
ax.bar(np.arange(len(agree)), agree["kappa"].values, color=GREY, width=0.5)
for i, v in enumerate(agree["kappa"].values):
    ax.text(i, v + 0.02, f"κ = {v:.3f}", ha="center", fontsize=6)
for y in (0.4, 0.6):
    ax.axhline(y, color="#777777", lw=0.5, ls=":")
ax.set_xticks(np.arange(len(agree))); ax.set_xticklabels(labels)
ax.set_ylabel("Cohen κ")
ax.set_ylim(0, max(agree["kappa"]) + 0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.28, 1.02, "d", transform=ax.transAxes, fontsize=8, fontweight="bold")

fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(OUT, "figure3.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure3.png"), dpi=300, bbox_inches="tight")
print("wrote figure3.pdf / figure3.png")
