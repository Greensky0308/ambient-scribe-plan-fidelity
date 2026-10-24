import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE = "#0072B2"; ORANGE = "#D55E00"; GREEN = "#009E73"; PURPLE = "#CC79A7"; GREY = "#BDBDBD"
LIGHT_BLUE = "#56B4E9"

cases_df = pd.read_csv(os.path.join(DATA, "table4_plan_cases.csv"))
cases = [tuple(r) for r in cases_df[["case", "category", "severity"]].itertuples(index=False, name=None)]

fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))

# a attrition funnel
ax = axes[0, 0]; ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
cx = 3.0
bands = [(5.4, 3.9, 5.6, 3.4, "#0B4F6C"), (3.9, 2.4, 3.4, 1.6, BLUE),
         (2.4, 0.9, 1.6, 0.8, LIGHT_BLUE)]
for ytop, ybot, wt, wb, col in bands:
    ax.add_patch(plt.Polygon([[cx - wt / 2, ytop], [cx + wt / 2, ytop],
                              [cx + wb / 2, ybot], [cx - wb / 2, ybot]],
                             closed=True, facecolor=col, edgecolor="white", lw=0.6))
labels = [
    ("321 text encounters", 4.65, 5.5),
    ("42 adjudicated\n(32 LLM-flagged + 10 controls)", 3.15, 4.3),
    ("7 confirmed high/medium\n(2.2%; 1 HIGH, 6 MEDIUM)", 1.65, 3.6),
]
for text, yy, xedge in labels:
    ax.plot([xedge, 5.85], [yy, yy], color="#888888", lw=0.5)
    ax.text(5.95, yy, text, ha="left", va="center", fontsize=6.2, linespacing=1.05)

# b confirmed cases by category and severity
ax = axes[0, 1]
cats = ["Follow-up / timing", "Medication", "Test / monitoring"]
short = ["Follow-up", "Medication", "Test"]
for i, (case, cat, sev) in enumerate(cases):
    ax.scatter(cats.index(cat), len(cases) - 1 - i, s=48,
               color=ORANGE if sev == "HIGH" else LIGHT_BLUE,
               edgecolor="white", linewidth=0.6, zorder=3)
ax.set_yticks(range(len(cases)))
ax.set_yticklabels([c[0] for c in cases][::-1], fontsize=6.0)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(short, rotation=15, ha="right")
for i, cat in enumerate(cats):
    ax.text(i, len(cases) + 0.15, f"n={sum(1 for c in cases if c[1] == cat)}",
            ha="center", fontsize=5.6, color="#555555")
ax.set_xlim(-0.5, len(cats) - 0.5); ax.set_ylim(-0.5, len(cases) + 0.5)
ax.set_ylabel("Confirmed case")
ax.spines[["top", "right"]].set_visible(False)
handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ORANGE,
                  markeredgecolor="white", markersize=4.5, label="HIGH"),
           Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=LIGHT_BLUE,
                  markeredgecolor="white", markersize=4.5, label="MEDIUM")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5.8,
          handletextpad=0.2, borderaxespad=0.2)

# c adjudicated discrepancies by severity
ax = axes[0, 2]
sev_df = pd.read_csv(os.path.join(DATA, "severity_counts.csv")).set_index("severity")
sev_names = ["LOW", "MEDIUM", "HIGH"]
sev_counts = [int(sev_df.loc[s, "count"]) for s in sev_names]
sev_colors = [LIGHT_BLUE, BLUE, "#0B4F6C"]
y = np.arange(len(sev_names))
ax.barh(y, sev_counts, height=0.4, color=sev_colors, edgecolor="white", linewidth=0.4)
for i, v in enumerate(sev_counts):
    ax.text(v + 0.12, i, str(v), va="center", fontsize=6.2)
ax.set_yticks(y); ax.set_yticklabels(sev_names)
ax.set_xlabel("Adjudicated discrepancies (n)")
ax.set_xlim(0, 7.5)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.98, 1.06, "high/medium: 7/7 actionable", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=5.2, color="#555555")

# d A/B agreement matrix
ax = axes[1, 0]
agree = pd.read_csv(os.path.join(DATA, "reviewer_agreement.csv")).iloc[0]
mat = np.array([[agree.a_no_b_no, agree.a_no_b_yes], [agree.a_yes_b_no, agree.a_yes_b_yes]])
ax.pcolormesh(np.arange(3) - 0.5, np.arange(3) - 0.5, mat, cmap="Blues", vmin=0, vmax=30,
              edgecolors="white", linewidth=0.6)
ax.set_xlim(-0.7, 1.7); ax.set_ylim(1.7, -0.7)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                fontsize=8, color="white" if mat[i, j] > 15 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["B: no", "B: yes"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["A: no", "A: yes"])
ax.set_xlabel("Reviewer B"); ax.set_ylabel("Reviewer A")
ax.set_title("A/B agreement 36/42 (85.7%), κ = 0.635", fontsize=6.5, pad=3)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)

# e plan -> note change map
ax = axes[1, 1]; ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.add_patch(FancyBboxPatch((0.15, 5.2), 5.0, 0.55,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#CDE8D3", edgecolor="white", lw=0.5))
ax.add_patch(FancyBboxPatch((6.05, 5.2), 3.8, 0.55,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            facecolor="#FAD9C4", edgecolor="white", lw=0.5))
ax.text(2.65, 5.475, "Dialogue plan", ha="center", va="center", fontsize=6.4, fontweight="bold")
ax.text(7.95, 5.475, "Final note", ha="center", va="center", fontsize=6.4, fontweight="bold")
ex_df = pd.read_csv(os.path.join(DATA, "plan_examples.csv"))
examples = [tuple(r) for r in ex_df[["case", "severity", "plan", "note", "change"]].itertuples(index=False, name=None)]
for i, (case, sev, plan, note, change) in enumerate(examples):
    yc = 4.35 - i * 1.5
    ax.add_patch(FancyBboxPatch((0.15, yc - 0.45), 5.0, 0.95,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#E8F3E8", edgecolor="white", lw=0.5))
    ax.add_patch(FancyBboxPatch((6.05, yc - 0.45), 3.8, 0.95,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#FDECDD", edgecolor="white", lw=0.5))
    ax.text(2.65, yc + 0.22, f"{case} · {sev}", ha="center", va="center",
            fontsize=5.4, fontweight="bold", color="#333333")
    ax.text(2.65, yc - 0.20, plan, ha="center", va="center", fontsize=5.8)
    ax.text(7.95, yc + 0.22, change, ha="center", va="center",
            fontsize=5.6, fontweight="bold", color="#333333")
    ax.text(7.95, yc - 0.20, note, ha="center", va="center", fontsize=5.8)
    ax.annotate("", xy=(6.0, yc), xytext=(5.2, yc),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333"))

# f reviewer concordance
ax = axes[1, 2]; ax.axis("off")
conc_df = pd.read_csv(os.path.join(DATA, "reviewer_concordance.csv"))
rows = [tuple(r) for r in conc_df[["case", "A", "B", "C"]].itertuples(index=False, name=None)]
ax.set_xlim(-2.2, 2.6); ax.set_ylim(-0.5, 11.6)
for j, h in enumerate(["A", "B", "C"]):
    ax.text(j, 11.1, h, ha="center", va="center", fontsize=6.0, fontweight="bold")
for i, (case, a, b, c) in enumerate(rows):
    yy = 10 - i
    ax.text(-2.1, yy, case, ha="left", va="center", fontsize=5.6)
    for j, v in enumerate([a, b, c]):
        ax.scatter(j, yy, s=22 if v == "Yes" else 14,
                   color=BLUE if v == "Yes" else "#D9D9D9",
                   edgecolor="white", linewidth=0.5, zorder=2)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)

fig.tight_layout(w_pad=1.4, h_pad=7.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
letters = [["a", "d"], ["b", "e"], ["c", "f"]]
for col in range(axes.shape[1]):
    col_axes = axes[:, col]
    xs_all = []
    for ax in col_axes:
        xs_all.append(ax.get_window_extent(renderer).x0)
        if ax.yaxis.label.get_text():
            xs_all.append(ax.yaxis.label.get_window_extent(renderer).x0)
        for lab in ax.get_yticklabels():
            if lab.get_text():
                xs_all.append(lab.get_window_extent(renderer).x0)
    xcol = min(xs_all)
    for row, ax in enumerate(col_axes):
        yy = ax.get_window_extent(renderer).y1 + 5
        fig.text(xcol / fig.bbox.width, yy / fig.bbox.height, letters[col][row],
                 fontsize=8, ha="left", va="bottom")

fig.savefig(os.path.join(OUT, "figure3.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure3.png"), dpi=300, bbox_inches="tight")
print("wrote figure3.pdf")
