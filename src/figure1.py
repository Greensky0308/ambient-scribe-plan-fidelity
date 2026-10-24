import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)
cl = pd.read_csv(os.path.join(DATA, "case_level.csv"))
f1 = pd.read_csv(os.path.join(DATA, "element_f1.csv"))

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
ORANGE = "#3FBACB"; GREEN = "#0E7F6F"; PURPLE = "#CC79A7"; GREY = "#BDBDBD"; TEAL = "#00796B"
LABELS = {
    "hepatotoxic_any": "Hepatotoxic medication", "liver_medications": "Liver-related medication",
    "masld_components": "MASLD components", "alcohol": "Alcohol use",
    "alcohol_quantity": "Alcohol quantity", "alcohol_counseling": "Alcohol counselling",
    "liver_labs": "Liver labs", "liver_followup": "Liver follow-up",
    "viral_hepatitis": "Viral hepatitis", "decompensation": "Decompensation",
    "liver_diagnosis": "Liver diagnosis",
}
els = [e for e in f1.element.unique() if e in LABELS]
els = sorted(els, key=lambda e: f1[(f1.element == e)].F1.mean())

fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))

# a heatmap F1
ax = axes[0, 0]
mat = f1.pivot(index="element", columns="model", values="F1").reindex(els)[["flash", "pro"]]
im = ax.pcolormesh(np.arange(mat.shape[1] + 1) - 0.5, np.arange(mat.shape[0] + 1) - 0.5,
                   mat.values, vmin=0.85, vmax=1.0, shading="flat",
                   cmap=LinearSegmentedColormap.from_list("white_green", ["#FFFFFF", "#0E7F6F"]),
                   edgecolors="white", linewidth=0.4)
ax.set_xlim(-0.5, mat.shape[1] - 0.5); ax.set_ylim(mat.shape[0] - 0.5, -0.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Flash", "Pro"])
ax.set_yticks(range(len(els))); ax.set_yticklabels([LABELS[e] for e in els])
for i in range(len(els)):
    for j in range(2):
        v = mat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.4,
                    color="white" if v > 0.96 else "black")
ax.set_title("Element-level F1", fontsize=6.8, pad=3)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)

# b F1 with CI
ax = axes[0, 1]
y = np.arange(len(els))
for k, (model, color) in enumerate([("flash", ORANGE), ("pro", GREEN)]):
    sub = f1[f1.model == model].set_index("element").reindex(els)
    ax.errorbar(sub.F1, y + (0.12 if k else -0.12), xerr=[sub.F1 - sub.F1_lo, sub.F1_hi - sub.F1],
                fmt="o", ms=3, color=color, ecolor=color, elinewidth=0.6, capsize=1.5,
                label="Flash" if k == 0 else "Pro")
ax.set_yticks(y); ax.set_yticklabels([LABELS[e] for e in els])
ax.invert_yaxis()
ax.set_xlim(0.78, 1.005)
ax.set_xlabel("F1 (95% bootstrap CI)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", frameon=False, handletextpad=0.2, borderaxespad=0.2)

# c per-encounter completeness
ax = axes[0, 2]
rows = []
for model in ["gen_flash", "gen_pro"]:
    sub = cl[cl.model == model]
    for _, r in sub.iterrows():
        d = [r[f"d_{e}"] for e in els]
        g = [r[f"g_{e}"] for e in els]
        den = sum(bool(x) for x in d)
        if den:
            rows.append({"model": "Flash" if model == "gen_flash" else "Pro",
                         "completeness": sum(bool(a) and bool(b) for a, b in zip(g, d)) / den})
comp = pd.DataFrame(rows)
levels = [0.0, 0.6, 0.75, 0.8, 1.0]
counts = comp.groupby(["model", "completeness"]).size()
for k, (model, color, lab) in enumerate([("Flash", ORANGE, "Flash"), ("Pro", GREEN, "Pro")]):
    for i, lv in enumerate(levels):
        cnt = int(counts.get((model, lv), 0))
        y = i + (k - 0.5) * 0.18
        if cnt:
            ax.hlines(y, 1, cnt, color=color, linewidth=0.8, alpha=0.9, zorder=2)
            ax.scatter(cnt, y, s=15, color=color, edgecolor="white", linewidth=0.4,
                       zorder=3, label=lab if i == len(levels) - 1 else None)
            ax.text(cnt * 1.1, y, str(cnt), va="center", fontsize=4.8, color="#333333")
ax.set_xscale("log")
ax.set_xlim(0.8, 450)
ax.set_yticks(range(len(levels)))
ax.set_yticklabels(["0", "0.60", "0.75", "0.80", "1.00"])
ax.set_xlabel("Encounters (n, log scale)")
ax.set_ylabel("Per-encounter completeness")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.2)

# d element counts vs clinician note
ax = axes[1, 0]
dd = pd.read_csv(os.path.join(DATA, "element_counts.csv"))
dd = dd.groupby("element")[["both", "model_only", "gold_only"]].sum().reindex(els).fillna(0)
y = np.arange(len(dd))
ax.barh(y, dd.both, color=GREEN, label="Both", height=0.7)
ax.barh(y, dd.model_only, left=dd.both, color="#CE6BA7", label="Model only", height=0.7)
ax.barh(y, dd.gold_only, left=dd.both + dd.model_only, color="#3FBACB", label="Clinician only", height=0.7)
ax.invert_yaxis()
ax.set_yticks(y); ax.set_yticklabels([LABELS[e] for e in dd.index])
ax.set_xlabel("Element counts (two models)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=5.2, handletextpad=0.2)

# e FN / FP
ax = axes[1, 1]
sub = f1[f1.model == "flash"].set_index("element").reindex(els)
y = np.arange(len(els))
ax.barh(y, -sub.FN, height=0.7, color="#4F91CB", label="FN (missed)")
ax.barh(y, sub.FP, height=0.7, color="#C167A1", label="FP (added)")
ax.axvline(0, color="#333333", linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels([LABELS[e] for e in els])
ax.invert_yaxis()
ax.set_xlabel("Count (Flash)")
ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xlim(-4, 4)
ax.set_xticklabels(["-3", "-2", "-1", "0", "1", "2", "3"])
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=5.2, handletextpad=0.2)

# f incomplete capture by number of dialogue-positive elements
ax = axes[1, 2]
rows = []
for model in ["gen_flash", "gen_pro"]:
    for _, r in cl[cl.model == model].iterrows():
        d = [bool(r[f"d_{e}"]) for e in els]
        g = [bool(r[f"g_{e}"]) for e in els]
        den = sum(d)
        if den:
            cap = sum(a and b for a, b in zip(g, d))
            rows.append({"model": model, "n": den, "incomplete": cap < den})
dfc = pd.DataFrame(rows)
ns_vals = [1, 2, 3, 4, 5, 6, 7, 9]
xb = np.arange(len(ns_vals))
for k, (model, color, lab) in enumerate([("gen_flash", ORANGE, "Flash"), ("gen_pro", GREEN, "Pro")]):
    vals, txts = [], []
    for n in ns_vals:
        sub = dfc[(dfc.model == model) & (dfc.n == n)]
        kk, nn = int(sub.incomplete.sum()), len(sub)
        vals.append(100 * kk / nn); txts.append(f"{kk}/{nn}")
    ax.bar(xb + (k - 0.5) * 0.34, vals, width=0.32, color=color,
           edgecolor="white", linewidth=0.4, label=lab)
    for xv, yv, t in zip(xb + (k - 0.5) * 0.34, vals, txts):
        if yv > 0:
            ax.text(xv - 0.03 if k == 0 else xv + 0.03, yv + 1.5, t,
                    ha="right" if k == 0 else "left", fontsize=4.6, color="#333333")
ax.set_xticks(xb)
ax.set_xticklabels([f"{n}\n({int(((dfc.n == n).sum()) // 2)})" for n in ns_vals])
ax.set_xlabel("Dialogue-positive liver elements (n)")
ax.set_ylabel("Encounters with incomplete capture (%)")
ax.set_ylim(0, 60)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=5.2, handletextpad=0.2)


fig.tight_layout(w_pad=1.0, h_pad=1.0)
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
        y = ax.get_window_extent(renderer).y1 + 5
        fig.text(xcol / fig.bbox.width, y / fig.bbox.height, letters[col][row],
                 fontsize=8, ha="left", va="bottom")

fig.savefig(os.path.join(OUT, "figure1.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure1.png"), dpi=300, bbox_inches="tight")
print("wrote figure1.pdf")
