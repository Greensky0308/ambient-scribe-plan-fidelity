import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)
STRESS = DATA

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 5.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE = "#0072B2"; ORANGE = "#D55E00"; GREEN = "#009E73"; PURPLE = "#CC79A7"; GREY = "#BDBDBD"

fz = pd.read_csv(os.path.join(STRESS, "stress_results_fuzzy.csv"))
lg = pd.read_csv(os.path.join(STRESS, "stress_results_large.csv"))
cols = ["pair", "brand", "generic", "form", "condition", "model", "recognized", "similarity", "transcript"]
df = pd.concat([fz[cols], lg[cols]], ignore_index=True)
df["recognized"] = df["recognized"].astype(bool)
conds = ["clean", "fast", "noise"]
models = ["small", "medium", "large"]
mcolors = {"small": BLUE, "medium": GREEN, "large": ORANGE}


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))

# a recognition by condition x model, collapsed over form
ax = axes[0, 0]
x = np.arange(3); w = 0.24
for k, m in enumerate(models):
    vals, lo, hi = [], [], []
    for c in conds:
        sub = df[(df.model == m) & (df.condition == c)]
        p = sub.recognized.mean(); lo_, hi_ = wilson(sub.recognized.sum(), len(sub))
        vals.append(100 * p); lo.append(100 * (p - lo_)); hi.append(100 * (hi_ - p))
    ax.bar(x + (k - 1) * w, vals, width=w, color=mcolors[m], edgecolor="white", linewidth=0.3, label=m.capitalize())
    ax.errorbar(x + (k - 1) * w, vals, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.5)
ax.set_xticks(x); ax.set_xticklabels(["Clean", "Fast", "Noise (5 dB SNR)"])
ax.set_ylim(0, 110); ax.set_ylabel("Recognition (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=5.4,
          handletextpad=0.2, borderaxespad=0.2)

# b per-pair similarity heatmap under noise
ax = axes[0, 1]
combos = [(m, f) for m in models for f in ["brand", "generic"]]
mat = np.zeros((12, len(combos)))
pairs = df[["pair", "brand", "generic"]].drop_duplicates().sort_values("pair")
for i, r in enumerate(pairs.itertuples()):
    for j, (m, f) in enumerate(combos):
        sub = df[(df.pair == r.pair) & (df.model == m) & (df.form == f) & (df.condition == "noise")]
        mat[i, j] = sub.similarity.mean() if len(sub) else np.nan
im = ax.pcolormesh(np.arange(mat.shape[1] + 1) - 0.5, np.arange(mat.shape[0] + 1) - 0.5,
                   mat, vmin=0, vmax=1, cmap="Blues", edgecolors="white", linewidth=0.2)
ax.set_xlim(-0.5, mat.shape[1] - 0.5); ax.set_ylim(mat.shape[0] - 0.5, -0.5)
ax.set_yticks(range(12)); ax.set_yticklabels([f"{r.brand}/{r.generic}" for r in pairs.itertuples()], fontsize=4.8)
ax.set_xticks(range(len(combos)))
ax.set_xticklabels([f"{m[0].upper()}-{f[0].upper()}" for m, f in combos], fontsize=5.2)
ax.set_title("Noise (5 dB SNR) similarity", fontsize=6.0, pad=2)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)

# c clean -> noise dumbbell
ax = axes[0, 2]
for i, (m, f) in enumerate(combos):
    clean = df[(df.model == m) & (df.form == f) & (df.condition == "clean")].recognized.mean() * 100
    noise = df[(df.model == m) & (df.form == f) & (df.condition == "noise")].recognized.mean() * 100
    ax.plot([clean, noise], [i, i], color=GREY, lw=0.7, zorder=1)
    ax.scatter(clean, i, color=GREEN, s=14, zorder=2)
    ax.scatter(noise, i, color=ORANGE, s=14, zorder=2)
ax.set_yticks(range(len(combos))); ax.set_yticklabels([f"{m.capitalize()} {f}" for m, f in combos], fontsize=5.2)
ax.set_xlabel("Recognition (%)", fontsize=6.2); ax.set_xlim(0, 105)
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)

# d similarity distributions by condition
ax = axes[1, 0]
sns.violinplot(data=df, x="condition", y="similarity", order=conds,
               palette=["#56B4E9", "#009E73", "#D55E00"], inner=None, linewidth=0.4,
               cut=0, ax=ax, density_norm="width")
sns.stripplot(data=df, x="condition", y="similarity", color="#333333", size=0.6, alpha=0.25, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("String similarity")
ax.set_xticks(range(3)); ax.set_xticklabels(["Clean", "Fast", "Noise (5 dB SNR)"])
ax.set_ylim(0, 1.02)
ax.spines[["top", "right"]].set_visible(False)

# e brand vs generic by condition
ax = axes[1, 1]
x = np.arange(3); w = 0.34
for k, f in enumerate(["brand", "generic"]):
    vals, lo, hi = [], [], []
    for c in conds:
        sub = df[(df.form == f) & (df.condition == c)]
        p = sub.recognized.mean(); lo_, hi_ = wilson(sub.recognized.sum(), len(sub))
        vals.append(100 * p); lo.append(100 * (p - lo_)); hi.append(100 * (hi_ - p))
    ax.bar(x + (k - 0.5) * w, vals, width=w, color=[BLUE, ORANGE][k], edgecolor="white", linewidth=0.3, label=f.capitalize())
    ax.errorbar(x + (k - 0.5) * w, vals, yerr=[lo, hi], fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.5)
ax.set_xticks(x); ax.set_xticklabels(["Clean", "Fast", "Noise (5 dB SNR)"])
ax.set_ylim(0, 110); ax.set_ylabel("Recognition (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=5.4, ncol=1,
          handletextpad=0.2, borderaxespad=0.2)

# f model x condition interaction
ax = axes[1, 2]
for m in models:
    vals, lo, hi = [], [], []
    for c in conds:
        sub = df[(df.model == m) & (df.condition == c)]
        p = sub.recognized.mean(); lo_, hi_ = wilson(sub.recognized.sum(), len(sub))
        vals.append(100 * p); lo.append(100 * (p - lo_)); hi.append(100 * (hi_ - p))
    ax.plot(conds, vals, marker="o", ms=3, color=mcolors[m], label=m.capitalize(), lw=0.9)
    ax.fill_between(conds, np.array(vals) - np.array(lo), np.array(vals) + np.array(hi),
                    color=mcolors[m], alpha=0.12, linewidth=0)
ax.set_ylim(0, 105); ax.set_ylabel("Recognition (%)")
ax.set_xticks(range(3)); ax.set_xticklabels(["Clean", "Fast", "Noise (5 dB SNR)"])
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=5.4,
          handletextpad=0.2, borderaxespad=0.2)

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
        yy = ax.get_window_extent(renderer).y1 + 5
        fig.text(xcol / fig.bbox.width, yy / fig.bbox.height, letters[col][row],
                 fontsize=8, ha="left", va="bottom")

fig.savefig(os.path.join(OUT, "figure4.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure4.png"), dpi=300, bbox_inches="tight")
print("wrote figure4.pdf")
