import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)
ac = pd.read_csv(os.path.join(DATA, "audio_case_level.csv"))

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
BLUE = "#0072B2"; ORANGE = "#D55E00"; GREEN = "#009E73"; PURPLE = "#CC79A7"; GREY = "#BDBDBD"
pipes = ["small_ASR", "medium_ASR", "small_note", "medium_note"]
labels = ["Small ASR", "Medium ASR", "Small note", "Medium note"]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))

# a recall with CI
ax = axes[0, 0]
vals, los, his = [], [], []
for p in pipes:
    k, n = int(ac[p].sum()), len(ac)
    lo, hi = wilson(k, n)
    vals.append(100 * k / n); los.append(100 * (k / n - lo)); his.append(100 * (hi - k / n))
x = np.arange(len(pipes))
ax.bar(x, vals, color=[BLUE, GREEN, PURPLE, ORANGE], width=0.62, edgecolor="white", linewidth=0.4)
ax.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2)
for i, v in enumerate(vals): ax.text(i, v + his[i] + 2.5, f"{v:.1f}%", ha="center", fontsize=5.6)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylim(0, 114); ax.set_ylabel("Drug-name recall (%)")
ax.spines[["top", "right"]].set_visible(False)

# b miss matrix
ax = axes[0, 1]
fail = ac[(~ac[pipes]).any(axis=1)].copy()
fail["label"] = fail["case"] + ":\n" + fail["drug"]
fail = fail.iloc[::-1].reset_index(drop=True)
for r, row in fail.iterrows():
    for j, p in enumerate(pipes):
        miss = not bool(row[p])
        ax.scatter(j, r, s=30 if miss else 20, color="#8A6EC2" if miss else "#E6E6E6",
                   edgecolor="white", linewidth=0.5, zorder=2)
ax.set_yticks(range(len(fail))); ax.set_yticklabels(fail["label"], fontsize=5.6, linespacing=0.95)
ax.set_xticks(range(len(pipes))); ax.set_xticklabels(["S-ASR", "M-ASR", "S-note", "M-note"], rotation=20, ha="right")
ax.set_xlim(-0.6, len(pipes) - 0.4); ax.set_ylim(-0.6, len(fail) - 0.4)
ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)

# c ASR GEE forest
ax = axes[0, 2]
or_tab = pd.read_csv(os.path.join(DATA, "asr_gee_ORs.csv"))
or_tab = or_tab[or_tab.term != "Intercept"]
pretty = {"C(model)[T.medium]": "Medium vs. large", "C(model)[T.small]": "Small vs. large",
          "C(condition)[T.fast]": "Fast vs. clean", "C(condition)[T.noise]": "Noise vs. clean",
          "C(form)[T.generic]": "Generic vs. brand"}
or_tab["label"] = or_tab.term.map(pretty)
y = np.arange(len(or_tab))
ax.errorbar(or_tab.OR, y, xerr=[or_tab.OR - or_tab.lo, or_tab.hi - or_tab.OR],
            fmt="o", ms=3.5, color=BLUE, ecolor="#333333", elinewidth=0.6, capsize=2)
ax.axvline(1, color=GREY, lw=0.6, ls="--")
ax.set_yticks(y); ax.set_yticklabels(or_tab.label)
ax.set_xscale("log"); ax.set_xlabel("Odds ratio (95% CI)")
ax.spines[["top", "right"]].set_visible(False)

# d per-drug misses
ax = axes[1, 0]
miss = ac.melt(id_vars=["case", "drug"], value_vars=pipes, var_name="pipe", value_name="ok")
miss = miss[~miss.ok].groupby(["drug", "pipe"]).size().unstack(fill_value=0)
miss = miss.loc[miss.sum(axis=1).sort_values(ascending=False).index][:8]
y = np.arange(len(miss))
left = np.zeros(len(miss))
for j, p in enumerate(pipes):
    ax.barh(y, miss[p].values, left=left, color=[BLUE, GREEN, PURPLE, ORANGE][j], height=0.7,
            edgecolor="white", linewidth=0.4, label=labels[j])
    left += miss[p].values
ax.set_yticks(y); ax.set_yticklabels(miss.index)
ax.set_xlabel("Missed instances (n)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=4.8, handletextpad=0.2)

# e case-level misses
ax = axes[1, 1]
g = ac.groupby("case").agg(n=("drug", "size"), miss=("small_ASR", lambda x: int((~x).sum())))
gc = g.groupby(["n", "miss"]).size().reset_index(name="cases")
ax.scatter(gc.n, gc.miss, s=14 + gc.cases * 16, color=BLUE, alpha=0.7,
           edgecolor="white", linewidth=0.5)
for _, r in gc.iterrows():
    ax.text(r.n, r.miss, str(int(r.cases)), ha="center", va="center",
            fontsize=5.0, color="white")
ax.set_xticks(sorted(gc.n.unique())); ax.set_yticks([0, 1])
ax.set_xlim(0.6, 3.4); ax.set_ylim(-0.15, 1.15)
ax.set_xlabel("Drug instances per case")
ax.set_ylabel("Missed instances (small ASR)")
ax.text(0.02, 0.95, "Bubble size ~ cases", transform=ax.transAxes,
        fontsize=4.6, color="#555555", va="top")
ax.spines[["top", "right"]].set_visible(False)

# f paired ASR vs. note recall
ax = axes[1, 2]
for k, (a, b, color, tcol, lab) in enumerate([("small_ASR", "small_note", "#3FBACB", "#173A42", "Small"),
                                               ("medium_ASR", "medium_note", "#8A6EC2", "#FFFFFF", "Medium")]):
    x, y = [], []
    for cid, sub in ac.groupby("case"):
        x.append(round(sub[a].mean(), 2)); y.append(round(sub[b].mean(), 2))
    d = pd.DataFrame({"x": x, "y": y}).groupby(["x", "y"]).size().reset_index(name="cases")
    dx = (k - 0.5) * 0.07
    ax.scatter(d.x + dx, d.y, s=46, color=color, alpha=0.95,
               edgecolor="white", linewidth=0.5, label=lab, zorder=3)
    for _, r in d.iterrows():
        ax.text(r.x + dx, r.y, str(int(r.cases)), ha="center", va="center",
                fontsize=4.4, color=tcol, zorder=4)
ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--")
ax.set_xlabel("ASR recall")
ax.set_ylabel("Note recall")
ax.set_xlim(-0.06, 1.12); ax.set_ylim(-0.06, 1.08)
ax.spines[["top", "right"]].set_visible(False)
handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c,
                  markeredgecolor="white", markersize=4, label=l)
           for c, l in [("#3FBACB", f"Small (n={ac.case.nunique()})"), ("#8A6EC2", f"Medium (n={ac.case.nunique()})")]]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=4.6,
          handletextpad=0.2, borderaxespad=0.3)

fig.tight_layout(w_pad=1.6, h_pad=1.0)
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
fig.savefig(os.path.join(OUT, "figure2.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "figure2.png"), dpi=300, bbox_inches="tight")
print("wrote figure2.pdf")
