import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")

elements = ["alcohol", "alcohol_quantity", "alcohol_counseling", "hepatotoxic_any",
            "liver_labs", "masld_components", "liver_medications", "decompensation",
            "liver_followup", "viral_hepatitis", "family_history_liver", "liver_diagnosis"]
cl = pd.read_csv(os.path.join(DATA, "case_level.csv"))
rng = np.random.default_rng(42)


def stats(d, g, mask=None):
    if mask is not None:
        d, g = d[mask], g[mask]
    tp = int(np.sum(d & g)); fp = int(np.sum(~d & g)); fn = int(np.sum(d & ~g))
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (
        prec and rec and not np.isnan(prec) and not np.isnan(rec)) else np.nan
    return tp, fp, fn, prec, rec, f1


rows = []
for el in elements:
    for model, label in (("gen_flash", "flash"), ("gen_pro", "pro")):
        sub = cl[cl.model == model]
        d = sub[f"d_{el}"].astype(bool).values
        g = sub[f"g_{el}"].astype(bool).values
        tp, fp, fn, prec, rec, f1 = stats(d, g)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(d), len(d))
            boots.append(stats(d, g, idx)[5])
        boots = np.array([x for x in boots if not np.isnan(x)])
        rows.append({"element": el, "model": label, "n": len(d), "TP": tp, "FP": fp, "FN": fn,
                     "precision": prec, "recall": rec, "F1": f1,
                     "F1_lo": np.percentile(boots, 2.5) if len(boots) else np.nan,
                     "F1_hi": np.percentile(boots, 97.5) if len(boots) else np.nan})

pd.DataFrame(rows).to_csv(os.path.join(DATA, "element_f1.csv"), index=False)
print("wrote element_f1.csv")
