import os
import numpy as np
import pandas as pd
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


ac = pd.read_csv(os.path.join(DATA, "audio_case_level.csv"))
rows = []
for p in ["small_ASR", "medium_ASR", "small_note", "medium_note"]:
    k, n = int(ac[p].sum()), len(ac)
    lo, hi = wilson(k, n)
    rows.append({"pipeline": p, "instances": n, "recalled": k, "recall": k / n,
                 "lo": lo, "hi": hi})
pd.DataFrame(rows).to_csv(os.path.join(DATA, "audio_recall.csv"), index=False)

fz = pd.read_csv(os.path.join(DATA, "stress_results_fuzzy.csv"))
lg = pd.read_csv(os.path.join(DATA, "stress_results_large.csv"))
cols = ["pair", "form", "condition", "model", "recognized"]
asr = pd.concat([fz[cols], lg[cols]], ignore_index=True)
asr["not_recognized"] = 1 - asr["recognized"].astype(int)
gee = GEE.from_formula("not_recognized ~ C(model) + C(condition) + C(form)", groups="pair",
                       data=asr, family=Binomial(), cov_struct=Exchangeable()).fit()
ci = gee.conf_int()
pd.DataFrame({"term": gee.params.index, "beta": gee.params.values,
              "OR": np.exp(gee.params.values),
              "lo": np.exp(ci[0].values), "hi": np.exp(ci[1].values),
              "p": gee.pvalues.values}).to_csv(os.path.join(DATA, "asr_gee_ORs.csv"), index=False)
print("wrote audio_recall.csv and asr_gee_ORs.csv")
