import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)

cl = pd.read_csv(os.path.join(DATA, "case_level.csv"))
ac = pd.read_csv(os.path.join(DATA, "audio_case_level.csv"))

elements = ["alcohol", "alcohol_quantity", "alcohol_counseling", "hepatotoxic_any",
            "liver_labs", "masld_components", "liver_medications", "decompensation",
            "liver_followup", "viral_hepatitis", "family_history_liver", "liver_diagnosis"]
rows = []
for el in elements:
    for m in ["gen_flash", "gen_pro"]:
        s = cl[cl.model == m]
        d = s[f"d_{el}"].astype(bool); g = s[f"g_{el}"].astype(bool)
        rows.append({"element": el, "model": m.replace("gen_", ""),
                     "positive": int(d.sum()), "captured": int((d & g).sum()),
                     "omitted": int((d & ~g).sum()),
                     "over_documented": int((~d & g).sum())})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table2_text_fidelity.csv"), index=False)

rows = []
for p in ["small_ASR", "medium_ASR", "small_note", "medium_note"]:
    k, n = int(ac[p].sum()), len(ac)
    any_miss = int(ac.groupby("case")[p].apply(lambda x: (~x).any()).sum())
    rows.append({"pipeline": p, "instances": n, "recalled": k, "missed": n - k,
                 "recall_pct": round(100 * k / n, 1),
                 "cases_with_any_miss": any_miss, "cases": ac["case"].nunique()})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table3_audio_fidelity.csv"), index=False)

fz = pd.read_csv(os.path.join(DATA, "stress_results_fuzzy.csv"))
lg = pd.read_csv(os.path.join(DATA, "stress_results_large.csv"))
df = pd.concat([fz, lg], ignore_index=True)
rows = []
for (c, m, f), sub in df.groupby(["condition", "model", "form"]):
    rows.append({"condition": c, "model": m, "form": f,
                 "recognized": int(sub.recognized.sum()), "n": len(sub),
                 "percent": round(100 * sub.recognized.mean(), 1)})
pd.DataFrame(rows).sort_values(["condition", "model", "form"]).to_csv(
    os.path.join(OUT, "stress_summary.csv"), index=False)

flow = pd.read_csv(os.path.join(DATA, "plan_flow.csv")).set_index("metric")["value"].to_dict()
n = int(flow["total_encounters"])
pos = int(flow["screen_positive"])
tp = int(flow["detected"])
fp = int(flow["fp"])
fn = int(flow["fn"])
neg = int(flow["screen_negative"])
ev_neg = int(flow["evaluable_negative"])
confirmed = int(flow["confirmed_cases"])


def pct(a, b):
    return 100 * a / b


def per1000(v):
    return round(v / n * 1000)


rows = [
    {"metric": "Encounters, n",
     "per_264": f"{n}", "per_1000": f"{per1000(n)}"},
    {"metric": "Routed to review, n (%)",
     "per_264": f"{pos} ({pct(pos, n):.1f})", "per_1000": f"{per1000(pos)} ({pct(pos, n):.1f})"},
    {"metric": "Confirmed high/medium discrepancy among reviewed, n (%)",
     "per_264": f"{tp} ({pct(tp, pos):.1f})", "per_1000": f"{per1000(tp)} ({pct(tp, pos):.1f})"},
    {"metric": "Reviewed without confirmed high/medium discrepancy, n (%)",
     "per_264": f"{fp} ({pct(fp, pos):.1f})", "per_1000": f"{per1000(fp)} ({pct(fp, pos):.1f})"},
    {"metric": "Not reviewed, n",
     "per_264": f"{neg}", "per_1000": f"{per1000(neg)}"},
    {"metric": "Missed high/medium discrepancy among unreviewed, n (%)",
     "per_264": f"{fn} ({pct(fn, ev_neg):.1f})", "per_1000": f"{per1000(fn)} ({pct(fn, ev_neg):.1f})"},
    {"metric": "Reviews per confirmed high/medium discrepancy, n",
     "per_264": f"{pos / tp:.2f}", "per_1000": f"{pos / tp:.2f}"},
    {"metric": "Reviews per confirmed high/medium discrepancy with full review, n",
     "per_264": f"{n / confirmed:.2f}", "per_1000": f"{n / confirmed:.2f}"},
    {"metric": "Reduction in review volume vs. full review, %",
     "per_264": f"{pct(n - pos, n):.1f}", "per_1000": f"{pct(n - pos, n):.1f}"},
    {"metric": "Missed share of confirmed high/medium discrepancies, %",
     "per_264": f"{pct(fn, confirmed):.1f}", "per_1000": f"{pct(fn, confirmed):.1f}"},
]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table5_operational_yield.csv"), index=False)
print("wrote table outputs")
