import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "derived")
df = pd.read_csv(os.path.join(DATA, "element_gold_model.csv"))
df["both"] = df.gold_present & df.model_present
df["model_only"] = (~df.gold_present) & df.model_present
df["gold_only"] = df.gold_present & (~df.model_present)
out = df.groupby(["element", "model"])[["both", "model_only", "gold_only"]].sum().reset_index()
out.to_csv(os.path.join(DATA, "element_counts.csv"), index=False)
print("wrote element_counts.csv")
