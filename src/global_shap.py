"""Global SHAP importance of the deployed model on the hold-out set (Algorithm 6, population view).
Usage: PYTHONPATH=src python src/global_shap.py"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns

from report import RiskReporter
from schema import FIELDS, option_label

os.chdir(Path(__file__).resolve().parent.parent)   # paths below are relative to the project root
r = RiskReporter()
Z = r.inf.transform(r.xtest)
sv = r.explainer.shap_values(Z) if r.shap_method == "TreeSHAP" else r.explainer.shap_values(Z, nsamples=300, silent=True)
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)          # (n, features, classes)


def pretty(feat):
    raw, cat = r._describe(feat); label = FIELDS[raw][1]
    return f"{label}: {option_label(raw, cat)}" if cat is not None else label


names = [pretty(f) for f in r.inf.selected_names]
imp = pd.DataFrame(np.abs(sv).mean(0), index=names, columns=r.classes)          # mean |phi| per feature and class
imp["overall"] = imp.mean(1)
imp = imp.sort_values("overall", ascending=False)
imp.to_csv("results/shap_global.csv")

top = imp.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top.index, top["overall"], color="#2b6cb0"); ax.set_xlabel("Mean |SHAP value| (averaged over classes, hold-out set)")
ax.set_title(f"What drives the {r.best} model? ({r.shap_method})"); plt.tight_layout()
plt.savefig("results/figures/10_shap_global.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(8.5, 6))
sns.heatmap(imp.head(12).drop(columns="overall"), annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"label": "mean |SHAP|"})
ax.set_title("Top features per deficiency class"); plt.tight_layout()
plt.savefig("results/figures/11_shap_per_class.png", dpi=200); plt.close()
print(imp.head(10).round(3).to_string())
