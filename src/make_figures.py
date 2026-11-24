"""Presentation figures from results/*.csv|json. Usage: PYTHONPATH=src python src/make_figures.py"""
import json
from pathlib import Path

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "results/figures"
FIG.mkdir(exist_ok=True, parents=True)
sns.set_theme(style="whitegrid", font_scale=1.0)
BLUE, GREY, ACCENT = "#2b6cb0", "#a0aec0", "#dd6b20"

folds = pd.read_csv(RES / "cv_folds.csv")
sel = json.load(open(RES / "selection.json"))
best, classes = sel["best"], sel["classes"]
nine = folds[folds.model != "Majority baseline"]
summ = pd.read_csv(RES / "model_comparison.csv", index_col=0)
order = list(summ.sort_values("rank").index)
base = folds[folds.model == "Majority baseline"][["accuracy", "f1", "auc"]].mean()


def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, dpi=200); plt.close()


# 1. headline: macro-F1 +/- sd, best highlighted, baseline line
fig, ax = plt.subplots(figsize=(9, 4.8))
m, s = summ.loc[order, "f1_mean"], summ.loc[order, "f1_std"]
ax.bar(order, m, yerr=s, capsize=4, color=[ACCENT if o == best else BLUE for o in order])
ax.axhline(base.f1, color="k", ls="--", lw=1); ax.text(len(order) - .5, base.f1 + .005, "majority baseline", ha="right", fontsize=8)
for i, v in enumerate(m): ax.text(i, v + s.iloc[i] + .006, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("Macro-F1 (mean ± sd over folds)"); ax.set_title(f"Nine classifiers compared - stratified 5-fold CV x{int(sel['n_folds']/5)} repeats")
plt.xticks(rotation=25, ha="right"); save("01_macro_f1_comparison.png")

# 2. all five metrics heat-map
cols = {"accuracy_mean": "Accuracy", "precision_mean": "Precision", "recall_mean": "Recall", "f1_mean": "Macro-F1", "auc_mean": "AUC-ROC"}
hm = summ.loc[order, list(cols)].rename(columns=cols)
fig, ax = plt.subplots(figsize=(7.5, 5))
sns.heatmap(hm, annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=ax); ax.set_ylabel("")
ax.set_title("All five metrics (ranked best -> worst)"); save("02_metrics_heatmap.png")

# 3. fold-to-fold spread
fig, ax = plt.subplots(figsize=(9, 4.8))
sns.boxplot(data=nine, x="model", y="f1", order=order, color="#bee3f8", ax=ax)
sns.stripplot(data=nine, x="model", y="f1", order=order, color=BLUE, size=3, alpha=.6, ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Macro-F1 per fold"); plt.xticks(rotation=25, ha="right"); ax.set_title("Stability across folds"); save("03_fold_spread.png")

# 4. per-class F1 (where do models succeed/fail?)
pc = nine.groupby("model")[[f"f1_class{i}" for i in range(len(classes))]].mean().loc[order]; pc.columns = classes
fig, ax = plt.subplots(figsize=(8.5, 5)); sns.heatmap(pc, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "F1"})
ax.set_ylabel(""); ax.set_title("Per-class F1 - which deficiencies are learnable?"); save("04_per_class_f1.png")

# 5. confusion matrices - out-of-fold (CV) and hold-out
oofd = joblib.load(RES / "oof_repeat0.joblib"); y_dev = oofd["y_dev"]; o = oofd["oof"][best]
cm = confusion_matrix(y_dev[o.idx.values], o.pred.values, labels=range(len(classes)))
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, mat, ttl in [(axs[0], cm, f"{best}: cross-validated (n={cm.sum()})"), (axs[1], np.array(sel["holdout_confusion"]), f"{best}: untouched hold-out (n={int(np.sum(sel['holdout_confusion']))})")]:
    norm = mat / mat.sum(1, keepdims=True).clip(min=1)
    sns.heatmap(norm, annot=mat, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (row-normalised colour)"); ax.set_title(ttl)
    ax.tick_params(axis="x", rotation=30)
save("05_confusion_matrices.png")

# 6. training cost
fig, ax = plt.subplots(figsize=(8, 4)); t = summ.loc[order, "fit_seconds_mean"].sort_values()
ax.barh(t.index, t.values, color=GREY); ax.set_xscale("log"); ax.set_xlabel("Mean training time per fold (s, log scale)"); ax.set_title("Training cost"); save("06_training_time.png")

# 7. class distribution
df = pd.read_csv(ROOT / "data/processed/nhanes_2005_06_modelling.csv")
fig, ax = plt.subplots(figsize=(7.5, 4)); vc = df.deficiency.value_counts()
ax.barh(vc.index[::-1], vc.values[::-1], color=BLUE)
for i, v in enumerate(vc.values[::-1]): ax.text(v + 10, i, f"{v} ({v/len(df):.1%})", va="center", fontsize=9)
ax.set_xlim(0, vc.max() * 1.25); ax.set_title(f"Class distribution (N = {len(df)}) - why SMOTE is needed"); save("07_class_distribution.png")

# 8. sensitivity study
sp = RES / "sensitivity_folds.csv"
if sp.exists():
    sd = pd.read_csv(sp); sd = sd[sd.model != "Majority baseline"]
    piv = sd.groupby(["config", "model"]).f1.mean().unstack().loc[:, order]
    fig, ax = plt.subplots(figsize=(10, 4.5)); sns.heatmap(piv, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar_kws={"label": "Macro-F1"})
    ax.set_ylabel(""); ax.set_xlabel(""); ax.set_title("Sensitivity: SMOTE strength and RFE feature budget (5-fold CV)"); plt.xticks(rotation=25, ha="right"); save("08_sensitivity.png")
    acc = sd.groupby(["config", "model"]).accuracy.mean().unstack().loc[:, order]
    fig, ax = plt.subplots(figsize=(10, 4.5)); sns.heatmap(acc, annot=True, fmt=".2f", cmap="Greens", ax=ax, cbar_kws={"label": "Accuracy"})
    ax.set_ylabel(""); ax.set_xlabel(""); ax.set_title("Sensitivity: accuracy (majority baseline = %.2f)" % base.accuracy); plt.xticks(rotation=25, ha="right"); save("09_sensitivity_accuracy.png")
print("figures ->", FIG)
