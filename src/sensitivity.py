"""How much do the conclusions depend on SMOTE strength and on the RFE feature budget?
One stratified 5-fold pass per configuration (same dev split and folds as the main run).

Usage: PYTHONPATH=src python src/sensitivity.py
"""
import warnings
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import pipeline as P
from train_compare import ROOT, RES, run_fold

warnings.filterwarnings("ignore")
CONFIGS = {  # name: (smote_ratio, p_star)
    "no SMOTE, p*=30":      (0.0, 30),
    "SMOTE 0.25, p*=30":    (0.25, 30),
    "SMOTE 0.5, p*=30":     (0.5, 30),
    "SMOTE 1.0, p*=30 (documented)": (1.0, 30),
    "SMOTE 1.0, p*=15":     (1.0, 15),
    "SMOTE 1.0, all features": (1.0, 10_000),
}

def main():
    df = pd.read_csv(ROOT / "data/processed/nhanes_2005_06_modelling.csv")
    le = LabelEncoder().fit(df.deficiency); n = len(le.classes_)
    X, y = df[P.NUM_COLS + P.CAT_COLS], le.transform(df.deficiency)
    Xdev, _, ydev, _ = train_test_split(X, y, test_size=0.15, stratify=y, random_state=P.SEED)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=P.SEED).split(Xdev, ydev))
    out = []
    for name, (ratio, pstar) in CONFIGS.items():
        pstar = min(pstar, len(P.make_preprocessor().fit(Xdev).get_feature_names_out()))
        res = Parallel(n_jobs=4)(delayed(run_fold)(i, 0, Xdev, ydev, tr, va, n, ratio, pstar) for i, (tr, va) in enumerate(splits))
        d = pd.DataFrame([r for rows, _ in res for r in rows]); d["config"] = name; out.append(d)
        print(name, d[d.model != "Majority baseline"].groupby("model").f1.mean().round(3).to_dict(), flush=True)
    pd.concat(out).to_csv(RES / "sensitivity_folds.csv", index=False)

if __name__ == "__main__":
    main()
