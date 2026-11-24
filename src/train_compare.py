"""Algorithms 4 and 5: train all nine classifiers under identical stratified CV,
compare them, pick the best, retrain it, evaluate once on a hold-out set and serialise it.

Usage:  PYTHONPATH=src python src/train_compare.py [--repeats 3] [--smoke]
"""
import argparse, json, time, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

import pipeline as P

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES, MODELS = ROOT / "results", ROOT / "models"
RES.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
F1_TIE = 0.005   # models within this macro-F1 margin count as tied -> AUC-ROC breaks the tie


def metrics(y, pred, proba, n_classes):
    out = dict(accuracy=accuracy_score(y, pred),
               precision=precision_score(y, pred, average="macro", zero_division=0),
               recall=recall_score(y, pred, average="macro", zero_division=0),
               f1=f1_score(y, pred, average="macro", zero_division=0),
               f1_weighted=f1_score(y, pred, average="weighted", zero_division=0),
               auc=roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes))))
    for c, v in enumerate(f1_score(y, pred, average=None, labels=list(range(n_classes)), zero_division=0)):
        out[f"f1_class{c}"] = v
    return out


def run_fold(fold_id, repeat, Xdev, ydev, tr, va, n_classes, smote_ratio=1.0, p_star=P.P_STAR):
    pre, sel, Ztr, ytr = P.prepare_fold(Xdev.iloc[tr], ydev[tr], smote_ratio, p_star)
    Zva = sel.transform(pre.transform(Xdev.iloc[va])); yva = ydev[va]
    rows, oof = [], {}
    models = P.make_models(n_classes)
    for name, m in models.items():
        t0 = time.time(); m.fit(Ztr, ytr); fit_s = time.time() - t0
        proba = m.predict_proba(Zva); pred = proba.argmax(1)
        rows.append(dict(model=name, fold=fold_id, repeat=repeat, fit_seconds=fit_s, **metrics(yva, pred, proba, n_classes)))
        oof[name] = (va, pred)
    # reference only (not one of the nine): always predict the most common class of the REAL (un-balanced) training labels
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(tr), 1)), ydev[tr])
    proba = dummy.predict_proba(np.zeros((len(va), 1))); pred = proba.argmax(1)
    rows.append(dict(model="Majority baseline", fold=fold_id, repeat=repeat, fit_seconds=0.0,
                     **metrics(yva, pred, proba, n_classes)))
    oof["Majority baseline"] = (va, pred)
    return rows, oof


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="1 fold only, to time the pipeline"); a = ap.parse_args()

    df = pd.read_csv(ROOT / "data/processed/nhanes_2005_06_modelling.csv")
    le = LabelEncoder().fit(df.deficiency); classes = list(le.classes_); n_classes = len(classes)
    X, y = df[P.NUM_COLS + P.CAT_COLS], le.transform(df.deficiency)
    # hold-out set: untouched until the single final evaluation of the chosen model
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=0.15, stratify=y, random_state=P.SEED)
    print(f"classes={classes}\ndev={len(Xdev)} holdout={len(Xtest)}", flush=True)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=P.SEED)
    splits = list(cv.split(Xdev, ydev))[: 1 if a.smoke else None]
    t0 = time.time()
    res = Parallel(n_jobs=4, verbose=5)(delayed(run_fold)(i, i // 5, Xdev, ydev, tr, va, n_classes)
                                         for i, (tr, va) in enumerate(splits))
    print(f"CV done in {time.time()-t0:.0f}s", flush=True)
    cvdf = pd.DataFrame([r for rows, _ in res for r in rows])
    cvdf.to_csv(RES / ("cv_folds_smoke.csv" if a.smoke else "cv_folds.csv"), index=False)

    # pooled out-of-fold predictions of repeat 0 -> confusion matrices
    oof = {}
    for rows, o in res[:5]:
        for name, (va, pred) in o.items():
            oof.setdefault(name, []).append(pd.DataFrame({"idx": va, "pred": pred}))
    oof = {k: pd.concat(v).sort_values("idx") for k, v in oof.items()}
    joblib.dump({"oof": oof, "y_dev": ydev, "classes": classes}, RES / "oof_repeat0.joblib")

    nine = cvdf[cvdf.model != "Majority baseline"]
    summ = nine.groupby("model")[["accuracy", "precision", "recall", "f1", "f1_weighted", "auc", "fit_seconds"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.sort_values("f1_mean", ascending=False)
    base = cvdf[cvdf.model == "Majority baseline"][["accuracy", "f1", "auc"]].mean()
    print(summ[["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]].round(4).to_string())
    print("baseline", base.round(4).to_dict(), flush=True)
    if a.smoke: return

    # ---- Algorithm 5: rank by macro-F1, AUC-ROC breaks ties ------------------------
    ranked = list(summ.sort_values("f1_mean", ascending=False).index)           # primary criterion: macro-F1
    top_f1 = summ.loc[ranked[0], "f1_mean"]
    tied = [m for m in ranked if top_f1 - summ.loc[m, "f1_mean"] <= F1_TIE]    # "tie" = within F1_TIE of the leader
    leader = max(tied, key=lambda m: summ.loc[m, "auc_mean"])                  # tie-breaker: AUC-ROC
    ranked = [leader] + [m for m in ranked if m != leader]
    best = ranked[0]; summ["rank"] = [ranked.index(m) + 1 for m in summ.index]
    summ.to_csv(RES / "model_comparison.csv")

    # significance: Friedman over all folds, then paired Wilcoxon winner vs each rival (Holm-corrected)
    wide = nine.pivot_table(index=["repeat", "fold"], columns="model", values="f1")
    fr = stats.friedmanchisquare(*[wide[m] for m in wide.columns])
    pv = {m: stats.wilcoxon(wide[best], wide[m]).pvalue for m in wide.columns if m != best}
    srt = sorted(pv.items(), key=lambda kv: kv[1]); k = len(srt)
    holm, run = {}, 0.0
    for i, (m, p) in enumerate(srt):
        run = max(run, min(1.0, p * (k - i))); holm[m] = run
    sig = dict(best=best, ranked=ranked, friedman_chi2=float(fr.statistic), friedman_p=float(fr.pvalue),
               winner_vs_rival_holm_p={m: float(v) for m, v in holm.items()}, n_folds=int(len(wide)),
               baseline_accuracy=float(base.accuracy), baseline_f1=float(base.f1))
    print(json.dumps(sig, indent=2), flush=True)

    # ---- retrain winner on the full dev set (balanced), evaluate once on hold-out -------
    pre, sel, Zb, yb = P.prepare_fold(Xdev, ydev)
    model = P.make_models(n_classes)[best].fit(Zb, yb)
    inf = P.InferencePipeline(pre, sel, model, classes)
    proba = inf.predict_proba(Xtest); pred = proba.argmax(1)
    hold = metrics(ytest, pred, proba, n_classes)
    cm = confusion_matrix(ytest, pred, labels=list(range(n_classes)))
    print("HOLD-OUT", {k: round(v, 4) for k, v in hold.items() if not k.startswith("f1_class")}, flush=True)
    sig.update(holdout=hold, holdout_confusion=cm.tolist(), classes=classes,
               selected_features=inf.selected_names, n_dev=int(len(Xdev)), n_holdout=int(len(Xtest)))
    json.dump(sig, open(RES / "selection.json", "w"), indent=2)

    # background sample (preprocessed + selected) for SHAP, and the deployable artefact
    rng = np.random.RandomState(P.SEED); idx = rng.choice(len(Zb), 200, replace=False)
    joblib.dump(dict(pipeline=inf, best_name=best, classes=classes, background=Zb[idx],
                     xtest=Xtest, ytest=ytest, input_columns=P.NUM_COLS + P.CAT_COLS), MODELS / "best_model.joblib")
    print("saved", MODELS / "best_model.joblib")


if __name__ == "__main__":
    main()
