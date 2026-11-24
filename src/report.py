"""Algorithms 6 and 7: SHAP explanation + personalised risk report for one person."""
import json
from pathlib import Path

import joblib, numpy as np, pandas as pd, shap

import pipeline as P                      # noqa: F401  (unpickling needs the InferencePipeline class)
from build_dataset import NOMINAL
from knowledge_base import DISCLAIMER, KB
from schema import FIELDS, option_label

ROOT = Path(__file__).resolve().parent.parent


class RiskReporter:
    def __init__(self):
        art = joblib.load(ROOT / "models/best_model.joblib")
        self.inf, self.best, self.classes = art["pipeline"], art["best_name"], art["classes"]
        self.cols, self.bg = art["input_columns"], art["background"]
        self.xtest, self.ytest = art["xtest"], art["ytest"]
        self.sel = json.load(open(ROOT / "results/selection.json"))
        df = pd.read_csv(ROOT / "data/processed/nhanes_2005_06_modelling.csv")
        self.prevalence = df.deficiency.value_counts(normalize=True).to_dict()
        self.medians = df[self.cols].median(numeric_only=True).to_dict()
        m = self.inf.model
        try:                                          # exact TreeSHAP where the library supports the model (Algorithm 6)
            self.explainer = shap.TreeExplainer(m); self.shap_method = "TreeSHAP"
        except Exception:                             # e.g. multi-class sklearn GradientBoosting, SVM, KNN, MLP, ...
            self.explainer = shap.KernelExplainer(m.predict_proba, shap.kmeans(self.bg, 15)); self.shap_method = "KernelSHAP"

    # ---- input handling ----------------------------------------------------------------
    def to_frame(self, raw: dict) -> pd.DataFrame:
        row, errors = {}, []
        for c in self.cols:
            v = raw.get(c)
            if v in (None, ""):
                row[c] = np.nan; continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                errors.append(f"{FIELDS[c][1]}: '{v}' is not a number"); row[c] = np.nan; continue
            group, label, kind, extra = FIELDS[c]
            if kind == "choice" and option_label(c, v) is None:
                errors.append(f"{label}: unknown answer {v:g}"); v = np.nan
            elif kind == "number" and not (extra["min"] <= v <= extra["max"]):
                errors.append(f"{label}: {v:g} is outside {extra['min']}-{extra['max']}"); v = np.nan
            row[c] = v
        if np.isnan(row["bmi"]) and not np.isnan(row["height_cm"]) and not np.isnan(row["weight_kg"]):
            row["bmi"] = row["weight_kg"] / (row["height_cm"] / 100) ** 2
        if errors:
            raise ValueError("; ".join(errors))
        return pd.DataFrame([row], columns=self.cols)

    # ---- explanation -------------------------------------------------------------------
    def _shap(self, z):
        sv = self.explainer.shap_values(z) if not isinstance(self.explainer, shap.KernelExplainer) \
            else self.explainer.shap_values(z, nsamples=200, silent=True)
        if isinstance(sv, list):                      # older shap: list over classes of (1, f)
            return np.stack([s[0] for s in sv])
        sv = np.asarray(sv)
        return sv[0].T if sv.ndim == 3 else sv        # (1, f, c) -> (c, f)

    def _describe(self, feat):
        """selected-feature name -> (raw field, category value or None)"""
        for raw in NOMINAL.values():
            if feat.startswith(raw + "_"):
                return raw, float(feat[len(raw) + 1:])
        return feat, None

    def report(self, raw: dict) -> dict:
        X = self.to_frame(raw)
        z = self.inf.transform(X); proba = self.inf.model.predict_proba(z)[0]
        k = int(proba.argmax()); pred = self.classes[k]                       # Eq. 13
        phi = self._shap(z)[k]
        names = self.inf.selected_names
        drivers = []
        for j in np.argsort(-np.abs(phi))[:6]:
            field, cat = self._describe(names[j]); group, label, kind, extra = FIELDS[field]
            given = X.iloc[0][field]
            if cat is not None:                       # one-hot indicator: say whether it applies to this person
                label = f"{label}: {option_label(field, cat)}"
                if np.isnan(given):
                    shown = "not provided (most common answer assumed)"
                elif given == cat:
                    shown = "applies to you"
                else:
                    shown = f"does not apply (you answered {option_label(field, given)})"
            elif np.isnan(given):
                shown = "not provided (typical value assumed)"
            elif kind == "choice":
                shown = option_label(field, given)
            else:
                shown = f"{given:g} {extra.get('unit', '')}".strip()
            drivers.append(dict(feature=label, your_value=shown, shap=float(phi[j]),
                                direction="raises" if phi[j] > 0 else "lowers"))
        order = np.argsort(-proba)
        h = self.sel["holdout"]
        return dict(
            predicted=pred, confidence=float(proba[k]), kb=KB[pred], drivers=drivers,
            scores=[dict(label=self.classes[i], score=float(proba[i]), prevalence=float(self.prevalence.get(self.classes[i], 0)))
                    for i in order],
            runner_up=self.classes[order[1]] if len(order) > 1 else None,
            model=dict(name=self.best, shap_method=self.shap_method, holdout_macro_f1=h["f1"], holdout_accuracy=h["accuracy"], holdout_auc=h["auc"],
                       n_train=self.sel["n_dev"]),
            disclaimer=DISCLAIMER,
        )

    def samples(self, n=6, seed=7):
        """Real, unseen hold-out people for live demos. Their lab-based label is returned as 'actual'."""
        rng = np.random.RandomState(seed); picks = []
        for c in range(len(self.classes)):
            idx = np.where(self.ytest == c)[0]
            if len(idx):
                picks.append(int(rng.choice(idx)))
        out = []
        for i in picks[:n]:
            row = {k: (None if pd.isna(v) else float(v)) for k, v in self.xtest.iloc[i].items()}
            out.append(dict(id=i, actual=self.classes[self.ytest[i]], inputs=row))
        return out
