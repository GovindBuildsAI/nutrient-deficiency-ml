"""Modules 1-3 and the nine classifiers (Algorithms 1-4 of the methodology).

Everything that *learns from data* (imputer medians, scaler mean/std, SMOTE,
RFE) is fitted on the training part of a fold only, so the validation fold is
never seen while preparing data. This keeps the comparison leak-free.
"""
import numpy as np, pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from build_dataset import NOMINAL, NUMERIC, ORDINAL

NUM_COLS = list(NUMERIC.values()) + list(ORDINAL.values())   # ordinal scales stay ordered and numeric
CAT_COLS = list(NOMINAL.values())
SEED = 42
P_STAR = 30   # target number of features kept by RFE (p* in Algorithm 3)


def make_preprocessor():
    """Algorithm 1: median imputation (Eq.1), one-hot (Eq.2), z-score (Eq.3)."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)], verbose_feature_names_out=False)


def make_selector(p_star=P_STAR):
    """Algorithm 3: recursive feature elimination driven by Random-Forest importance (Eq.5)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
    return RFE(rf, n_features_to_select=p_star, step=0.1)


def make_smote(y=None, ratio=1.0):
    """Algorithm 2: SMOTE interpolation (Eq.4), k = 5 neighbours.
    ratio=1.0 balances every class up to the majority count (the documented design);
    ratio=0 disables SMOTE; 0<ratio<1 only lifts each minority class to ratio x majority (used in the sensitivity study)."""
    strategy = "auto"
    if ratio < 1.0:
        counts = np.bincount(y); top = counts.max()
        strategy = {c: int(ratio * top) for c, n in enumerate(counts) if n < ratio * top}
    return SMOTE(k_neighbors=5, sampling_strategy=strategy, random_state=SEED)


def make_models(n_classes):
    """The nine classifiers, default hyper-parameters, identical data for all."""
    return {
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
        "XGBoost":           XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=1, eval_metric="mlogloss",
                                           objective="multi:softprob", num_class=n_classes, verbosity=0),
        "SVM":               SVC(probability=True, random_state=SEED),
        "Decision Tree":     DecisionTreeClassifier(random_state=SEED),
        "KNN":               KNeighborsClassifier(),
        "Naive Bayes":       GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "AdaBoost":          AdaBoostClassifier(random_state=SEED),
        "MLP":               MLPClassifier(max_iter=500, random_state=SEED),
    }


def prepare_fold(X_train, y_train, smote_ratio=1.0, p_star=P_STAR):
    """Fit preprocessing -> SMOTE -> RFE on one training set. Returns fitted pieces and balanced data."""
    pre = make_preprocessor().fit(X_train)
    Z = pre.transform(X_train)
    Zb, yb = (make_smote(y_train, smote_ratio).fit_resample(Z, y_train) if smote_ratio > 0 else (Z, y_train))
    sel = make_selector(p_star).fit(Zb, yb)
    return pre, sel, sel.transform(Zb), yb


class InferencePipeline:
    """preprocess -> select -> model. SMOTE is training-only so it is not part of inference."""
    def __init__(self, pre, sel, model, classes):
        self.pre, self.sel, self.model, self.classes = pre, sel, model, list(classes)
        names = np.array(pre.get_feature_names_out())
        self.selected_names = list(names[sel.support_])

    def transform(self, X):
        return self.sel.transform(self.pre.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))
