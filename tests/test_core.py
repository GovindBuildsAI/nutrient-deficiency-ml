import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import build_dataset as b                      # noqa: E402
import pipeline as P                           # noqa: E402


def lab_row(**kw):
    base = dict(SEQN=1, RIDAGEYR=30, LBDVIDMS=80, LBXVIA=50, LBXVIC=1.0, LBXB12=500, LBXFOL=10, LBXPLP=50,
                LBXFER=60, LBDPCT=30)
    base.update(kw)
    return base


def test_label_none_when_all_labs_normal():
    assert b.label(pd.DataFrame([lab_row()])).deficiency.iloc[0] == "No Deficiency"


@pytest.mark.parametrize("lab,expect", [
    (dict(LBDVIDMS=20), "Vitamin D"), (dict(LBXVIC=0.1), "Vitamin C"),
    (dict(LBXB12=150), "B-Vitamins"), (dict(LBXVIA=25), "Vitamin A")])
def test_label_each_cutoff(lab, expect):
    assert b.label(pd.DataFrame([lab_row(**lab)])).deficiency.iloc[0] == expect


def test_label_primary_is_largest_relative_shortfall():
    # vit D 29 vs 30 (3% short), vit C 0.05 vs 0.2 (75% short) -> Vitamin C is primary, both counted
    t = b.label(pd.DataFrame([lab_row(LBDVIDMS=29, LBXVIC=0.05)]))
    assert t.deficiency.iloc[0] == "Vitamin C" and t.n_deficient.iloc[0] == 2


def test_iron_is_not_a_class_and_does_not_restrict_the_cohort():
    # no ferritin / transferrin values at all must not drop the person (that would make the cohort female-only)
    t = b.label(pd.DataFrame([lab_row(LBXFER=np.nan, LBDPCT=np.nan)]))
    assert len(t) == 1 and "Iron" not in set(t.deficiency)


def test_rows_without_required_labs_are_dropped():
    assert len(b.label(pd.DataFrame([lab_row(LBXB12=np.nan)]))) == 0


def test_preprocessor_is_fitted_on_training_data_only():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data/processed/nhanes_2005_06_modelling.csv")
    X = df[P.NUM_COLS + P.CAT_COLS]; train = X.iloc[:1000]
    pre = P.make_preprocessor().fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    i = P.NUM_COLS.index("age_years")
    assert scaler.mean_[i] == pytest.approx(train.age_years.mean())          # not the full-data mean
    assert scaler.mean_[i] != pytest.approx(X.age_years.mean(), abs=1e-6)


def test_smote_changes_only_the_training_set_size():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data/processed/nhanes_2005_06_modelling.csv")
    X, y = df[P.NUM_COLS + P.CAT_COLS], pd.factorize(df.deficiency)[0]
    pre, sel, Zb, yb = P.prepare_fold(X.iloc[:1200], y[:1200])
    assert Zb.shape[1] == P.P_STAR and len(set(np.bincount(yb))) == 1          # balanced, p* features kept
    assert sel.transform(pre.transform(X.iloc[1200:1210])).shape == (10, P.P_STAR)


def test_every_model_input_has_a_ui_definition():
    import schema
    assert set(schema.FIELDS) == set(P.NUM_COLS + P.CAT_COLS)
