"""Module 0 - build the NHANES 2005-06 modelling table.

Features are what a person could plausibly self-report (demographics, body
measures, typical diet, lifestyle, sun exposure). Labels come from LAB values
that the model never sees, using standard clinical cut-offs.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
rd = lambda f: pd.read_sas(RAW / f"{f}.xpt", format="xport")

# ---- clinical cut-offs (below = deficient) ---------------------------------
CUTOFFS = {
    "Vitamin D":  ("LBDVIDMS", 30.0, "25(OH)D < 30 nmol/L (IOM deficiency)"),
    "Vitamin A":  ("LBXVIA",   30.0, "serum retinol < 30 ug/dL (1.05 umol/L, WHO 'inadequate')"),
    "Vitamin C":  ("LBXVIC",    0.2, "serum vitamin C < 0.2 mg/dL (11.4 umol/L, deficiency)"),
}
B_CUTOFFS = {"LBXB12": 200.0, "LBXFOL": 3.0, "LBXPLP": 20.0}   # B12 pg/mL, folate ng/mL, PLP nmol/L

def load():
    t = rd("DEMO_D")[["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1","DMDEDUC2","INDFMPIR","RIDEXPRG","DMDBORN"]]
    for f in ["BMX_D","VID_D","VITAEC_D","VIC_D","FOLATE_D","B12_D","VIT_B6_D",
              "DR1TOT_D","SMQ_D","ALQ_D","PAQ_D","DEQ_D","MCQ_D","DBQ_D","BPQ_D","HSQ_D","DIQ_D"]:
        t = t.merge(rd(f), on="SEQN", how="left")
    # pandas reads SAS XPT zeros/special-missing as +-5.4e-79. In the dietary file these occur only for people
    # with a valid recall (never for people without one) and there are no true zeros, so they mean "zero intake".
    # Anywhere else they would be special-missing, i.e. unknown.
    num = t.select_dtypes("number").columns
    tiny = (t[num].abs() < 1e-60) & (t[num] != 0)
    for c in num:
        t.loc[tiny[c], c] = 0.0 if c.startswith("DR1T") else np.nan
    return t

def label(t):
    t = t.copy()
    need = ["LBDVIDMS","LBXVIA","LBXVIC","LBXB12","LBXFOL","LBXPLP"]
    t = t.dropna(subset=need)
    t = t[t.RIDAGEYR >= 12]
    # Iron is deliberately NOT a class: in this survey round ferritin / transferrin saturation were measured in females only,
    # so requiring them would give an all-female cohort. Dropping Iron keeps both sexes (every person aged 12+ with the six labs above).
    short = pd.DataFrame(index=t.index)
    for name,(col,thr,_) in CUTOFFS.items():
        short[name] = (1 - t[col]/thr).clip(lower=0)
    short["B-Vitamins"] = pd.concat([(1 - t[c]/thr).clip(lower=0) for c,thr in B_CUTOFFS.items()], axis=1).max(axis=1)
    # a nutrient is deficient only if its shortfall is > 0; primary = largest relative shortfall
    primary = short.idxmax(axis=1)
    t["deficiency"] = np.where(short.max(axis=1) > 0, primary, "No Deficiency")
    t["n_deficient"] = (short > 0).sum(axis=1)
    return t

# ---- self-reportable feature spec -----------------------------------------
NUMERIC = {  # NHANES code -> readable name
    "RIDAGEYR":"age_years","INDFMPIR":"income_poverty_ratio","BMXBMI":"bmi","BMXWAIST":"waist_cm","BMXHT":"height_cm","BMXWT":"weight_kg",
    "DR1TKCAL":"energy_kcal","DR1TPROT":"protein_g","DR1TCARB":"carbs_g","DR1TSUGR":"sugar_g","DR1TFIBE":"fibre_g","DR1TTFAT":"fat_g",
    "DR1TSFAT":"sat_fat_g","DR1TCHOL":"cholesterol_mg","DR1TVARA":"intake_vitA_mcg","DR1TVC":"intake_vitC_mg","DR1TATOC":"intake_vitE_mg",
    "DR1TFOLA":"intake_folate_mcg","DR1TVB12":"intake_b12_mcg","DR1TVB6":"intake_b6_mg","DR1TCALC":"intake_calcium_mg","DR1TIRON":"intake_iron_mg",
    "DR1TZINC":"intake_zinc_mg","DR1TMAGN":"intake_magnesium_mg","DR1TCAFF":"caffeine_mg","DR1TALCO":"alcohol_g","DR1TSODI":"sodium_mg",
    "DR1TPOTA":"potassium_mg",
}
ORDINAL = {  # ordered answer scales (label-encoded, kept ordered)
    "DMDEDUC2":"education_level","SMQ040":"smoking_now","PAQ180":"daily_activity_level",
    "DED031":"skin_reaction_to_sun","DEQ034A":"stays_in_shade","DEQ034C":"wears_long_sleeves",
    "HSD010":"general_health","DBQ700":"self_rated_diet","DBQ197":"milk_frequency",
}
NOMINAL = {  # unordered categories (one-hot) and yes/no flags
    "RIAGENDR":"gender","RIDEXPRG":"pregnancy","SMQ020":"smoked_100_cigs","ALQ101":"drinks_alcohol",
    "DEQ038G":"sunburn_past_year","MCQ010":"asthma","MCQ053":"treated_for_anaemia","BPQ020":"high_blood_pressure","DIQ010":"diabetes",
}
NO_ANSWER = {7, 9, 77, 99, 777, 999, 7777, 9999}   # NHANES "refused / don't know"
# answer codes that are not part of the ordered scale (checked against the CDC 2005-06 codebooks)
NOT_ON_SCALE = {"DED031": {6}, "DEQ034A": {6}, "DBQ197": {4}}   # "Other" / "don't go out in the sun" / "Varied"

def features(t):
    X = pd.DataFrame(index=t.index)
    for c, n in NUMERIC.items():
        X[n] = t[c].astype(float)
    for c, n in {**ORDINAL, **NOMINAL}.items():
        v = t[c].astype(float)
        X[n] = v.where(~v.isin(NO_ANSWER | NOT_ON_SCALE.get(c, set())), np.nan)
    return X

def build():
    t = label(load())
    X = features(t)
    X["deficiency"] = t["deficiency"].values
    X["n_deficient"] = t["n_deficient"].values
    X.insert(0, "SEQN", t["SEQN"].astype(int).values)
    X.to_csv(OUT / "nhanes_2005_06_modelling.csv", index=False)
    return X

if __name__ == "__main__":
    X = build()
    print("rows", len(X), "features", X.shape[1] - 3)
    print(X.deficiency.value_counts()); print("multi-deficient:", (X.n_deficient > 1).sum())
    print("missing % (top):"); print((X.isna().mean() * 100).round(1).sort_values(ascending=False).head(8))
