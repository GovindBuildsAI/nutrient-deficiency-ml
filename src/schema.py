"""Human-readable description of every model input: label, group, units, and answer options.

Answer codes are NHANES 2005-06 codes, checked against the CDC codebooks, so a form
answer goes straight into the model without translation.
"""
from build_dataset import NOMINAL, NUMERIC, ORDINAL

YES_NO = [(1, "Yes"), (2, "No")]
FREQ5 = [(1, "Always"), (2, "Most of the time"), (3, "Sometimes"), (4, "Rarely"), (5, "Never")]
HEALTH5 = [(1, "Excellent"), (2, "Very good"), (3, "Good"), (4, "Fair"), (5, "Poor")]

# name -> (group, label, kind, extra)   kind: "choice" (options) or "number" (unit, min, max)
FIELDS = {
    # --- About you
    "age_years":            ("About you", "Age", "number", dict(unit="years", min=12, max=85, help="The model was trained on US people aged 12-85.")),
    "gender":               ("About you", "Sex", "choice", dict(options=[(1, "Male"), (2, "Female")])),
    "pregnancy":            ("About you", "Currently pregnant?", "choice", dict(options=[(1, "Yes"), (2, "No"), (3, "Not sure")], help="Females only; leave blank otherwise.")),
    "education_level":      ("About you", "Highest education (adults 20+)", "choice", dict(options=[(1, "Less than 9th grade"), (2, "9-11th grade"), (3, "High school / GED"), (4, "Some college / associate degree"), (5, "College graduate or above")])),
    "income_poverty_ratio": ("About you", "Family income-to-poverty ratio", "number", dict(unit="ratio 0-5", min=0, max=5, help="Optional. 1.0 = at the poverty line; 5 = 5x or more.")),
    # --- Body
    "height_cm":            ("Body measures", "Height", "number", dict(unit="cm", min=100, max=220)),
    "weight_kg":            ("Body measures", "Weight", "number", dict(unit="kg", min=25, max=250)),
    "bmi":                  ("Body measures", "BMI", "number", dict(unit="kg/m²", min=12, max=70, computed=True, help="Calculated from height and weight if left blank.")),
    "waist_cm":             ("Body measures", "Waist circumference", "number", dict(unit="cm", min=40, max=180)),
    # --- Lifestyle
    "smoked_100_cigs":      ("Lifestyle", "Smoked at least 100 cigarettes in your life?", "choice", dict(options=YES_NO)),
    "smoking_now":          ("Lifestyle", "Do you smoke now?", "choice", dict(options=[(1, "Every day"), (2, "Some days"), (3, "Not at all")])),
    "drinks_alcohol":       ("Lifestyle", "Had at least 12 alcoholic drinks in a year?", "choice", dict(options=YES_NO)),
    "daily_activity_level": ("Lifestyle", "Usual daily activity", "choice", dict(options=[(1, "Mostly sitting"), (2, "Standing / walking a lot"), (3, "Lifting light loads / climbing"), (4, "Heavy work / carrying heavy loads")])),
    "general_health":       ("Lifestyle", "General health", "choice", dict(options=HEALTH5)),
    "self_rated_diet":      ("Lifestyle", "How healthy is your diet?", "choice", dict(options=HEALTH5)),
    "milk_frequency":       ("Lifestyle", "Milk to drink or on cereal (past 30 days)", "choice", dict(options=[(0, "Never"), (1, "Rarely (< once a week)"), (2, "Sometimes (weekly)"), (3, "Often (daily or more)")])),
    # --- Sun exposure
    "skin_reaction_to_sun": ("Sun exposure", "Skin after 30 min in sun, no protection (after months indoors)", "choice", dict(options=[(1, "Severe sunburn with blisters"), (2, "Severe sunburn with peeling"), (3, "Mild burn with some tanning"), (4, "Turns darker, no burn"), (5, "Nothing happens")])),
    "stays_in_shade":       ("Sun exposure", "Stay in the shade on a sunny day (>1 hour)", "choice", dict(options=FREQ5)),
    "wears_long_sleeves":   ("Sun exposure", "Wear a long-sleeved shirt in the sun", "choice", dict(options=FREQ5)),
    "sunburn_past_year":    ("Sun exposure", "Sunburn in the past year?", "choice", dict(options=[(1, "Yes, at least once"), (2, "Never")])),
    # --- Health history
    "asthma":               ("Health history", "Ever told you have asthma?", "choice", dict(options=YES_NO)),
    "high_blood_pressure":  ("Health history", "Ever told you have high blood pressure?", "choice", dict(options=YES_NO)),
    "diabetes":             ("Health history", "Ever told you have diabetes?", "choice", dict(options=[(1, "Yes"), (2, "No"), (3, "Borderline")])),
    "treated_for_anaemia":  ("Health history", "Treated for anaemia in the last 3 months?", "choice", dict(options=YES_NO)),
}

# typical-day diet (24-hour recall style). Optional - blank values are imputed with the training median.
DIET = {
    "energy_kcal": ("Energy", "kcal"), "protein_g": ("Protein", "g"), "carbs_g": ("Carbohydrate", "g"), "sugar_g": ("Sugars", "g"),
    "fibre_g": ("Fibre", "g"), "fat_g": ("Total fat", "g"), "sat_fat_g": ("Saturated fat", "g"), "cholesterol_mg": ("Cholesterol", "mg"),
    "intake_vitA_mcg": ("Vitamin A (RAE)", "mcg"), "intake_vitC_mg": ("Vitamin C", "mg"), "intake_vitE_mg": ("Vitamin E", "mg"),
    "intake_folate_mcg": ("Folate", "mcg"), "intake_b12_mcg": ("Vitamin B12", "mcg"), "intake_b6_mg": ("Vitamin B6", "mg"),
    "intake_calcium_mg": ("Calcium", "mg"), "intake_iron_mg": ("Iron", "mg"), "intake_zinc_mg": ("Zinc", "mg"),
    "intake_magnesium_mg": ("Magnesium", "mg"), "caffeine_mg": ("Caffeine", "mg"), "alcohol_g": ("Alcohol", "g"),
    "sodium_mg": ("Sodium", "mg"), "potassium_mg": ("Potassium", "mg"),
}
for k, (label, unit) in DIET.items():
    FIELDS[k] = ("Typical daily diet (optional)", f"{label} per day", "number", dict(unit=unit, min=0, max=100000))

ALL_COLS = list(NUMERIC.values()) + list(ORDINAL.values()) + list(NOMINAL.values())
assert set(FIELDS) == set(ALL_COLS), set(FIELDS) ^ set(ALL_COLS)   # schema must cover exactly the model inputs


def ui_schema():
    out = []
    for name, (group, label, kind, extra) in FIELDS.items():
        f = dict(name=name, group=group, label=label, kind=kind)
        f.update({k: ([{"value": v, "label": l} for v, l in x] if k == "options" else x) for k, x in extra.items()})
        out.append(f)
    return out


def option_label(name, value):
    group, label, kind, extra = FIELDS[name]
    if kind == "choice":
        for v, l in extra["options"]:
            if float(v) == float(value):
                return l
    return None
