"""Curated dietary-recommendation knowledge base (Algorithm 7, step 4).

General public-health nutrition guidance only - it supports, and never replaces, a blood test
and advice from a clinician. Each entry also records the clinical cut-off used to label the class.
"""

KB = {
    "B-Vitamins": dict(
        title="Low B-vitamin status (B12, folate or B6)",
        cutoff="Serum B12 < 200 pg/mL, or serum folate < 3 ng/mL, or plasma PLP (B6) < 20 nmol/L",
        summary="B12, folate and B6 are needed for red blood cells and the nervous system. Vegetarians and vegans are at particular risk of low B12.",
        foods=["B12: eggs, dairy, fish, meat, and fortified cereals or plant milks", "Folate: leafy greens, lentils, beans, citrus fruit and fortified grains", "B6: chickpeas, potatoes, bananas, poultry and fish"],
        tips=["If you follow a vegan diet, a B12 supplement or fortified foods are usually essential.", "Alcohol and some medicines (e.g. metformin) can lower B-vitamin levels."],
        see_doctor="Tingling or numbness in hands and feet, memory problems or unusual fatigue: ask for a B12 / folate test. Folate is especially important if you may become pregnant."),
    "Vitamin D": dict(
        title="Low vitamin D status",
        cutoff="Serum 25(OH)D < 30 nmol/L (12 ng/mL)",
        summary="Vitamin D supports bones and muscles and is made in the skin from sunlight. Few foods contain much of it.",
        foods=["Oily fish (salmon, mackerel, sardines)", "Egg yolks", "Vitamin-D-fortified milk, cereals and spreads", "UV-exposed mushrooms"],
        tips=["Short, regular sun exposure of face and arms around midday helps - balance this against sunburn risk.", "Darker skin, covering clothing, high latitude and indoor work all reduce vitamin D production."],
        see_doctor="Bone or muscle pain, or frequent fractures: ask about a 25(OH)D blood test. Supplement dose should be guided by a clinician."),
    "Vitamin C": dict(
        title="Low vitamin C status",
        cutoff="Serum vitamin C < 0.2 mg/dL (11.4 µmol/L)",
        summary="Vitamin C is needed for healing, gums and iron absorption. Smoking increases the body's need for it.",
        foods=["Citrus fruit, amla, guava, kiwi and papaya", "Bell peppers, tomatoes, broccoli and cabbage", "Strawberries and other berries"],
        tips=["Eat fresh fruit or vegetables daily - vitamin C is lost with long cooking.", "Smokers need more vitamin C; stopping smoking is the best fix."],
        see_doctor="Bleeding gums, easy bruising or slow wound healing: see a clinician."),
    "Vitamin A": dict(
        title="Low vitamin A status",
        cutoff="Serum retinol < 30 µg/dL (1.05 µmol/L) - the WHO 'inadequate' range, not severe deficiency",
        summary="Vitamin A supports vision, skin and immunity. In this population a low level is usually mild.",
        foods=["Orange and yellow vegetables and fruit: carrots, sweet potato, pumpkin, mango, papaya", "Dark leafy greens", "Eggs and dairy; liver (small, occasional portions)"],
        tips=["Eat these foods with a little fat - it improves absorption of vitamin A.", "Avoid high-dose vitamin A supplements unless prescribed, especially in pregnancy."],
        see_doctor="Difficulty seeing in dim light or very dry eyes: see a clinician promptly."),
    "No Deficiency": dict(
        title="No deficiency pattern detected",
        cutoff="No measured nutrient fell below its cut-off",
        summary="The model did not match your profile to any of the deficiency patterns it learned.",
        foods=["A varied diet with vegetables, fruit, pulses, whole grains, dairy or alternatives, and protein foods"],
        tips=["Keep a varied diet and regular physical activity.", "This is a screening estimate - a blood test is the only way to confirm nutrient status."],
        see_doctor="If you have symptoms such as persistent fatigue, a blood test is more reliable than this screening tool."),
}

DISCLAIMER = ("This tool is an educational screening aid built for a student project. It was trained on a US survey population aged 12-85 and has not been validated elsewhere. It estimates risk from self-reported "
              "information and is NOT a diagnosis. Only a blood test can confirm a nutrient deficiency. "
              "Please consult a qualified clinician before changing your diet or taking supplements.")
