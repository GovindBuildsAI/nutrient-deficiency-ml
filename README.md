# Human Nutrient Deficiency Prediction using Machine Learning
VTU Major Project Phase-2 (BIS786) · Sir MVIT · Implementation of the Review-1 methodology (Chapter 4)


## Quick start (Windows)
1. Install **Python 3.10-3.12** from python.org (tick "Add Python to PATH").
2. Double-click `setup_windows.bat` (one time, installs packages, a few minutes).
3. Double-click `run_app_windows.bat` - the app opens at http://127.0.0.1:8000.
4. Optional tests: `.venv\Scripts\python -m pytest -q tests`

The data, trained model and built web UI are included, so no Node.js and no download is needed.
macOS/Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && PYTHONPATH=src .venv/bin/uvicorn app:app --port 8000`
(In the Run section below, replace `.venv/bin/python` with `.venv\Scripts\python` on Windows.)

## Headline result (all numbers reproduced by the code in this folder)
Nine classifiers, identical data, stratified 5-fold CV repeated 3x (15 folds), SMOTE + RFE fitted inside each training fold.
**Cohort:** 6,227 people from NHANES 2005-06, **men and women (51% female), ages 12-85**; 46 self-reportable features; 5 classes.

| Rank | Model | Macro-F1 | AUC-ROC | Accuracy |
|---|---|---|---|---|
| **1** | **Gradient Boosting** | **0.299** | **0.706** | 0.682 |
| 2 | SVM | 0.291 | 0.679 | 0.621 |
| 3 | Random Forest | 0.290 | 0.717 | 0.701 |
| 4 | MLP | 0.274 | 0.650 | 0.614 |
| 5 | XGBoost | 0.257 | 0.706 | 0.736 |
| 6 | AdaBoost | 0.240 | 0.686 | 0.382 |
| 7 | Decision Tree | 0.239 | 0.544 | 0.516 |
| 8 | Naive Bayes | 0.218 | 0.663 | 0.309 |
| 9 | KNN | 0.211 | 0.593 | 0.334 |
| ref | always predict "No Deficiency" | 0.173 | 0.500 | 0.762 |

Winner on the untouched 15% hold-out (935 people, used once): macro-F1 **0.276**, AUC **0.691**, accuracy **0.674**.

**How strongly can we say Gradient Boosting is "best"?** Honestly: it is first on the documented criterion (macro-F1, AUC tie-break), but
* it is **statistically tied with SVM and Random Forest** (Holm-corrected p = 0.50 for each) and only just ahead of MLP (p = 0.045); it clearly beats XGBoost, AdaBoost, Decision Tree, Naive Bayes and KNN (p < 0.001). Repeated-CV folds overlap, so the p-values are optimistic.
* in the sensitivity study (6 SMOTE/feature-budget settings, `results/sensitivity_*.csv`) the winner changes: **SVM is the most consistent** (mean rank 2.3, best in 3 of 6 settings), Gradient Boosting averages rank 3.3. (The sensitivity study runs only the first of the three CV repeats, 5 folds, so its "documented" setting differs slightly from the 15-fold main table: on that one repeat Random Forest 0.304 and SVM 0.300 sit just above Gradient Boosting 0.295.) The defensible statement is "Gradient Boosting ranks first under the documented protocol; Gradient Boosting, SVM and Random Forest are practically indistinguishable on this data".

## What the results do NOT show - say this before an examiner does
* **The 80% accuracy target (Objective 2) is not met** - accuracy is 61-74% for the top models, and *below* the 76.2% you get by always guessing "No Deficiency". SMOTE trades overall accuracy for finding minority deficiencies, so macro-F1/AUC are the right yardsticks. With only self-reportable inputs, predicting blood-measured deficiency is genuinely hard (AUC ~0.7 = clearly better than chance, not diagnostic).
* Vitamin A (126 people) and Vitamin C (201) are small classes; per-class F1 for Vitamin A is only ~0.08.
* SMOTE does help: mean macro-F1 over the nine models 0.259 with SMOTE vs 0.227 without.

## Deviations from the Review-1 document (all forced by the data, none hidden)
1. **Classes: 5, not 9.** Labels come from NHANES 2005-06 blood tests with standard cut-offs (`src/build_dataset.py`): Vitamin D, Vitamin A, Vitamin C, B-Vitamins, No Deficiency.
   * **Zinc** is only measured in NHANES 2011-16 and **magnesium** serum is not in NHANES at all.
   * **Calcium** had 7 cases (serum calcium is tightly regulated) - too few to learn.
   * **Iron was dropped on purpose.** In this survey round the iron markers (ferritin, transferrin saturation) were measured only in females; requiring them gave an all-female cohort (2,549 women). Dropping Iron keeps both sexes (6,227 people).
   * **Vitamin A** uses the WHO "inadequate" cut-off (<30 ug/dL) because true deficiency had too few cases.
2. **One primary label per person.** 323 of 6,227 people fall below more than one cut-off; the label is the nutrient with the largest relative shortfall (the document's argmax design cannot express multi-label).
3. **Leak-free order.** The document runs SMOTE and feature selection before cross-validation. Doing so would leak validation data into training, so here preprocessing -> SMOTE -> RFE are fitted inside every training fold (`src/pipeline.py`, `tests/test_core.py`). Repeated 5-fold CV (x3) and a hold-out set were added; the F1-then-AUC rule uses "tie = within 0.005 F1 of the leader".
4. **SHAP:** exact TreeSHAP is used when the installed `shap` supports the winning model. The current winner (multi-class sklearn Gradient Boosting) is not supported, so the code falls back to KernelSHAP (still Shapley values; additivity verified to 0.0 error).
5. **Storage:** submissions go to SQLite (`data/submissions.sqlite`), not PostgreSQL, and there is no Docker file - neither is installed on the development Mac, and I did not ship untested config. The code touches the DB in one place (`src/app.py`).
6. **Hyper-parameters** are scikit-learn/XGBoost defaults (no tuning), as agreed.
7. **Population:** NHANES 2005-06 (US), ages 12-85, N = 6,227, 51% female. Features are the 46 self-reportable fields in `src/schema.py`; the dietary ones are 24-hour-recall nutrient intakes, not a food-frequency questionnaire. Not validated outside this population.
8. **Race/ethnicity and country of birth removed** at the team's request (sensitive attributes). Race had been the top SHAP feature in an early experiment, so removing it costs some accuracy: part of that earlier signal was a population-level demographic correlate, not a modifiable risk factor.

## Data-quality issues found and fixed during the build
Checked against the CDC codebooks: several NHANES sun-exposure variables were first mis-named (they are shade/sleeve/sunburn questions), one dietary column was *added* B12 not total B12, and SAS zero/missing placeholders (5.4e-79) had to be handled. Lab labels were never affected. The full pipeline was rerun after each fix. Most recently the cohort was widened from females-only to both sexes (deviation 1).

## Run it
```bash
cd "College Project Phase Two"
scripts/download_data.sh                                   # 19 public CDC files, ~30 MB (resumable)
.venv/bin/python src/build_dataset.py                      # builds data/processed/nhanes_2005_06_modelling.csv
PYTHONPATH=src .venv/bin/python src/train_compare.py       # ~12 min: 9 models x 15 folds, selection, hold-out
PYTHONPATH=src .venv/bin/python src/sensitivity.py        # ~15-25 min
PYTHONPATH=src .venv/bin/python src/make_figures.py && PYTHONPATH=src .venv/bin/python src/global_shap.py
.venv/bin/python -m pytest -q tests                        # 11 tests
./run_app.sh                                               # web app at http://127.0.0.1:8000
```
`data/raw` also holds four unused CDC files (`FERTIN_D`, `TFR_D`, `FETIB_D`, `OCQ_D`) left over from earlier experiments; the pipeline reads only the 19 files the download script fetches. A from-scratch re-download has not been re-tested since the iron files were removed.

(xgboost needs the libomp copy already placed inside `.venv`; see the guide, Part F step 4. Rebuild the UI with `cd webapp && npm run build`.)

## Folder map
`src/` pipeline, training, SHAP/report, API · `webapp/` React UI · `data/raw` NHANES XPT files (CDC, public domain) · `data/processed` modelling table · `results/` metrics, selection.json, sensitivity, `figures/` (11 slide-ready PNGs) · `models/best_model.joblib` · `tests/` · `docs/` guide generator · `Software_Guide.pdf`

## Likely questions
* *Why is accuracy below the majority baseline?* SMOTE-balanced training; macro-F1/AUC chosen in Algorithm 5 for exactly this reason.
* *Why only 5 classes?* No public dataset measures all nine together, and Iron was dropped to keep both sexes; see deviation 1.
* *Why not use race/ethnicity?* Sensitive attribute and a population-level correlate, not a cause; removing it costs some accuracy.
* *Why not 80%?* Self-report-only features; blood-measured labels; see "What the results do NOT show".
