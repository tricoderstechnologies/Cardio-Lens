# CardioLens AI

Explainable machine-learning research and model-auditing prototype (Veritas AI International High School Hackathon 2026, AI + Healthcare).

> **This project is a prototype and is not medical advice.** It is not clinically validated and must not be used for diagnosis, screening, treatment, or to inform healthcare decisions.

## Problem statement and research question
Small clinical datasets make it easy to report a high accuracy and hide where a model fails.
**Can a transparent machine-learning model classify angiographic heart-disease presence in a small historical dataset while making its influential features, uncertainty, and subgroup error patterns explicit?**

## Objectives
Global and local explainability, uncertainty communication, false-positive/false-negative analysis, and exploratory subgroup auditing, all computed from executed code.

## Dataset and attribution
UCI Heart Disease, **Cleveland subset only** (`processed.cleveland.data`). Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., Froelicher, V. (1989). DOI: https://doi.org/10.24432/C52P4X. Licence: CC BY 4.0. Official page: https://archive.ics.uci.edu/dataset/45/heart+disease

**Provenance note:** the build sandbox could not reach archive.ics.uci.edu, so the file was downloaded from a public GitHub mirror (`StatQuest/logistic_regression_demo`). It matches the published Cleveland class counts (164/55/36/35/13, 303 rows). To use the original, replace `data/raw/processed.cleveland.data` with the file from the UCI archive and re-run the pipeline.

## Data cleaning (actual values, see `outputs/data_quality_report.md`)
- 303 records before and after cleaning (no rows dropped, no duplicates).
- Missing values (`?`): `ca` 4, `thal` 2 (6 rows). They are imputed inside the pipelines, so imputation statistics come from training folds only.
- Target: `num > 0` becomes 1. Distribution: 164 no disease (54.1%), 139 disease present (45.9%).

## Architecture and methodology
- Stratified 80/20 split, seed 42: 242 train / 61 test (28 test positives).
- Five-fold StratifiedKFold CV on the training set only.
- Preprocessing: median imputation + (Logistic Regression only) scaling for numeric; most-frequent imputation + one-hot (unknown categories ignored) for categorical.
- Models: Dummy (most frequent), Logistic Regression (L2, C=1, balanced class weights), Random Forest (300 trees, min leaf 2, balanced). Hyperparameters were fixed in advance, not tuned; the test set was never used for selection.
- Class handling: `class_weight="balanced"`; no SMOTE (classes are close to balanced).

## Results (held-out test, n=61; threshold 0.5)
| Model | Acc | Prec | Recall | F1 | Macro F1 | ROC-AUC | FPR | FNR | CV ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.541 | 0.000 | 0.000 | 0.000 | 0.351 | 0.500 | 0.000 | 1.000 | 0.500 ± 0.000 |
| Logistic Regression | 0.869 | 0.812 | 0.929 | 0.867 | 0.869 | 0.958 | 0.182 | 0.071 | 0.907 ± 0.018 |
| Random Forest | 0.902 | 0.844 | 0.964 | 0.900 | 0.902 | 0.948 | 0.152 | 0.036 | 0.900 ± 0.032 |

Confusion matrices (TN/FP/FN/TP): Baseline 33/0/28/0; Logistic Regression 27/6/2/26; Random Forest 28/5/1/27. Cross-validated accuracy was 0.847 ± 0.022 (LR) and 0.793 ± 0.014 (RF), so the training-data comparison favours Logistic Regression even though Random Forest scored higher on this one small test split. One test-set error is worth about 1.6 percentage points of accuracy, so the two models should be treated as comparable. Full numbers: `outputs/metrics.json`.

## Explainability
- **Global:** permutation importance of the Logistic Regression pipeline on the held-out test set (ROC-AUC drop, 50 repeats). Top 8: ca, cp, slope, exang, thal, oldpeak, thalach, trestbps (`outputs/permutation_importance.csv`). `ca` dominates (0.092 ± 0.036). This describes the model, not medical causation.
- **Local:** SHAP `LinearExplainer` (log-odds) on a chosen held-out example; the app lets you pick any test row. The default example is the test row whose probability is closest to 0.5 (row 172, p = 0.50, "uncertain model output").
- **Uncertainty categories (educational, not clinical risk):** <0.35 lower model likelihood within this dataset; 0.35-0.65 uncertain model output; >0.65 higher model likelihood within this dataset. Probabilities may be uncalibrated (calibration was not assessed).

## Error analysis
Logistic Regression made 6 false positives and 2 false negatives on 61 test cases. Both models err more toward false positives than false negatives at threshold 0.5; the threshold was not tuned.

## Subgroup audit (Logistic Regression, exploratory; `outputs/subgroup_metrics.csv`)
| Group | n | Positives | Recall | FNR | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| Female (sex=0) | 20 | 7 | insufficient | insufficient | insufficient | n/a |
| Male (sex=1) | 41 | 21 | 0.952 | 0.048 | 0.300 | 0.938 |
| Below 45 | 13 | 3 | insufficient | insufficient | insufficient | n/a |
| 45-59 | 30 | 14 | 0.929 | 0.071 | 0.188 | 0.987 |
| 60 and above | 18 | 11 | insufficient | insufficient | insufficient | n/a |

"Insufficient" means: *Insufficient sample size for reliable comparison.* (rule: n ≥ 30 and ≥ 10 per class; the app still shows the raw values for transparency, and the CSV holds them). No conclusions about fairness or populations can be drawn.

## Limitations
303 patients, one historical clinic, 61-patient test set (wide uncertainty, no confidence intervals computed), single split, no external validation, no calibration analysis, observational data, the dataset's sex coding only has two categories, and age groups are tiny.

## Ethical considerations
No patient data is collected. Outputs are model behaviour on public historical data, not risk to a person. Showing subgroup error patterns is for auditing, not for claiming a group is or is not treated fairly.

## Installation and execution
```bash
pip install -r requirements.txt
python -m src.prepare_data   # clean data, write quality report
python -m src.train_eval     # train, CV, evaluate, plots, metrics.json
python -m src.audit          # subgroup table
python -m src.explain        # permutation importance + local example
python -m pytest -q          # 22 tests
streamlit run src/app.py         # research dashboard
uvicorn src.api:app --port 8000  # website + live API at http://127.0.0.1:8000/
```

## Testing status
Executed and passing: data acquisition, cleaning, training, CV, evaluation, explainability, subgroup audit, 11 pytest tests, and Streamlit app launch (all four pages rendered without exceptions using Streamlit's AppTest and a headless server returned HTTP 200). Not done: visual check of the interface in a real browser.

## AI disclosure
AI tools were used to assist with code drafting, documentation, and project planning; all dataset statistics, model results, and conclusions were generated and verified from the executed project code.

## Website and API (integrated)
`src/api.py` is a FastAPI service that loads the saved scikit-learn pipelines and serves the static site in `website/` at `/`.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | liveness |
| `GET /api/results` | saved outputs of the executed project (metrics, ROC, importance, subgroups, 61 test examples) |
| `GET /api/schema` | the 13 model inputs and allowed values |
| `POST /api/predict` | live inference: probabilities from Logistic Regression and Random Forest plus SHAP contributions for Logistic Regression; blanks are imputed; out-of-range values return 422 |

Interactive docs are at `/docs`. Inputs are processed in memory and are not stored or logged by this code.

The site fetches `api/results` on load. If the API answers, the Model Playground runs the real models; if not (for example on GitHub Pages), it falls back to the saved results in `website/assets/data.js` and the playground is disabled with an explanation. To host the site and API on different origins, set `<meta name="cardiolens-api" content="https://your-api.example/">` in `website/index.html` and start the API with `CARDIOLENS_CORS=https://your-site.example`.
After re-running the pipeline, refresh the static fallback with `python website/build_data.py`.

**Deploying the live version:** the API needs a Python host (Render, Railway, Fly.io, a VM): `pip install -r requirements.txt && uvicorn src.api:app --host 0.0.0.0 --port $PORT`. Static-only hosting (GitHub Pages/Netlify/Vercel) serves the saved-results version.
