# CardioLens AI: Auditing a Small Heart-Disease Classifier
Veritas AI International High School Hackathon 2026 | AI + Healthcare

**This project is a prototype and is not medical advice.** It is not clinically validated; it must not be used for diagnosis, screening, treatment, or to inform healthcare decisions.

**Question.** Can a transparent model classify angiographic heart-disease presence in a small historical dataset while making influential features, uncertainty and subgroup errors explicit?

**Data.** UCI Cleveland Heart Disease (CC BY 4.0, DOI 10.24432/C52P4X): 303 records, 6 missing values (`ca` 4, `thal` 2), target num>0: 139 positive / 164 negative. Stratified 80/20 split (242/61), 5-fold CV on training data only, seed 42.

**Methods.** Dummy baseline, L2 Logistic Regression (primary), Random Forest; imputation and scaling inside pipelines; permutation importance and SHAP for explanation; subgroup audit with sample-size safeguards.

**Results (test, n=61).** Baseline accuracy 0.541, ROC-AUC 0.500. Logistic Regression accuracy 0.869, recall 0.929, ROC-AUC 0.958 (CV ROC-AUC 0.907 ± 0.018; 6 FP, 2 FN). Random Forest accuracy 0.902, recall 0.964, ROC-AUC 0.948 (CV 0.900 ± 0.032; 5 FP, 1 FN). Cross-validation accuracy favoured Logistic Regression (0.847 vs 0.793), and the test gap equals about two patients, so the models are comparable. Top permutation-importance features: ca, cp, slope, exang, thal.

**Subgroups.** Only male (n=41) and age 45-59 (n=30) met the reliability rule; female, below 45 and 60+ are flagged *Insufficient sample size for reliable comparison.*

**Limitations.** One clinic, 303 patients, one small test split, no external validation or calibration check; importance is not causation; subgroup results are exploratory and support no fairness claim.

**AI disclosure.** AI tools were used to assist with code drafting, documentation, and project planning; all dataset statistics, model results, and conclusions were generated and verified from the executed project code.
