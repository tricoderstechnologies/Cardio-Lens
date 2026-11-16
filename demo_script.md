# Two-minute demo script (note: the brief you sent was cut off at "15–", so this timing is my own)

**0:00-0:15 Problem.** "Small medical datasets make models look better than they are. Our question: can a transparent model classify heart-disease presence in 303 historical patients while showing its features, uncertainty and where it fails? This is a research prototype, not medical advice."

**0:15-0:40 Overview page.** Show dataset counts (303 records, 139 positive, 6 missing cells), the train/test split and the limitations box.

**0:40-1:05 Model Audit.** "A dummy baseline gets 54% accuracy. Logistic Regression gets 87% with ROC-AUC 0.958; Random Forest 90%. But cross-validation favoured Logistic Regression, and the gap is about two patients, so we call them comparable." Point at confusion matrices: 6 false positives, 2 false negatives.

**1:05-1:30 Explainability.** "Permutation importance says ca, cp and slope matter most to this model. That is not causation." Select the default example: probability 0.50, "uncertain model output", and show the SHAP bars.

**1:30-1:50 Subgroup Audit.** "Only males and ages 45-59 have enough cases. The rest show: Insufficient sample size for reliable comparison. We refuse to draw fairness conclusions."

**1:50-2:00 Close.** "CardioLens shows how to report a model honestly: results, uncertainty and weaknesses. Not validated, not for diagnosis, screening, treatment or healthcare decisions."
