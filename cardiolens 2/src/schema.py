"""Single source of truth for the 13 model inputs (UCI Cleveland data dictionary)."""
SPECS = [
    {"name": "age", "label": "Age (years)", "kind": "num", "min": 20, "max": 100, "step": 1},
    {"name": "sex", "label": "Sex (dataset encoding)", "kind": "cat", "options": [[0, "0 = female"], [1, "1 = male"]]},
    {"name": "cp", "label": "Chest pain type", "kind": "cat", "options": [[1, "1 = typical angina"], [2, "2 = atypical angina"], [3, "3 = non-anginal pain"], [4, "4 = asymptomatic"]]},
    {"name": "trestbps", "label": "Resting blood pressure (mm Hg)", "kind": "num", "min": 80, "max": 220, "step": 1},
    {"name": "chol", "label": "Serum cholesterol (mg/dl)", "kind": "num", "min": 100, "max": 600, "step": 1},
    {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dl", "kind": "cat", "options": [[0, "0 = no"], [1, "1 = yes"]]},
    {"name": "restecg", "label": "Resting ECG result", "kind": "cat", "options": [[0, "0 = normal"], [1, "1 = ST-T abnormality"], [2, "2 = LV hypertrophy"]]},
    {"name": "thalach", "label": "Maximum heart rate achieved", "kind": "num", "min": 60, "max": 220, "step": 1},
    {"name": "exang", "label": "Exercise-induced angina", "kind": "cat", "options": [[0, "0 = no"], [1, "1 = yes"]]},
    {"name": "oldpeak", "label": "ST depression (exercise vs rest)", "kind": "num", "min": 0, "max": 7, "step": 0.1},
    {"name": "slope", "label": "Slope of peak-exercise ST segment", "kind": "cat", "options": [[1, "1 = upsloping"], [2, "2 = flat"], [3, "3 = downsloping"]]},
    {"name": "ca", "label": "Major vessels coloured by fluoroscopy (0-3)", "kind": "num", "min": 0, "max": 3, "step": 1},
    {"name": "thal", "label": "Thallium test result", "kind": "cat", "options": [[3, "3 = normal"], [6, "6 = fixed defect"], [7, "7 = reversible defect"]]},
]
BY_NAME = {s["name"]: s for s in SPECS}
