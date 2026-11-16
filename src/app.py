"""CardioLens AI - Streamlit research interface. Every number is read from executed outputs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.audit import INSUFFICIENT, PRIMARY
from src.explain import (load_primary, local_explanation, plot_local, uncertainty_category)
from src.prepare_data import FEATURES, PROCESSED, load_raw
from src.train_eval import MODEL_NAMES, get_split

OUT = ROOT / "outputs"
DISCLAIMER = "This project is a prototype and is not medical advice."

st.set_page_config(page_title="CardioLens AI", layout="wide")
st.markdown("""<style>
.block-container{max-width:1100px;padding-top:2rem}
h1,h2,h3{font-weight:600;letter-spacing:-0.01em}
</style>""", unsafe_allow_html=True)

if not (OUT / "metrics.json").exists():
    st.error("Outputs not found. Run: python -m src.prepare_data && python -m src.train_eval "
             "&& python -m src.audit && python -m src.explain")
    st.stop()


@st.cache_data
def load_all():
    metrics = json.loads((OUT / "metrics.json").read_text())
    df = pd.read_csv(PROCESSED)
    return metrics, df, load_raw(), pd.read_csv(OUT / "test_predictions.csv"), \
        pd.read_csv(OUT / "subgroup_metrics.csv"), pd.read_csv(OUT / "permutation_importance.csv")


metrics, df, raw, preds, sub, imp = load_all()
page = st.sidebar.radio("Navigate", ["1. Overview", "2. Model Audit", "3. Explainability",
                                     "4. Subgroup Audit"])
st.sidebar.caption(DISCLAIMER)


def fmt(v):
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}"


if page.startswith("1"):
    st.title("CardioLens AI")
    st.caption("Explainable machine-learning research and model-auditing prototype | "
               "Veritas AI International High School Hackathon 2026 | Track: AI + Healthcare")
    st.warning(DISCLAIMER + " It is not clinically validated and must not be used for diagnosis, "
               "screening, treatment, or to inform healthcare decisions.")
    st.subheader("Research question")
    st.write("Can a transparent machine-learning model classify angiographic heart-disease presence "
             "in a small historical dataset while making its influential features, uncertainty, "
             "and subgroup error patterns explicit?")
    st.subheader("Objective")
    st.write("Evaluate models transparently: global and local explainability, uncertainty "
             "communication, false-positive/false-negative analysis, and exploratory subgroup auditing.")
    st.subheader("Dataset")
    vc = df["target"].value_counts().sort_index()
    c = st.columns(5)
    c[0].metric("Records (raw)", len(raw)); c[1].metric("Records (clean)", len(df))
    c[2].metric("No disease (0)", int(vc.get(0, 0))); c[3].metric("Disease present (1)", int(vc.get(1, 0)))
    c[4].metric("Missing cells", int(raw.isna().sum().sum()))
    st.caption("UCI Heart Disease, Cleveland subset. Detrano et al. DOI 10.24432/C52P4X. CC BY 4.0. "
               "Target: num > 0 mapped to 1. Cleveland only; other hospital subsets are not used.")
    st.write("Missing values by column:", raw.isna().sum()[raw.isna().sum() > 0].to_dict())
    st.subheader("Machine-learning architecture")
    st.markdown(f"""
- Stratified 80/20 split (seed {metrics['seed']}): {metrics['n_train']} train / {metrics['n_test']} test.
- 5-fold stratified cross-validation on the training set only.
- Preprocessing inside pipelines: median imputation (numeric), most-frequent imputation and one-hot
  encoding with unknown-category handling (categorical), scaling for Logistic Regression.
- Models: Dummy baseline, L2 Logistic Regression (primary), Random Forest (comparison).
- The test set was not used for model selection, tuning, or feature selection.
""")
    st.subheader("Important limitations")
    st.markdown(f"""
- Only {len(df)} records from one historical clinic; the {metrics['n_test']}-patient test set gives wide uncertainty.
- Results do not generalise to other populations or to current clinical practice.
- Feature importance describes the model, not medical causation.
- Probabilities may not be calibrated. Subgroup results are exploratory.
""")

elif page.startswith("2"):
    st.title("Model audit")
    st.caption(f"Held-out test set: n={metrics['n_test']} ({metrics['test_positive']} positive). "
               "Cross-validation uses training data only. Threshold 0.5.")
    rows = []
    for n in MODEL_NAMES:
        t, cv = metrics["models"][n]["test"], metrics["models"][n]["cv"]
        rows.append({"Model": n, "Accuracy": t["accuracy"], "Precision": t["precision"],
                     "Recall": t["recall"], "F1": t["f1"], "Macro F1": t["macro_f1"],
                     "ROC-AUC": t["roc_auc"], "FPR": t["false_positive_rate"],
                     "FNR": t["false_negative_rate"],
                     "CV ROC-AUC (mean ± sd)": f"{cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f}",
                     "CV accuracy (mean ± sd)": f"{cv['accuracy']['mean']:.3f} ± {cv['accuracy']['std']:.3f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format(precision=3), width="stretch")
    st.image(str(OUT / "model_comparison.png"))
    a, b = st.columns(2)
    a.image(str(OUT / "roc_curve.png"))
    with b:
        st.markdown("**Confusion counts (test)**")
        st.dataframe(pd.DataFrame({n: {k: metrics["models"][n]["test"][k] for k in ["tn", "fp", "fn", "tp"]}
                                   for n in MODEL_NAMES}), width="stretch")
    st.image(str(OUT / "confusion_matrix.png"))
    st.subheader("Error analysis")
    for n in MODEL_NAMES[1:]:
        t = metrics["models"][n]["test"]
        st.markdown(f"**{n}:** {t['fp']} false positives (FPR {fmt(t['false_positive_rate'])}) and "
                    f"{t['fn']} false negatives (FNR {fmt(t['false_negative_rate'])}). "
                    "A false negative is a diseased case the model labelled as no disease; a false "
                    "positive is the reverse. With so few test cases, one error moves these rates a lot.")
    st.info("The Dummy baseline predicts the majority class only, so its recall is 0 and its ROC-AUC is 0.5. "
            "Differences between Logistic Regression and Random Forest are small relative to the "
            "cross-validation spread and should not be over-interpreted.")

elif page.startswith("3"):
    st.title("Explainability")
    st.subheader("Global: permutation importance (held-out data)")
    st.image(str(OUT / "feature_importance.png"))
    st.dataframe(imp.head(8).style.format(precision=4), width="stretch")
    st.caption("Importance = mean drop in ROC-AUC when a feature is shuffled. It describes this model "
               "on this dataset and does not establish medical causation.")
    st.subheader("Local: one held-out example")
    X_tr, X_te, y_tr, y_te = get_split(df)
    pipe = load_primary()
    pcol = f"proba::{PRIMARY}"
    default = int((preds[pcol] - 0.5).abs().values.argmin())
    i = st.selectbox("Held-out test example (row_id)", list(range(len(X_te))), index=default,
                     format_func=lambda k: f"row {X_te.index[k]}")
    exp, method, base, p = local_explanation(pipe, X_tr, X_te.iloc[[i]])
    c = st.columns(3)
    c[0].metric("Model-predicted probability", f"{p:.3f}")
    c[1].metric("True label (dataset)", int(y_te.iloc[i]))
    c[2].metric("Model output category", uncertainty_category(p))
    plot_local(exp, p, method, OUT / "local_explanation.png")
    st.image(str(OUT / "local_explanation.png"))
    st.caption(f"Method: {method}. Positive values push the model output toward 'disease present'.")
    st.dataframe(exp.head(8).style.format(precision=3), width="stretch")
    st.markdown("""
**Probability categories** (educational, not clinical risk): below 0.35 lower model likelihood within
this dataset; 0.35-0.65 uncertain model output; above 0.65 higher model likelihood within this dataset.
Probabilities may not be calibrated and are not a patient's medical risk.
""")
    st.warning("Explanations describe how the model behaves, not why a person has or lacks disease. " + DISCLAIMER)

else:
    st.title("Subgroup audit (exploratory)")
    st.caption(f"Logistic Regression on the held-out test set (n={metrics['n_test']}). Reliable rates need "
               "n ≥ 30 and ≥ 10 cases per class; ROC-AUC is shown only under that rule. No causal or fairness claims.")
    show = sub[["attribute", "group", "n", "positives", "recall", "false_negative_rate",
                "false_positive_rate", "roc_auc", "note"]].copy()
    for a in ["sex", "age"]:
        st.subheader("Sex (dataset encoding: 0 = female, 1 = male)" if a == "sex" else "Age group")
        st.dataframe(show[show["attribute"] == a].drop(columns="attribute").style.format(precision=3, na_rep="n/a"),
                     width="stretch", hide_index=True)
    if (~sub["reliable"]).any():
        st.warning(INSUFFICIENT + " Flagged groups above show values for transparency only.")
