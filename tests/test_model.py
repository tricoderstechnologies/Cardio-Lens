import json
import numpy as np
import pandas as pd
from src.prepare_data import FEATURES, PROCESSED, ROOT, TARGET
from src.train_eval import MODEL_NAMES, build_pipeline, compute_metrics, get_split

OUT = ROOT / "outputs"


def test_split_is_stratified_and_disjoint():
    df = pd.read_csv(PROCESSED)
    X_tr, X_te, y_tr, y_te = get_split(df)
    assert len(X_tr) + len(X_te) == len(df)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert len(X_te) == round(0.2 * len(df)) or len(X_te) == int(np.ceil(0.2 * len(df)))


def test_unknown_category_and_missing_handled():
    df = pd.read_csv(PROCESSED)
    X_tr, X_te, y_tr, _ = get_split(df)
    pipe = build_pipeline("Logistic Regression (L2)").fit(X_tr, y_tr)
    row = X_te.iloc[[0]].copy()
    row["cp"] = 99.0; row["chol"] = np.nan
    p = pipe.predict_proba(row)[0, 1]
    assert 0 <= p <= 1


def test_reported_metrics_match_saved_predictions():
    m = json.loads((OUT / "metrics.json").read_text())
    p = pd.read_csv(OUT / "test_predictions.csv")
    for n in MODEL_NAMES:
        again = compute_metrics(p["y_true"], p[f"pred::{n}"], p[f"proba::{n}"])
        for k in ["accuracy", "recall", "f1", "macro_f1", "roc_auc", "tn", "fp", "fn", "tp"]:
            assert np.isclose(m["models"][n]["test"][k], again[k])


def test_logistic_beats_baseline_on_cv_auc():
    m = json.loads((OUT / "metrics.json").read_text())["models"]
    assert m["Logistic Regression (L2)"]["cv"]["roc_auc"]["mean"] > m["Baseline (Dummy)"]["cv"]["roc_auc"]["mean"]
