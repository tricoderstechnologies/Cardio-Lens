"""Build the results payload used by the website (served by the API and exported to data.js)."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.explain import load_primary, local_explanation
from src.prepare_data import FEATURES, PROCESSED, ROOT, load_raw
from src.schema import SPECS
from src.train_eval import MODEL_NAMES, get_split

OUT = ROOT / "outputs"


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def build_payload():
    m = json.loads((OUT / "metrics.json").read_text())
    raw, df = load_raw(), pd.read_csv(PROCESSED)
    X_tr, X_te, y_tr, y_te = get_split(df)
    preds = pd.read_csv(OUT / "test_predictions.csv")
    pipe = load_primary()
    roc = {}
    for n in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(preds["y_true"], preds[f"proba::{n}"])
        roc[n] = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]
    imp = pd.read_csv(OUT / "permutation_importance.csv")
    sub = pd.read_csv(OUT / "subgroup_metrics.csv").astype(object).where(lambda d: d.notna(), None)
    examples, method = [], ""
    for i in range(len(X_te)):
        exp, method, base, p = local_explanation(pipe, X_tr, X_te.iloc[[i]])
        examples.append({"row_id": int(X_te.index[i]), "y_true": int(y_te.iloc[i]), "proba": round(p, 4),
                         "features": {f: _clean(X_te.iloc[i][f]) for f in FEATURES},
                         "contrib": [{"feature": r.feature, "value": _clean(r.value), "c": round(float(r.contribution), 4)}
                                     for r in exp.head(8).itertuples()]})
    return {
        "dataset": {"n_raw": len(raw), "n_clean": len(df), "n0": int((df.target == 0).sum()),
                    "n1": int((df.target == 1).sum()), "n_features": len(FEATURES),
                    "missing": {k: int(v) for k, v in raw.isna().sum().items() if v > 0},
                    "rows_missing": int(raw.isna().any(axis=1).sum()),
                    "num_dist": {str(k): int(v) for k, v in raw["num"].value_counts().sort_index().items()}},
        "split": {k: m[k] for k in ["seed", "n_train", "n_test", "train_positive", "test_positive", "threshold"]},
        "models": m["models"], "roc": roc, "importance": imp.to_dict("records"),
        "subgroups": json.loads(sub.to_json(orient="records")), "examples": examples,
        "local_method": method, "schema": SPECS,
    }
