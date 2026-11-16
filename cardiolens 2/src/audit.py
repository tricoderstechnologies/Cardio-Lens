"""Exploratory subgroup audit of the primary (Logistic Regression) model."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.prepare_data import PROCESSED, ROOT
from src.train_eval import get_split

OUT = ROOT / "outputs"
PRIMARY = "Logistic Regression (L2)"
INSUFFICIENT = "Insufficient sample size for reliable comparison."
MIN_CLASS = 10   # per-class minimum for a rate to be shown as reliable
MIN_N_AUC = 30   # minimum subgroup size for ROC-AUC


def age_group(age):
    return pd.cut(age, bins=[-np.inf, 45, 60, np.inf], right=False,
                  labels=["Below 45", "45-59", "60 and above"])


def subgroup_row(attr, group, y, pred, proba):
    y, pred, proba = map(np.asarray, (y, pred, proba))
    n, pos = len(y), int((y == 1).sum())
    neg = n - pos
    tp = int(((y == 1) & (pred == 1)).sum()); fn = pos - tp
    fp = int(((y == 0) & (pred == 1)).sum())
    auc = None
    if n >= MIN_N_AUC and pos >= MIN_CLASS and neg >= MIN_CLASS:
        auc = float(roc_auc_score(y, proba))
    pos_ok, neg_ok = pos >= MIN_CLASS, neg >= MIN_CLASS
    return {
        "attribute": attr, "group": str(group), "n": n, "positives": pos, "negatives": neg,
        "recall": tp / pos if pos else None, "false_negative_rate": fn / pos if pos else None,
        "false_positive_rate": fp / neg if neg else None, "roc_auc": auc,
        "fn": fn, "fp": fp, "tp": tp, "tn": neg - fp,
        "reliable": bool(n >= MIN_N_AUC and pos_ok and neg_ok),
        "note": "" if (n >= MIN_N_AUC and pos_ok and neg_ok) else INSUFFICIENT,
    }


def subgroup_table(X_test, y_test, pred, proba):
    d = X_test.copy()
    d["y"], d["pred"], d["proba"] = np.asarray(y_test), np.asarray(pred), np.asarray(proba)
    rows = []
    for code, label in [(0.0, "Female (sex=0)"), (1.0, "Male (sex=1)")]:
        s = d[d["sex"] == code]
        if len(s):
            rows.append(subgroup_row("sex", label, s["y"], s["pred"], s["proba"]))
    d["age_group"] = age_group(d["age"])
    for g in ["Below 45", "45-59", "60 and above"]:
        s = d[d["age_group"] == g]
        rows.append(subgroup_row("age", g, s["y"], s["pred"], s["proba"]))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(PROCESSED)
    _, X_te, _, y_te = get_split(df)
    p = pd.read_csv(OUT / "test_predictions.csv")
    t = subgroup_table(X_te, y_te, p[f"pred::{PRIMARY}"].values, p[f"proba::{PRIMARY}"].values)
    t.to_csv(OUT / "subgroup_metrics.csv", index=False)
    print(t.to_string())
    return t


if __name__ == "__main__":
    main()
