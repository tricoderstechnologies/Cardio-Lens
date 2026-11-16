"""Train and evaluate Dummy, Logistic Regression (L2) and Random Forest."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.prepare_data import (CATEGORICAL, FEATURES, NUMERIC, PROCESSED, ROOT, TARGET,
                              clean, load_raw)

SEED = 42
OUT = ROOT / "outputs"
MODEL_DIR = OUT / "models"
MODEL_NAMES = ["Baseline (Dummy)", "Logistic Regression (L2)", "Random Forest"]


def build_pipeline(name):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if name.startswith("Logistic"):
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    if name.startswith("Baseline"):
        clf = DummyClassifier(strategy="most_frequent")
    elif name.startswith("Logistic"):
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     min_samples_leaf=2, random_state=SEED, n_jobs=1)
    return Pipeline([("pre", pre), ("clf", clf)])


def get_split(df):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else None,
    }


def run():
    warnings.filterwarnings("ignore", message="Precision is ill-defined")
    df = pd.read_csv(PROCESSED)
    X_tr, X_te, y_tr, y_te = get_split(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "macro_f1": "f1_macro", "roc_auc": "roc_auc"}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {"seed": SEED, "n_train": len(X_tr), "n_test": len(X_te),
               "train_positive": int(y_tr.sum()), "test_positive": int(y_te.sum()),
               "threshold": 0.5, "models": {}}
    preds = pd.DataFrame({"row_id": X_te.index, "y_true": y_te.values})
    for name in MODEL_NAMES:
        pipe = build_pipeline(name)
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)  # train data only
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results["models"][name] = {
            "test": compute_metrics(y_te, pred, proba),
            "cv": {k: {"mean": float(np.mean(cvres[f"test_{k}"])),
                       "std": float(np.std(cvres[f"test_{k}"]))} for k in scoring},
        }
        preds[f"proba::{name}"] = proba
        preds[f"pred::{name}"] = pred
        joblib.dump(pipe, MODEL_DIR / (name.split(" ")[0].lower() + ".joblib"))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    make_plots(results, preds)
    return results


def make_plots(results, preds):
    colors = ["#9AA5B1", "#1F6FB2", "#2E8B57"]
    keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.26
    for i, n in enumerate(MODEL_NAMES):
        vals = [results["models"][n]["test"][k] or 0 for k in keys]
        ax.bar(np.arange(len(keys)) + (i - 1) * w, vals, w, label=n, color=colors[i])
    ax.set_xticks(range(len(keys)), ["Accuracy", "Precision", "Recall", "F1", "Macro F1", "ROC-AUC"])
    ax.set_ylim(0, 1.05); ax.set_title("Held-out test performance"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, n in zip(axes, MODEL_NAMES):
        t = results["models"][n]["test"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, v, ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(n, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n, c in zip(MODEL_NAMES, colors):
        fpr, tpr, _ = roc_curve(preds["y_true"], preds[f"proba::{n}"])
        auc = results["models"][n]["test"]["roc_auc"]
        ax.plot(fpr, tpr, color=c, label=f"{n} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (held-out test)"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    r = run()
    for n, m in r["models"].items():
        t = m["test"]
        print(f"{n}: acc={t['accuracy']:.3f} rec={t['recall']:.3f} auc={t['roc_auc']:.3f} "
              f"cvAUC={m['cv']['roc_auc']['mean']:.3f}±{m['cv']['roc_auc']['std']:.3f}")
