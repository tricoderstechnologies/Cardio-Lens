"""Global (permutation importance) and local (SHAP / coefficient) explanations."""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src.prepare_data import FEATURES, PROCESSED, ROOT
from src.train_eval import MODEL_DIR, SEED, get_split

OUT = ROOT / "outputs"


def uncertainty_category(p):
    if p < 0.35:
        return "Lower model likelihood within this dataset"
    if p <= 0.65:
        return "Uncertain model output"
    return "Higher model likelihood within this dataset"


def load_primary():
    return joblib.load(MODEL_DIR / "logistic.joblib")


def global_importance(pipe, X_te, y_te, n_repeats=50):
    r = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=SEED, n_jobs=1)
    return (pd.DataFrame({"feature": FEATURES, "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def _orig(name):  # 'cat__cp_4.0' -> 'cp'; 'num__age' -> 'age'
    return name.split("__", 1)[1].rsplit("_", 1)[0] if name.startswith("cat__") else name.split("__", 1)[1]


def local_explanation(pipe, X_train, x_row):
    """Per-feature contribution to the log-odds for one row.
    Uses SHAP LinearExplainer (background = training data); falls back to
    coefficient x standardised value if SHAP fails. Returns (df, method, base, proba)."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    xt = pre.transform(x_row)
    bg = pre.transform(X_train)
    xt = xt.toarray() if hasattr(xt, "toarray") else np.asarray(xt)
    bg = bg.toarray() if hasattr(bg, "toarray") else np.asarray(bg)
    try:
        import shap
        ex = shap.LinearExplainer(clf, shap.maskers.Independent(bg, max_samples=len(bg)))
        vals = np.asarray(ex.shap_values(xt))[0]
        base = float(np.ravel(ex.expected_value)[0])
        method = "SHAP LinearExplainer (log-odds)"
    except Exception:
        vals = clf.coef_[0] * (xt[0] - bg.mean(axis=0))
        base = float(clf.intercept_[0] + clf.coef_[0] @ bg.mean(axis=0))
        method = "Logistic Regression coefficient x (value - training mean) (log-odds)"
    d = pd.DataFrame({"feature": [_orig(n) for n in names], "encoded": names, "contribution": vals})
    d["value"] = d["feature"].map(lambda f: x_row.iloc[0][f])
    agg = d.groupby("feature", as_index=False).agg(contribution=("contribution", "sum"),
                                                   value=("value", "first"))
    agg = agg.reindex(agg["contribution"].abs().sort_values(ascending=False).index)
    proba = float(pipe.predict_proba(x_row)[0, 1])
    return agg.reset_index(drop=True), method, base, proba


def pick_example(proba_test):
    """Documented rule: the test example whose probability is closest to 0.5."""
    return int(np.argmin(np.abs(np.asarray(proba_test) - 0.5)))


def plot_global(imp, path):
    top = imp.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#1F6FB2")
    ax.set_xlabel("Drop in ROC-AUC when feature is permuted (held-out test set)")
    ax.set_title("Top 8 features: permutation importance (Logistic Regression)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_local(exp, proba, method, path):
    top = exp.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top["feature"], top["contribution"],
            color=["#C0392B" if v > 0 else "#1F6FB2" for v in top["contribution"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Contribution to log-odds (red raises, blue lowers model output)")
    ax.set_title(f"Local explanation (model probability {proba:.2f})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    df = pd.read_csv(PROCESSED)
    X_tr, X_te, y_tr, y_te = get_split(df)
    pipe = load_primary()
    imp = global_importance(pipe, X_te, y_te)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)
    plot_global(imp, OUT / "feature_importance.png")
    proba = pipe.predict_proba(X_te)[:, 1]
    i = pick_example(proba)
    exp, method, base, p = local_explanation(pipe, X_tr, X_te.iloc[[i]])
    exp.to_csv(OUT / "local_explanation_example.csv", index=False)
    plot_local(exp, p, method, OUT / "local_explanation.png")
    print(imp.head(8).to_string()); print(f"example row_id={X_te.index[i]} p={p:.3f} method={method}")
    print(exp.head(8).to_string())


if __name__ == "__main__":
    main()
