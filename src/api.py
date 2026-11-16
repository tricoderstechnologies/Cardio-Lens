"""CardioLens API: serves saved results and live inference from the real trained pipelines.

Run:  uvicorn src.api:app --port 8000      (also serves the website at /)
Inputs are processed in memory only; they are not stored or logged by this code.
Research prototype - not medical advice; not for diagnosis, screening, treatment or healthcare decisions.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.explain import local_explanation, uncertainty_category
from src.prepare_data import FEATURES, PROCESSED, ROOT
from src.schema import BY_NAME, SPECS
from src.site_data import build_payload
from src.train_eval import MODEL_DIR, get_split

DISCLAIMER = "This project is a prototype and is not medical advice."
NOTE = ("Output of a research model trained on 303 historical Cleveland records. It is not a clinical risk, "
        "is not clinically validated, probabilities may be uncalibrated, and it must not be used for "
        "diagnosis, screening, treatment or healthcare decisions.")


class Record(BaseModel):
    """One record in the Cleveland feature format. Omitted/null values are imputed by the pipeline."""
    age: float | None = None
    sex: float | None = None
    cp: float | None = None
    trestbps: float | None = None
    chol: float | None = None
    fbs: float | None = None
    restecg: float | None = None
    thalach: float | None = None
    exang: float | None = None
    oldpeak: float | None = None
    slope: float | None = None
    ca: float | None = None
    thal: float | None = None

    @field_validator(*FEATURES)
    @classmethod
    def in_range(cls, v, info):
        if v is None:
            return v
        s = BY_NAME[info.field_name]
        if s["kind"] == "cat":
            if v not in [o[0] for o in s["options"]]:
                raise ValueError(f"must be one of {[o[0] for o in s['options']]}")
        elif not (s["min"] <= v <= s["max"]):
            raise ValueError(f"must be between {s['min']} and {s['max']}")
        return v


@asynccontextmanager
async def lifespan(app):
    df = pd.read_csv(PROCESSED)
    app.state.X_train = get_split(df)[0]
    app.state.lr = joblib.load(MODEL_DIR / "logistic.joblib")
    app.state.rf = joblib.load(MODEL_DIR / "random.joblib")
    app.state.payload = build_payload()
    yield


app = FastAPI(title="CardioLens AI API", version="1.0", lifespan=lifespan,
              description=DISCLAIMER + " " + NOTE)
origins = [o for o in os.environ.get("CARDIOLENS_CORS", "").split(",") if o]
if origins:  # only needed when the static site is hosted on a different origin
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": ["logistic", "random_forest"], "disclaimer": DISCLAIMER}


@app.get("/api/schema")
def schema():
    return SPECS


@app.get("/api/results")
def results():
    """Saved outputs of the executed Python project (metrics, ROC, importance, subgroups, test examples)."""
    return app.state.payload


@app.post("/api/predict")
def predict(rec: Record):
    vals = rec.model_dump()
    row = pd.DataFrame([vals])[FEATURES].astype(float)
    out = {}
    for key, pipe in (("logistic_regression", app.state.lr), ("random_forest", app.state.rf)):
        p = float(pipe.predict_proba(row)[0, 1])
        out[key] = {"probability": round(p, 4), "category": uncertainty_category(p),
                    "class_at_threshold_0_5": int(p >= 0.5)}
    exp, method, base, p = local_explanation(app.state.lr, app.state.X_train, row)
    return {
        "models": out,
        "explanation": {"model": "logistic_regression", "method": method, "base_value_log_odds": base,
                        "contributions": [{"feature": r.feature,
                                           "value": None if np.isnan(r.value) else float(r.value),
                                           "contribution": float(r.contribution)} for r in exp.itertuples()]},
        "imputed_features": [k for k, v in vals.items() if v is None],
        "disclaimer": DISCLAIMER, "note": NOTE,
    }


site = ROOT / "website"
if site.exists():  # mounted last so /api/* routes win
    app.mount("/", StaticFiles(directory=site, html=True), name="site")
