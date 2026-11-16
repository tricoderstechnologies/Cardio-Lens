import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.special import expit

from src.api import app
from src.prepare_data import FEATURES, PROCESSED, ROOT
from src.train_eval import get_split

OUT = ROOT / "outputs"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_results_match_saved_outputs(client):
    r = client.get("/api/results").json()
    m = json.loads((OUT / "metrics.json").read_text())
    assert r["models"] == m["models"] and r["dataset"]["n_raw"] == 303
    assert len(r["examples"]) == m["n_test"] and len(r["schema"]) == 13


def test_live_prediction_matches_saved_test_predictions(client):
    p = pd.read_csv(OUT / "test_predictions.csv")
    df = pd.read_csv(PROCESSED); _, X_te, _, _ = get_split(df)
    for i in (0, 7, 30):
        row = {k: (None if np.isnan(v) else float(v)) for k, v in X_te.iloc[i][FEATURES].items()}
        j = client.post("/api/predict", json=row).json()
        assert abs(j["models"]["logistic_regression"]["probability"] - p["proba::Logistic Regression (L2)"].iloc[i]) < 1e-3
        assert abs(j["models"]["random_forest"]["probability"] - p["proba::Random Forest"].iloc[i]) < 1e-3


def test_shap_contributions_are_additive(client):
    df = pd.read_csv(PROCESSED); _, X_te, _, _ = get_split(df)
    row = {k: (None if np.isnan(v) else float(v)) for k, v in X_te.iloc[3][FEATURES].items()}
    j = client.post("/api/predict", json=row).json()
    e = j["explanation"]
    total = e["base_value_log_odds"] + sum(c["contribution"] for c in e["contributions"])
    assert abs(expit(total) - j["models"]["logistic_regression"]["probability"]) < 1e-4
    assert j["disclaimer"] == "This project is a prototype and is not medical advice."


def test_missing_values_are_imputed(client):
    j = client.post("/api/predict", json={"age": 55}).json()
    assert len(j["imputed_features"]) == 12 and 0 <= j["models"]["random_forest"]["probability"] <= 1


@pytest.mark.parametrize("bad", [{"age": 500}, {"cp": 9}, {"thal": 5}, {"oldpeak": -1}, {"sex": "x"}])
def test_invalid_input_rejected(client, bad):
    assert client.post("/api/predict", json=bad).status_code == 422


def test_site_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CardioLens" in r.text
