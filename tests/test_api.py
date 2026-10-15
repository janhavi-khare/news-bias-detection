import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import init_models


@pytest.fixture(scope="module")
def client():
    # Ensure models are preloaded before running client tests
    init_models()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "health" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "0" in data["id2label"]
    assert data["id2label"]["0"] == "Non-biased"
    assert data["id2label"]["1"] == "Biased"


def test_predict_endpoint_valid_neutral(client):
    payload = {
        "text": "The committee met to review the quarterly budget report.",
        "include_explanation": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "Non-biased"
    assert "probabilities" in data
    assert data["probabilities"]["Non-biased"] >= 0.5
    assert data["explanation"] is None


def test_predict_endpoint_valid_biased(client):
    payload = {
        "text": "The corrupt administration recklessly betrayed citizens with its disastrous decision.",
        "include_explanation": False
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] == "Biased"
    assert data["probabilities"]["Biased"] >= 0.5


def test_predict_endpoint_with_explanation(client):
    payload = {
        "text": "The corrupt politician lied to voters.",
        "include_explanation": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["explanation"] is not None
    assert data["explanation"]["success"] is True
    assert "disclaimer" in data["explanation"]
    assert len(data["explanation"]["token_highlights"]) > 0


def test_predict_endpoint_empty_input(client):
    # Empty string should fail validation
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [400, 422]


def test_predict_endpoint_whitespace_input(client):
    # Whitespace-only string
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code in [400, 422]


def test_explain_endpoint(client):
    payload = {
        "text": "The corrupt government shamelessly betrayed citizens.",
        "top_n": 5
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "disclaimer" in data
    assert len(data["top_words"]) <= 5


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["test_set_size"] == 527
    assert data["accuracy"] == 0.8197
    assert data["macro_f1"] == 0.8184
