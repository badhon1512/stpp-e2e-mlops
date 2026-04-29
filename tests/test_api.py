from fastapi.testclient import TestClient

from backend.app import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint_returns_loaded_model_details():
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()
    assert "model_uri" in data
    assert data["model_source"] in {"mlflow", "local", "stub"}


def test_predict_endpoint_returns_prediction_shape():
    payload = {
        "subject": "Billing outage in production",
        "body": "Customers cannot complete payments after the latest deploy.",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_priority" in data
    assert "confidence" in data
    assert "model_uri" in data
