from fastapi.testclient import TestClient

from src.app import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_version" in data


def test_predict():
    with TestClient(app) as client:
        payload = {
            "age": 28,
            "income": 32000,
            "years_employed": 1,
            "credit_score": 580,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
