from fastapi.testclient import TestClient

from src.app import app


def test_retrain_updates_model_version():
    with TestClient(app) as client:
        before = client.get("/health").json()["model_version"]
        r = client.post("/retrain")
        assert r.status_code == 200
        after = client.get("/health").json()["model_version"]
        assert after != before
