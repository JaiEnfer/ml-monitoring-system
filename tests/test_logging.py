from fastapi.testclient import TestClient
from sqlalchemy import text

from src.app import app
from src.db import get_engine, init_db


def test_prediction_is_logged():
    init_db()
    engine = get_engine()

    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar_one()

    with TestClient(app) as client:
        payload = {
            "age": 30,
            "income": 40000,
            "years_employed": 2,
            "credit_score": 600,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    with engine.connect() as conn:
        after = conn.execute(text("SELECT COUNT(*) FROM prediction_logs")).scalar_one()

    assert after == before + 1
