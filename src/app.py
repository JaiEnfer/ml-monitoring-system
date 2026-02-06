from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.db import PredictionLog, SessionLocal, init_db
from src.drift import generate_drift_report, get_drift_status
from src.model import ModelService
from src.schemas import PredictRequest, PredictResponse

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    model_service.load()
    yield

app = FastAPI(title="ML Monitoring System", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_service.model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = req.model_dump()
    proba = model_service.predict_proba(row)
    pred = 1 if proba >= 0.5 else 0

    # Log to SQLite
    db = SessionLocal()
    try:
        log = PredictionLog(
            age=row["age"],
            income=row["income"],
            years_employed=row["years_employed"],
            credit_score=row["credit_score"],
            probability=proba,
            prediction=pred,
            model_version=model_service.model_version,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return PredictResponse(
        prediction=pred,
        probability=proba,
        model_version=model_service.model_version,
    )

@app.get("/drift/report")
def drift_report():
    path = generate_drift_report(limit=500)
    return FileResponse(path, media_type="text/html", filename=path.name)

@app.get("/drift/status")
def drift_status():
    try:
        status = get_drift_status(limit=500)
        return {
            "drift_detected": status.drift_detected,
            "share_drifted_features": status.share_drifted_features,
            "drifted_features": status.drifted_features,
            "n_reference": status.n_reference,
            "n_current": status.n_current,
        }
    except Exception as e:
        # Return readable error instead of generic 500
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drift/alert")
def drift_alert(threshold: float = 0.5):
    """
    threshold: fraction of features drifted to raise an alert
    """
    status = get_drift_status(limit=500)
    level = "OK"
    if status.share_drifted_features >= threshold:
        level = "ALERT"

    return {
        "level": level,
        "threshold": threshold,
        "share_drifted_features": status.share_drifted_features,
        "drifted_features": status.drifted_features,
        "n_current": status.n_current,
        "n_reference": status.n_reference,
    }
