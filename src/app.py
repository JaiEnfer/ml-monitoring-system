from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.db import PredictionLog, SessionLocal, init_db
from src.drift import generate_drift_report
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
