from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: float = Field(..., ge=0)
    income: float = Field(..., ge=0)
    years_employed: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=0, le=1000)

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

    