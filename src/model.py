from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("artifacts") / "model.joblib"


class ModelService:
    def __init__(self) -> None:
        self._bundle = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python -m src.train"
            )
        self._bundle = joblib.load(MODEL_PATH)

    @property
    def model_version(self) -> str:
        return self._bundle["model_version"]

    @property
    def features(self) -> list[str]:
        return self._bundle["features"]

    def predict_proba(self, row: dict) -> float:
        model = self._bundle["model"]
        X = pd.DataFrame([row], columns=self.features)
        proba = float(model.predict_proba(X)[0, 1])
        return proba
