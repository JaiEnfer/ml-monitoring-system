from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "meta.json"
REFERENCE_PATH = ARTIFACT_DIR / "reference.csv"


FEATURES = ["age", "income", "years_employed", "credit_score"]
TARGET = "defaulted"

@dataclass
class TrainResult:
    auc: float

def make_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(35, 10, n).clip(18, 70)
    income = rng.normal(55000, 20000, n).clip(10000, 200000)
    years_employed = rng.normal(6, 4, n).clip(0, 40)
    credit_score = rng.normal(680, 60, n).clip(300, 850)


    # Create a probability of default with a simple rule 
    # Lower credit score + lower income + short employment => higher default chance
    logits = (
        -0.006 * (credit_score - 650)
        -0.00002 * (income - 50000)
        -0.08 * (years_employed - 5)
        + 0.01 * (age - 35)
    )
    prob = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "years_employed": years_employed,
            "credit_score": credit_score,
            "defaulted": y,
        }
    )
    return df


def train() -> TrainResult:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = make_synthetic_data()
    # Save a baseline (reference) dataset for drift monitoring
    df[FEATURES].sample(n=1000, random_state=42).to_csv(REFERENCE_PATH, index=False)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    joblib.dump(
        {"model": model, "features": FEATURES, "model_version": "v1"},
        MODEL_PATH,
    )

    return TrainResult(auc=auc)


if __name__ == "__main__":
    result = train()
    print(f"Trained model saved to {MODEL_PATH}. AUC={result.auc:.4f}")