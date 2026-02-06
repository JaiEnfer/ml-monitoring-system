from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.db import get_engine

ARTIFACT_DIR = Path("artifacts")
REFERENCE_PATH = ARTIFACT_DIR / "reference.csv"
REPORTS_DIR = Path("reports")
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"

FEATURES = ["age", "income", "years_employed", "credit_score"]


def load_current_data(limit: int = 500) -> pd.DataFrame:
    """
    Load the most recent prediction inputs from SQLite.
    We use only feature columns for drift detection.
    """
    engine = get_engine()
    query = f"""
        SELECT {", ".join(FEATURES)}
        FROM prediction_logs
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, engine)
    # Evidently expects the same column names as reference
    return df[FEATURES]


def generate_drift_report(limit: int = 500) -> Path:
    """
    Compare reference data vs recent production data and generate an HTML report.
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE_PATH}. Run: python -m src.train"
        )

    reference = pd.read_csv(REFERENCE_PATH)
    current = load_current_data(limit=limit)

    if current.empty:
        raise ValueError("No production data found yet. Call /predict a few times first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report.save_html(str(DRIFT_REPORT_PATH))
    return DRIFT_REPORT_PATH
