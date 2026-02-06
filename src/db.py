from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = Path("artifacts") / "predictions.db"
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp_utc = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # features
    age = Column(Float, nullable=False)
    income = Column(Float, nullable=False)
    years_employed = Column(Float, nullable=False)
    credit_score = Column(Float, nullable=False)

    # outputs
    probability = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)

    # metadata
    model_version = Column(String, nullable=False)


def get_engine():
    return create_engine(DB_URL, future=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
