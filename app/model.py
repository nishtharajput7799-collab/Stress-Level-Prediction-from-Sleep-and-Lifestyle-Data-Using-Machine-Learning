from pathlib import Path

import joblib
import pandas as pd

from app.schema import SleepStressInput
from ml.datasets.sleep.train import CATEGORICAL, NUMERIC

DISCLAIMER = (
    "Educational estimate from a statistical model. "
    "Not a medical or psychological diagnosis."
)


def stress_category(score: float) -> str:
    if score <= 4:
        return "Low"
    if score <= 6:
        return "Moderate"
    return "High"


class ModelService:
    """Wraps the trained sklearn pipeline so the API never touches raw column names."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: Path) -> "ModelService":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return cls(joblib.load(path))

    def predict_stress(self, data: SleepStressInput) -> float:
        row = {
            "Age": data.age,
            "Sleep Duration": data.sleep_duration,
            "Quality of Sleep": data.quality_of_sleep,
            "Physical Activity Level": data.physical_activity_level,
            "Heart Rate": data.heart_rate,
            "Daily Steps": data.daily_steps,
            "Systolic": data.systolic,
            "Diastolic": data.diastolic,
            "Gender": data.gender,
            "Occupation": data.occupation,
            "BMI Category": data.bmi_category,
            "Sleep Disorder": data.sleep_disorder,
        }
        frame = pd.DataFrame([row])[NUMERIC + CATEGORICAL]
        score = float(self.pipeline.predict(frame)[0])
        return round(min(max(score, 1.0), 10.0), 2)