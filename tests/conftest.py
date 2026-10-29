import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge

from app.main import app
from app.model import ModelService
from ml.datasets.sleep.train import CATEGORICAL, NUMERIC, build_pipeline


@pytest.fixture(scope="session")
def fitted_service():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "Age": rng.integers(27, 60, n), "Sleep Duration": rng.uniform(5.8, 8.5, n),
        "Quality of Sleep": rng.integers(4, 10, n), "Physical Activity Level": rng.integers(30, 90, n),
        "Heart Rate": rng.integers(65, 85, n), "Daily Steps": rng.integers(3000, 10000, n),
        "Systolic": rng.integers(115, 140, n), "Diastolic": rng.integers(75, 95, n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Occupation": rng.choice(["Doctor", "Nurse", "Engineer"], n),
        "BMI Category": rng.choice(["Normal", "Overweight", "Obese"], n),
        "Sleep Disorder": rng.choice(["None", "Insomnia", "Sleep Apnea"], n),
    })[NUMERIC + CATEGORICAL]
    y = rng.integers(3, 9, n)
    return ModelService(build_pipeline(Ridge()).fit(X, y))


@pytest.fixture
def client(fitted_service):
    app.state.sleep_model = fitted_service
    yield TestClient(app)
    app.state.sleep_model = None


@pytest.fixture
def no_model_client():
    app.state.sleep_model = None
    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "gender": "Female", "age": 29, "occupation": "Nurse", "sleep_duration": 6.5,
        "quality_of_sleep": 6, "physical_activity_level": 45, "bmi_category": "Normal",
        "systolic": 120, "diastolic": 80, "heart_rate": 72, "daily_steps": 6000,
        "sleep_disorder": "None",
    }