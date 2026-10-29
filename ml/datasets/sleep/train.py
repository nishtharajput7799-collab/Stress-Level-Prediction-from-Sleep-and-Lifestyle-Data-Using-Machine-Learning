"""Train a stress-level model on the Kaggle 'Sleep Health and Lifestyle' dataset.

Run from the project root:
    python -m ml.datasets.sleep.train
    python -m ml.datasets.sleep.train --data path/to/file.csv
"""
import argparse
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.common import compare_models, evaluate, save_artifacts

HERE = Path(__file__).parent
TARGET = "Stress Level"
NUMERIC = [
    "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Heart Rate", "Daily Steps", "Systolic", "Diastolic",
]
CATEGORICAL = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # "None" in Sleep Disorder is a real category, but pandas reads it as NaN
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
    # Blood Pressure comes as "126/83" -> two numeric columns
    bp = df["Blood Pressure"].str.split("/", expand=True).astype(float)
    df["Systolic"], df["Diastolic"] = bp[0], bp[1]
    return df.drop(columns=["Person ID", "Blood Pressure"], errors="ignore")


def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


def main(data_path: Path) -> None:
    df = load_data(data_path)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "ridge": build_pipeline(Ridge(alpha=1.0)),
        "random_forest": build_pipeline(RandomForestRegressor(n_estimators=300, random_state=42)),
        "gradient_boosting": build_pipeline(GradientBoostingRegressor(random_state=42)),
    }
    cv_results = compare_models(candidates, X_train, y_train)

    best_name = min(cv_results, key=lambda k: cv_results[k]["cv_rmse_mean"])
    best = candidates[best_name].fit(X_train, y_train)
    test_metrics = evaluate(best, X_test, y_test)
    print(f"\nBest model: {best_name} | test metrics: {test_metrics}")

    report = {
        "dataset": "Sleep Health and Lifestyle (Kaggle)",
        "target": TARGET,
        "n_rows": int(len(df)),
        "sklearn_version": sklearn.__version__,
        "cv_results": cv_results,
        "best_model": best_name,
        "test_metrics": test_metrics,
    }
    save_artifacts(best, report, Path("models/sleep_model.joblib"), Path("models/sleep_report.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    path = args.data or next((HERE / "data").glob("*.csv"), None)
    if path is None:
        raise SystemExit("No CSV found. Run scripts/download_data.sh or pass --data.")
    main(path)
