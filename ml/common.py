"""Shared helpers so every dataset trains and reports the same way."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


def compare_models(models: dict, X, y, cv: int = 5, seed: int = 42) -> dict:
    """Cross-validate each pipeline and return RMSE mean/std per model."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        results[name] = {
            "cv_rmse_mean": round(float(-scores.mean()), 4),
            "cv_rmse_std": round(float(scores.std()), 4),
        }
        print(f"{name:<18} CV RMSE = {-scores.mean():.3f} (+/- {scores.std():.3f})")
    return results


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
    }


def save_artifacts(model, report: dict, model_path: Path, report_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved model   -> {model_path}")
    print(f"Saved metrics -> {report_path}")
