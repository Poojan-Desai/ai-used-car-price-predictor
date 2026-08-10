"""Train the used-car price baseline and save its evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["year", "miles", "make", "model", "trim", "condition"]
TARGET = "price"
NUMERIC_FEATURES = ["year", "miles", "condition"]
CATEGORICAL_FEATURES = ["make", "model", "trim"]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the training CSV and fail clearly when its schema is incomplete."""
    data = pd.read_csv(path)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns")
    return data


def build_pipeline() -> Pipeline:
    """Create a deterministic preprocessing and regression pipeline."""
    preprocessing = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def train(data_path: Path, model_path: Path, metrics_path: Path) -> dict[str, float | int]:
    """Fit the baseline, evaluate one fixed holdout, and save reproducible outputs."""
    data = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.25,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics: dict[str, float | int] = {
        "rows": int(len(data)),
        "holdout_rows": int(len(x_test)),
        "mean_absolute_error": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cars.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(args.data, args.model, args.metrics)
    print(json.dumps(result, indent=2))
