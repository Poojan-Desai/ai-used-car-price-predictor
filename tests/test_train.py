from pathlib import Path

import joblib
import pandas as pd
import pytest

from train import FEATURES, load_dataset, train


def test_load_dataset_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "cars.csv"
    pd.DataFrame({"year": [2020], "price": [20_000]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_training_saves_model_and_metrics(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = train(Path("data/cars.csv"), model_path, metrics_path)

    assert model_path.is_file()
    assert metrics_path.is_file()
    assert metrics["rows"] == 15
    assert metrics["holdout_rows"] == 4
    assert metrics["mean_absolute_error"] >= 0

    pipeline = joblib.load(model_path)
    unseen = pd.DataFrame(
        [[2022, 10_000, "Mazda", "3", "Premium", 5]], columns=FEATURES
    )
    prediction = float(pipeline.predict(unseen)[0])
    assert prediction > 0
