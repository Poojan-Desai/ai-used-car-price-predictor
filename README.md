# Used Car Price Predictor

A compact machine-learning baseline that estimates a used vehicle's price from
its age, mileage, make, model, trim, and condition. The project demonstrates a
reproducible scikit-learn workflow: schema validation, mixed numeric/categorical
preprocessing, a deterministic random-forest model, holdout evaluation, saved
artifacts, and command-line inference.

> This is an early learning project. The included CSV contains only 15 sample
> rows, so its metrics are a pipeline check—not evidence of real-world pricing
> accuracy. A production study would require a larger, representative dataset,
> cross-validation, stronger baselines, and drift monitoring.

## Workflow

```text
CSV data -> schema checks -> train/holdout split
         -> one-hot encoding + Random Forest
         -> model artifact + evaluation metrics -> CLI estimate
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py
python predict.py \
  --year 2018 \
  --miles 60000 \
  --make Toyota \
  --model Corolla \
  --trim LE \
  --condition 4
```

Training writes `artifacts/model.joblib` and `artifacts/metrics.json`. These
generated files are intentionally excluded from version control.

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover dataset-schema validation, deterministic training artifacts,
and prediction behavior for an unseen category.

## Stack

Python, pandas, scikit-learn, NumPy, joblib, and pytest.

## Next steps

- Replace the sample CSV with a documented, sufficiently large dataset.
- Compare against a transparent linear baseline and use cross-validation.
- Add feature-quality checks and report uncertainty instead of one point value.
