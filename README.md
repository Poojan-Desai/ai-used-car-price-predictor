# Used Car Price Predictor

A compact machine-learning baseline that estimates a used vehicle's price from
its age, mileage, make, model, trim, and condition. The project demonstrates a
reproducible scikit-learn workflow: schema validation, mixed numeric/categorical
preprocessing, a deterministic random-forest model, holdout evaluation, saved
artifacts, and command-line inference. Evaluation now includes a median-price
baseline and three-fold cross-validation so the model is not presented without
a reference point.

> This is an early learning project. The included CSV contains only 15 sample
> rows, so its metrics are a pipeline check—not evidence of real-world pricing
> accuracy. A production study would require a larger, representative dataset,
> stronger baselines, uncertainty estimates, and drift monitoring.

## Workflow

```text
CSV data -> schema checks -> train/holdout split
         -> one-hot encoding + Random Forest
         -> median baseline + 3-fold cross-validation
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

The tests cover dataset-schema validation, baseline/cross-validation evidence,
deterministic training artifacts, and prediction behavior for an unseen
category.

## Stack

Python, pandas, scikit-learn, NumPy, joblib, and pytest.

## Next steps

- Replace the sample CSV with a documented, sufficiently large dataset.
- Compare against a transparent regularized linear model.
- Add feature-quality checks and report uncertainty instead of one point value.
