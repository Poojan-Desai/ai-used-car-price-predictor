# AI Used Car Price Predictor

I trained a small regression pipeline with scikit‑learn to estimate used car prices from year, miles, make/model/trim, and a simple condition score. The sample data here is tiny—just for structure—so I plan to replace it with a larger dataset.

## How I run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
python predict.py --year 2018 --miles 60000 --make Toyota --model Corolla --trim LE --condition .62
```

## What I learned
End‑to‑end ML basics: preprocessing (OHE), a quick RandomForest baseline, and simple metrics (R²/MAE).

## Notes
- I keep things simple and readable.
- If something feels off, I open an issue and fix it quickly.
