"""Estimate a used-car price with a previously trained baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--miles", type=int, required=True)
    parser.add_argument("--make", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trim", required=True)
    parser.add_argument("--condition", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_file.is_file():
        raise SystemExit(f"Model not found at {args.model_file}. Run train.py first.")
    if args.year < 1980 or args.miles < 0 or not 1 <= args.condition <= 5:
        raise SystemExit(
            "Year must be at least 1980, miles cannot be negative, and condition must be 1–5."
        )

    pipeline = joblib.load(args.model_file)
    sample = pd.DataFrame(
        [
            {
                "year": args.year,
                "miles": args.miles,
                "make": args.make,
                "model": args.model,
                "trim": args.trim,
                "condition": args.condition,
            }
        ]
    )
    estimate = float(pipeline.predict(sample)[0])
    print(f"Estimated price: ${estimate:,.2f}")


if __name__ == "__main__":
    main()
