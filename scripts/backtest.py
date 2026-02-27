from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from forex_css.backtest.engine import run_backtest, save_backtest_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strategy backtest on prepared dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset parquet.")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "model"])
    parser.add_argument("--model-path", type=Path, default=None, help="Required for --mode=model.")
    parser.add_argument("--min-pred-edge", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/backtest"))
    parser.add_argument("--prefix", type=str, default="backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_parquet(args.dataset)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset = dataset.sort_values(["timestamp", "currency"]).reset_index(drop=True)

    model_bundle = None
    if args.mode == "model":
        if args.model_path is None:
            raise ValueError("--model-path is required when --mode=model")
        with open(args.model_path, "rb") as f:
            model_bundle = pickle.load(f)

    result = run_backtest(
        dataset=dataset,
        target_horizon_hours=args.horizon_hours,
        mode=args.mode,
        model_bundle=model_bundle,
        min_prediction_edge=args.min_pred_edge,
    )

    saved = save_backtest_report(
        result=result,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    print(f"Saved backtest artifacts: {saved}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
