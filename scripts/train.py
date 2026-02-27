from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from forex_css.models.training import WalkForwardConfig, fit_final_models, walk_forward_train_and_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model with walk-forward validation.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset parquet.")
    parser.add_argument("--model", type=str, default="lightgbm", help="lightgbm|catboost|random_forest")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-timestamps", type=int, default=120)
    parser.add_argument("--min-pred-edge", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-model", type=Path, default=Path("models/model_h24.pkl"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/train_h24"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_parquet(args.dataset)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True)
    dataset = dataset.sort_values(["timestamp", "currency"]).reset_index(drop=True)

    wf_config = WalkForwardConfig(
        n_splits=args.n_splits,
        min_train_timestamps=args.min_train_timestamps,
        min_prediction_edge=args.min_pred_edge,
    )
    eval_result = walk_forward_train_and_score(
        dataset=dataset,
        horizon_hours=args.horizon_hours,
        model_name=args.model,
        wf_config=wf_config,
        random_state=args.random_state,
    )

    final_bundle = fit_final_models(
        dataset=dataset,
        feature_cols=eval_result["feature_columns"],
        target_strong=eval_result["target_strong"],
        target_weak=eval_result["target_weak"],
        model_name=args.model,
        random_state=args.random_state,
    )
    final_bundle["meta"] = {
        "horizon_hours": args.horizon_hours,
        "model_requested": args.model,
        "model_used": final_bundle["model_used"],
        "feature_columns": eval_result["feature_columns"],
        "target_strong": eval_result["target_strong"],
        "target_weak": eval_result["target_weak"],
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_model, "wb") as f:
        pickle.dump(final_bundle, f)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    eval_result["predictions"].to_csv(args.report_dir / "walk_forward_predictions.csv", index=False)
    eval_result["trades"].to_csv(args.report_dir / "walk_forward_trades_model.csv", index=False)
    eval_result["baseline_trades"].to_csv(args.report_dir / "walk_forward_trades_baseline.csv", index=False)

    metrics_payload = {
        "model": eval_result["metrics_model"],
        "baseline": eval_result["metrics_baseline"],
        "model_used": final_bundle["model_used"],
        "horizon_hours": args.horizon_hours,
    }
    with open(args.report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved model: {args.output_model}")
    print(f"Saved reports: {args.report_dir}")
    print(f"Model metrics: {metrics_payload['model']}")
    print(f"Baseline metrics: {metrics_payload['baseline']}")


if __name__ == "__main__":
    main()
