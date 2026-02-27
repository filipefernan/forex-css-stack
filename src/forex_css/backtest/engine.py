from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from forex_css.models.training import baseline_congruence_trades, infer_periods_per_year, performance_metrics


def equity_curve_from_returns(returns: pd.Series) -> pd.Series:
    clean = returns.fillna(0.0).astype(float)
    return (1.0 + clean).cumprod()


def select_model_trades(
    dataset: pd.DataFrame,
    model_bundle: dict[str, Any],
    min_prediction_edge: float = 0.0,
) -> pd.DataFrame:
    frame = dataset.copy().sort_values(["timestamp", "currency"]).reset_index(drop=True)
    feature_cols = model_bundle["feature_columns"]
    strong_col = model_bundle["target_strong"]
    weak_col = model_bundle["target_weak"]
    frame = frame.dropna(subset=feature_cols).copy()

    scored = frame[["timestamp", "currency", strong_col, weak_col]].copy()
    scored["pred_strong"] = model_bundle["model_strong"].predict(frame[feature_cols])
    scored["pred_weak"] = model_bundle["model_weak"].predict(frame[feature_cols])

    trades: list[dict[str, Any]] = []
    for ts, group in scored.groupby("timestamp", sort=True):
        idx_strong = group["pred_strong"].idxmax()
        idx_weak = group["pred_weak"].idxmax()
        best_strong = float(group.loc[idx_strong, "pred_strong"])
        best_weak = float(group.loc[idx_weak, "pred_weak"])

        if max(best_strong, best_weak) <= min_prediction_edge:
            continue

        if best_strong >= best_weak:
            chosen = group.loc[idx_strong]
            direction = "strong"
            realized = float(chosen[strong_col])
            prediction = best_strong
        else:
            chosen = group.loc[idx_weak]
            direction = "weak"
            realized = float(chosen[weak_col])
            prediction = best_weak

        trades.append(
            {
                "timestamp": ts,
                "currency": chosen["currency"],
                "direction": direction,
                "prediction": prediction,
                "realized_return": realized,
            }
        )

    return pd.DataFrame(trades)


def run_backtest(
    dataset: pd.DataFrame,
    target_horizon_hours: int,
    mode: str = "baseline",
    model_bundle: dict[str, Any] | None = None,
    min_prediction_edge: float = 0.0,
) -> dict[str, Any]:
    target_strong = f"target_ret_h{target_horizon_hours}_strong"
    target_weak = f"target_ret_h{target_horizon_hours}_weak"
    for col in (target_strong, target_weak):
        if col not in dataset.columns:
            raise ValueError(f"Missing target column: {col}")

    mode_key = mode.strip().lower()
    if mode_key == "baseline":
        trades = baseline_congruence_trades(
            dataset=dataset,
            target_strong_col=target_strong,
            target_weak_col=target_weak,
        )
    elif mode_key == "model":
        if model_bundle is None:
            raise ValueError("model_bundle is required for mode=model")
        trades = select_model_trades(
            dataset=dataset,
            model_bundle=model_bundle,
            min_prediction_edge=min_prediction_edge,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use baseline or model.")

    returns = trades["realized_return"] if not trades.empty else pd.Series(dtype=float)
    periods = infer_periods_per_year(dataset)
    metrics = performance_metrics(returns, periods_per_year=periods)
    curve = equity_curve_from_returns(returns)

    return {
        "trades": trades,
        "returns": returns,
        "equity_curve": curve,
        "metrics": metrics,
        "periods_per_year": periods,
    }


def save_backtest_report(
    result: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "backtest",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades_path = out / f"{prefix}_trades.csv"
    metrics_path = out / f"{prefix}_metrics.json"
    equity_path = out / f"{prefix}_equity.csv"
    figure_path = out / f"{prefix}_equity.png"

    result["trades"].to_csv(trades_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    result["equity_curve"].to_frame(name="equity").to_csv(equity_path, index=False)

    plt.figure(figsize=(10, 4))
    if not result["equity_curve"].empty:
        plt.plot(result["equity_curve"].to_numpy())
    plt.title("Equity Curve")
    plt.xlabel("Trade number")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=140)
    plt.close()

    return {
        "trades_path": trades_path,
        "metrics_path": metrics_path,
        "equity_path": equity_path,
        "figure_path": figure_path,
    }
