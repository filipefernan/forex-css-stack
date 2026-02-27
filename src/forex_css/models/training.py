from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


@dataclass(frozen=True)
class WalkForwardConfig:
    n_splits: int = 5
    min_train_timestamps: int = 120
    min_prediction_edge: float = 0.0


def performance_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {
            "num_trades": 0.0,
            "total_return": 0.0,
            "cagr_approx": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    equity = (1.0 + clean).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    sharpe = 0.0
    if clean.std(ddof=0) > 0:
        sharpe = (clean.mean() / clean.std(ddof=0)) * np.sqrt(periods_per_year)

    gains = clean[clean > 0].sum()
    losses = -clean[clean < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else np.inf

    years = max(len(clean) / periods_per_year, 1e-6)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    return {
        "num_trades": float(len(clean)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr_approx": cagr,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((clean > 0).mean()),
        "profit_factor": float(profit_factor),
    }


def infer_periods_per_year(dataset: pd.DataFrame) -> int:
    ts = pd.DatetimeIndex(dataset["timestamp"].dropna().unique()).sort_values()
    if len(ts) < 2:
        return 252
    median_delta = (ts[1:] - ts[:-1]).median()
    hours = median_delta / pd.Timedelta(hours=1)
    if hours <= 1.01:
        return 24 * 252
    if hours <= 4.01:
        return 6 * 252
    return 252


def make_walk_forward_splits(
    timestamps: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    unique_ts = pd.DatetimeIndex(pd.Series(timestamps).dropna().unique()).sort_values()
    total = len(unique_ts)
    if total <= config.min_train_timestamps + 1:
        raise ValueError(
            f"Not enough timestamps for walk-forward. total={total}, min_train={config.min_train_timestamps}"
        )

    available_test = total - config.min_train_timestamps
    test_size = max(1, available_test // config.n_splits)
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    train_end = config.min_train_timestamps
    for _ in range(config.n_splits):
        test_start = train_end
        test_end = min(total, test_start + test_size)
        if test_start >= total:
            break
        train_ts = unique_ts[:test_start]
        test_ts = unique_ts[test_start:test_end]
        if len(test_ts) == 0:
            break
        splits.append((train_ts, test_ts))
        train_end = test_end

    return splits


def build_regressor(model_name: str, random_state: int = 42) -> tuple[Any, str]:
    key = model_name.strip().lower()
    if key == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )
            return model, "lightgbm"
        except Exception:
            fallback = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.03,
                max_iter=300,
                random_state=random_state,
            )
            return fallback, "hist_gradient_boosting_fallback"
    if key in {"random_forest", "rf", "randomforest"}:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "random_forest"
    if key == "catboost":
        try:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(
                depth=6,
                learning_rate=0.03,
                iterations=600,
                loss_function="RMSE",
                verbose=False,
                random_seed=random_state,
            )
            return model, "catboost"
        except Exception:
            fallback = HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.03,
                max_iter=300,
                random_state=random_state,
            )
            return fallback, "hist_gradient_boosting_fallback"
    raise ValueError(f"Unsupported model: {model_name}. Use lightgbm, catboost or random_forest.")


def _pick_best_trade_per_timestamp(
    frame: pd.DataFrame,
    pred_strong_col: str,
    pred_weak_col: str,
    target_strong_col: str,
    target_weak_col: str,
    min_prediction_edge: float,
) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    for ts, group in frame.groupby("timestamp", sort=True):
        g = group.dropna(subset=[pred_strong_col, pred_weak_col])
        if g.empty:
            continue

        idx_strong = g[pred_strong_col].idxmax()
        idx_weak = g[pred_weak_col].idxmax()
        best_strong = float(g.loc[idx_strong, pred_strong_col])
        best_weak = float(g.loc[idx_weak, pred_weak_col])

        if max(best_strong, best_weak) <= min_prediction_edge:
            continue

        if best_strong >= best_weak:
            chosen = g.loc[idx_strong]
            direction = "strong"
            pred = best_strong
            realized = float(chosen[target_strong_col])
        else:
            chosen = g.loc[idx_weak]
            direction = "weak"
            pred = best_weak
            realized = float(chosen[target_weak_col])

        trades.append(
            {
                "timestamp": ts,
                "currency": chosen["currency"],
                "direction": direction,
                "prediction": pred,
                "realized_return": realized,
            }
        )

    return pd.DataFrame(trades)


def baseline_congruence_trades(
    dataset: pd.DataFrame,
    target_strong_col: str,
    target_weak_col: str,
) -> pd.DataFrame:
    required = ["tf_congruence_count", "tf_congruence_direction"]
    for col in required:
        if col not in dataset.columns:
            raise ValueError(f"Missing baseline column: {col}")

    trades: list[dict[str, Any]] = []
    for ts, group in dataset.groupby("timestamp", sort=True):
        g = group.copy()
        g = g[g["tf_congruence_direction"] != 0]
        if g.empty:
            continue
        sort_cols = ["tf_congruence_count"]
        if "tf_css_abs_mean" in g.columns:
            sort_cols.append("tf_css_abs_mean")
        g = g.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        chosen = g.iloc[0]
        direction = "strong" if int(chosen["tf_congruence_direction"]) > 0 else "weak"
        realized = float(chosen[target_strong_col] if direction == "strong" else chosen[target_weak_col])
        trades.append(
            {
                "timestamp": ts,
                "currency": chosen["currency"],
                "direction": direction,
                "prediction": float(chosen["tf_congruence_count"]),
                "realized_return": realized,
            }
        )
    return pd.DataFrame(trades)


def select_feature_columns(dataset: pd.DataFrame) -> list[str]:
    numeric = dataset.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if not c.startswith("target_")]
    for banned in ("tf_congruence_direction",):
        if banned in feature_cols:
            feature_cols.remove(banned)
    return feature_cols


def walk_forward_train_and_score(
    dataset: pd.DataFrame,
    horizon_hours: int,
    model_name: str,
    wf_config: WalkForwardConfig,
    random_state: int = 42,
) -> dict[str, Any]:
    target_strong = f"target_ret_h{horizon_hours}_strong"
    target_weak = f"target_ret_h{horizon_hours}_weak"
    for col in (target_strong, target_weak):
        if col not in dataset.columns:
            raise ValueError(f"Missing target column: {col}")

    frame = dataset.copy().sort_values(["timestamp", "currency"]).reset_index(drop=True)
    feature_cols = select_feature_columns(frame)
    if not feature_cols:
        raise ValueError("No feature columns available for training.")

    splits = make_walk_forward_splits(pd.DatetimeIndex(frame["timestamp"]), config=wf_config)
    split_predictions: list[pd.DataFrame] = []
    split_trades: list[pd.DataFrame] = []
    model_used_labels: list[str] = []

    for split_id, (train_ts, test_ts) in enumerate(splits, start=1):
        train_mask = frame["timestamp"].isin(train_ts)
        test_mask = frame["timestamp"].isin(test_ts)
        train_df = frame[train_mask].dropna(subset=feature_cols + [target_strong, target_weak])
        test_df = frame[test_mask].dropna(subset=feature_cols)
        if train_df.empty or test_df.empty:
            continue

        model_strong, model_used = build_regressor(model_name=model_name, random_state=random_state + split_id)
        model_weak, _ = build_regressor(model_name=model_name, random_state=random_state + 100 + split_id)
        model_used_labels.append(model_used)

        model_strong.fit(train_df[feature_cols], train_df[target_strong])
        model_weak.fit(train_df[feature_cols], train_df[target_weak])

        scored = test_df[["timestamp", "currency", target_strong, target_weak]].copy()
        scored["pred_strong"] = model_strong.predict(test_df[feature_cols])
        scored["pred_weak"] = model_weak.predict(test_df[feature_cols])
        scored["split_id"] = split_id
        split_predictions.append(scored)

        trades = _pick_best_trade_per_timestamp(
            frame=scored,
            pred_strong_col="pred_strong",
            pred_weak_col="pred_weak",
            target_strong_col=target_strong,
            target_weak_col=target_weak,
            min_prediction_edge=wf_config.min_prediction_edge,
        )
        trades["split_id"] = split_id
        split_trades.append(trades)

    predictions = pd.concat(split_predictions, axis=0, ignore_index=True) if split_predictions else pd.DataFrame()
    trades = pd.concat(split_trades, axis=0, ignore_index=True) if split_trades else pd.DataFrame()

    baseline_trades = baseline_congruence_trades(
        dataset=frame[frame["timestamp"].isin(predictions["timestamp"].unique())] if not predictions.empty else frame.iloc[0:0],
        target_strong_col=target_strong,
        target_weak_col=target_weak,
    )
    periods = infer_periods_per_year(frame)

    result = {
        "feature_columns": feature_cols,
        "target_strong": target_strong,
        "target_weak": target_weak,
        "model_requested": model_name,
        "model_used": model_used_labels[-1] if model_used_labels else model_name,
        "predictions": predictions,
        "trades": trades,
        "baseline_trades": baseline_trades,
        "metrics_model": performance_metrics(trades["realized_return"] if not trades.empty else pd.Series(dtype=float), periods),
        "metrics_baseline": performance_metrics(
            baseline_trades["realized_return"] if not baseline_trades.empty else pd.Series(dtype=float),
            periods,
        ),
    }
    return result


def fit_final_models(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target_strong: str,
    target_weak: str,
    model_name: str,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = dataset.dropna(subset=feature_cols + [target_strong, target_weak]).copy()
    if frame.empty:
        raise ValueError("No rows available to fit final models.")

    model_strong, model_used = build_regressor(model_name=model_name, random_state=random_state)
    model_weak, _ = build_regressor(model_name=model_name, random_state=random_state + 999)
    model_strong.fit(frame[feature_cols], frame[target_strong])
    model_weak.fit(frame[feature_cols], frame[target_weak])
    return {
        "model_used": model_used,
        "model_strong": model_strong,
        "model_weak": model_weak,
        "feature_columns": feature_cols,
        "target_strong": target_strong,
        "target_weak": target_weak,
    }
