from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual hybrid rule: M15 trigger + M30/H1 confirmation.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset parquet with M15/M30/H1 features.")
    parser.add_argument("--horizon-hours", type=int, default=4, help="Target horizon in hours (uses target_ret_h{h}_* columns).")
    parser.add_argument("--entry-extreme", type=float, default=0.20, help="Extreme threshold for CSS (+/- value).")
    parser.add_argument("--m15-extreme-bars", type=int, default=3, help="Minimum consecutive M15 bars in extreme zone.")
    parser.add_argument("--m15-lateral-window", type=int, default=5, help="Rolling window (bars) for M15 lateralization.")
    parser.add_argument("--m15-lateral-max-std", type=float, default=0.03, help="Max rolling std to consider M15 lateralized.")
    parser.add_argument(
        "--cut-ts",
        type=str,
        default="",
        help="UTC split timestamp (e.g. 2026-01-23 10:00:00+00:00). Empty = 75%%/25%% split by time.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/visual_hybrid"))
    parser.add_argument("--prefix", type=str, default="visual_hybrid")
    return parser.parse_args()


def _consecutive_true_counts(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(values), dtype=np.int32)
    count = 0
    for i, flag in enumerate(values):
        if flag:
            count += 1
        else:
            count = 0
        out[i] = count
    return pd.Series(out, index=mask.index)


def _performance_metrics(returns: pd.Series, periods_per_year: int = 252 * 24) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {
            "num_trades": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    equity = (1.0 + r).cumprod()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    std = r.std(ddof=0)
    sharpe = (r.mean() / std) * np.sqrt(periods_per_year) if std > 0 else 0.0
    return {
        "num_trades": float(len(r)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float(np.inf),
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "sharpe": float(sharpe),
    }


def _choose_cut_timestamp(ts: pd.DatetimeIndex, cut_ts_raw: str) -> pd.Timestamp:
    unique_ts = pd.DatetimeIndex(ts.dropna().unique()).sort_values()
    if len(unique_ts) == 0:
        raise ValueError("Dataset has no valid timestamps.")
    if cut_ts_raw.strip():
        cut = pd.Timestamp(cut_ts_raw)
        if cut.tzinfo is None:
            return cut.tz_localize("UTC")
        return cut.tz_convert("UTC")
    cut_pos = int(len(unique_ts) * 0.75)
    cut_pos = max(1, min(cut_pos, len(unique_ts) - 1))
    return pd.Timestamp(unique_ts[cut_pos]).tz_convert("UTC")


def main() -> None:
    args = parse_args()
    strong_col = f"target_ret_h{args.horizon_hours}_strong"
    weak_col = f"target_ret_h{args.horizon_hours}_weak"

    df = pd.read_parquet(args.dataset)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["currency", "timestamp"]).reset_index(drop=True)

    required = [
        "timestamp",
        "currency",
        "m15_css",
        "m15_css_slope",
        "m30_css",
        "h1_css",
        "h1_css_slope",
        strong_col,
        weak_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        if col not in ("timestamp", "currency"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    extreme = abs(float(args.entry_extreme))
    by_currency = df.groupby("currency", group_keys=False)

    df["m15_css_slope_prev"] = by_currency["m15_css_slope"].shift(1)
    df["m15_rolling_std"] = by_currency["m15_css"].transform(
        lambda s: s.rolling(args.m15_lateral_window, min_periods=args.m15_lateral_window).std(ddof=0)
    )

    low_mask = df["m15_css"] <= -extreme
    high_mask = df["m15_css"] >= extreme
    df["m15_low_streak"] = by_currency["m15_css"].transform(
        lambda s: _consecutive_true_counts(s <= -extreme).to_numpy()
    )
    df["m15_high_streak"] = by_currency["m15_css"].transform(
        lambda s: _consecutive_true_counts(s >= extreme).to_numpy()
    )

    m15_turn_up = (df["m15_css_slope_prev"] <= 0.0) & (df["m15_css_slope"] > 0.0)
    m15_turn_down = (df["m15_css_slope_prev"] >= 0.0) & (df["m15_css_slope"] < 0.0)

    m15_lateral_low = low_mask & (df["m15_rolling_std"] <= args.m15_lateral_max_std) & (df["m15_css_slope"] >= 0.0)
    m15_lateral_high = high_mask & (df["m15_rolling_std"] <= args.m15_lateral_max_std) & (df["m15_css_slope"] <= 0.0)

    m15_ready_long = (df["m15_low_streak"] >= args.m15_extreme_bars) & (m15_turn_up | m15_lateral_low)
    m15_ready_short = (df["m15_high_streak"] >= args.m15_extreme_bars) & (m15_turn_down | m15_lateral_high)

    # Hybrid confirmation requested by user:
    # M30 in extreme + H1 on same side of zero and inclining.
    conf_long = (df["m30_css"] <= -extreme) & (df["h1_css"] < 0.0) & (df["h1_css_slope"] > 0.0)
    conf_short = (df["m30_css"] >= extreme) & (df["h1_css"] > 0.0) & (df["h1_css_slope"] < 0.0)

    df["signal"] = np.where(m15_ready_long & conf_long, 1, np.where(m15_ready_short & conf_short, -1, 0))
    df["realized_return"] = np.where(df["signal"] == 1, df[strong_col], np.where(df["signal"] == -1, df[weak_col], np.nan))

    long_score = (-df["m15_css"]).clip(lower=0.0) + (-df["m30_css"]).clip(lower=0.0) + (-df["h1_css"]).clip(lower=0.0)
    short_score = (df["m15_css"]).clip(lower=0.0) + (df["m30_css"]).clip(lower=0.0) + (df["h1_css"]).clip(lower=0.0)
    df["score"] = np.where(df["signal"] == 1, long_score, np.where(df["signal"] == -1, short_score, np.nan))

    candidates = df[df["signal"] != 0].dropna(subset=["realized_return", "score"]).copy()
    trades = (
        candidates.sort_values(["timestamp", "score"], ascending=[True, False])
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    trades["direction"] = np.where(trades["signal"] == 1, "strong", "weak")

    cut_ts = _choose_cut_timestamp(pd.DatetimeIndex(df["timestamp"]), args.cut_ts)
    train = trades[trades["timestamp"] < cut_ts].copy()
    test = trades[trades["timestamp"] >= cut_ts].copy()

    metrics = {
        "cut_ts": str(cut_ts),
        "config": {
            "horizon_hours": args.horizon_hours,
            "entry_extreme": extreme,
            "m15_extreme_bars": args.m15_extreme_bars,
            "m15_lateral_window": args.m15_lateral_window,
            "m15_lateral_max_std": args.m15_lateral_max_std,
        },
        "all": _performance_metrics(trades["realized_return"]),
        "train": _performance_metrics(train["realized_return"]),
        "test": _performance_metrics(test["realized_return"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.prefix}_metrics.json"
    trades_path = args.output_dir / f"{args.prefix}_trades.csv"
    train_path = args.output_dir / f"{args.prefix}_train_trades.csv"
    test_path = args.output_dir / f"{args.prefix}_test_trades.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    trades.to_csv(trades_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print("Saved:", metrics_path)
    print("Saved:", trades_path)
    print("Saved:", train_path)
    print("Saved:", test_path)
    print("train_metrics:", metrics["train"])
    print("test_metrics:", metrics["test"])


if __name__ == "__main__":
    main()
