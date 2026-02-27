from __future__ import annotations

import numpy as np
import pandas as pd

from forex_css.constants import CURRENCIES


def build_single_tf_features(
    css_frame: pd.DataFrame,
    timeframe: str,
    level_cross_value: float = 0.20,
    zscore_window: int = 63,
) -> pd.DataFrame:
    if css_frame.empty:
        raise ValueError("css_frame is empty")
    missing = [c for c in CURRENCIES if c not in css_frame.columns]
    if missing:
        raise ValueError(f"css_frame missing currencies: {missing}")

    css_sorted = css_frame.sort_index()
    long = (
        css_sorted[CURRENCIES]
        .stack()
        .rename("css")
        .reset_index()
        .rename(columns={"level_0": "timestamp", "level_1": "currency"})
    )
    long["timeframe"] = timeframe.upper()

    long["css_prev"] = long.groupby("currency")["css"].shift(1)
    long["css_slope"] = long["css"] - long["css_prev"]
    long["css_sign"] = np.sign(long["css"])
    long["prev_sign"] = np.sign(long["css_prev"])
    long["is_reversal"] = ((long["css_sign"] * long["prev_sign"]) < 0).astype(int)

    long["cross_zero_up"] = ((long["css_prev"] <= 0.0) & (long["css"] > 0.0)).astype(int)
    long["cross_zero_down"] = ((long["css_prev"] >= 0.0) & (long["css"] < 0.0)).astype(int)
    long["cross_level_up"] = ((long["css_prev"] < level_cross_value) & (long["css"] >= level_cross_value)).astype(int)
    long["cross_level_down"] = (
        (long["css_prev"] > -level_cross_value) & (long["css"] <= -level_cross_value)
    ).astype(int)

    rolling_mean = long.groupby("currency")["css"].transform(
        lambda s: s.rolling(zscore_window, min_periods=max(20, zscore_window // 3)).mean()
    )
    rolling_std = long.groupby("currency")["css"].transform(
        lambda s: s.rolling(zscore_window, min_periods=max(20, zscore_window // 3)).std(ddof=0)
    )
    long["css_zscore"] = (long["css"] - rolling_mean) / rolling_std.replace(0, np.nan)
    long["is_extreme"] = (long["css_zscore"].abs() >= 1.5).astype(int)

    long["rank_desc"] = long.groupby("timestamp")["css"].rank(method="first", ascending=False).astype(int)
    long["rank_asc"] = long.groupby("timestamp")["css"].rank(method="first", ascending=True).astype(int)

    keep_cols = [
        "timestamp",
        "currency",
        "timeframe",
        "css",
        "css_prev",
        "css_slope",
        "css_sign",
        "rank_desc",
        "rank_asc",
        "cross_zero_up",
        "cross_zero_down",
        "cross_level_up",
        "cross_level_down",
        "css_zscore",
        "is_extreme",
        "is_reversal",
    ]
    return long[keep_cols].sort_values(["timestamp", "rank_desc"]).reset_index(drop=True)
