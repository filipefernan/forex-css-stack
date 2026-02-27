from __future__ import annotations

import numpy as np
import pandas as pd

from forex_css.dataset.builder import (
    CostConfig,
    DecisionConfig,
    add_congruence_features,
    build_decision_grid,
    build_decision_timestamps,
    compute_basket_targets,
)


def _make_hourly_candles(start: str, periods: int, base_price: float, drift: float) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    close = base_price + np.arange(periods) * drift
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.001
    low = np.minimum(open_, close) - 0.001
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_daily_decision_grid_has_8_rows_per_timestamp() -> None:
    idx = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="UTC")
    cfg = DecisionConfig(mode="daily", timezone="America/Bahia", decision_time="21:00")
    decision_ts = build_decision_timestamps(idx, cfg)
    grid = build_decision_grid(decision_ts)
    counts = grid.groupby("timestamp")["currency"].nunique()
    assert len(decision_ts) > 0
    assert (counts == 8).all()


def test_congruence_counts_are_computed() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"] * 2),
            "currency": ["EUR", "USD"],
            "h1_css_sign": [1, -1],
            "h4_css_sign": [1, -1],
            "d1_css_sign": [1, -1],
            "h1_css": [0.2, -0.2],
            "h4_css": [0.3, -0.3],
            "d1_css": [0.1, -0.1],
        }
    )
    out = add_congruence_features(frame, timeframes=["H1", "H4", "D1"])
    assert (out["tf_congruence_count"] == 3).all()
    assert set(out["tf_congruence_direction"].tolist()) == {1, -1}


def test_basket_target_direction_mapping() -> None:
    candles = {
        "EURUSD": _make_hourly_candles("2024-01-01", periods=200, base_price=1.1, drift=0.0003),
        "USDJPY": _make_hourly_candles("2024-01-01", periods=200, base_price=140.0, drift=-0.01),
    }
    decisions = pd.date_range("2024-01-03", periods=20, freq="6h", tz="UTC")
    targets = compute_basket_targets(
        decision_timestamps=decisions,
        candles_by_pair=candles,
        horizons_hours=[6],
        cost_config=CostConfig(spread_bps=0.0, slippage_bps=0.0),
    )
    eur = targets[targets["currency"] == "EUR"]["target_ret_h6_strong"].mean()
    usd = targets[targets["currency"] == "USD"]["target_ret_h6_strong"].mean()
    assert eur > 0
    assert usd < 0
