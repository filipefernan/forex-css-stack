from __future__ import annotations

import numpy as np
import pandas as pd

from forex_css.constants import CURRENCIES
from forex_css.features.single_tf import build_single_tf_features
from forex_css.indicators.css import CSSConfig, calculate_css_from_candles


def _make_candles(
    start: str = "2024-01-01",
    periods: int = 500,
    freq: str = "H",
    start_price: float = 1.0,
    drift: float = 0.0002,
    noise: float = 0.0001,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    increments = drift + rng.normal(0.0, noise, periods)
    close = start_price + np.cumsum(increments)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.0004
    low = np.minimum(open_, close) - 0.0004
    volume = np.full(periods, 1000.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_css_outputs_expected_shape_and_columns() -> None:
    candles = {
        "EURUSD": _make_candles(start_price=1.08, drift=0.00015),
        "GBPUSD": _make_candles(start_price=1.25, drift=0.00010),
        "USDJPY": _make_candles(start_price=145.0, drift=-0.010),
    }
    css = calculate_css_from_candles(candles, config=CSSConfig(ignore_future=True, timeframe="H1"))
    assert tuple(css.columns) == CURRENCIES
    valid = css.dropna()
    assert not valid.empty
    assert np.isfinite(valid.to_numpy()).all()


def test_single_pair_produces_opposite_css_on_base_and_quote() -> None:
    candles = {"EURUSD": _make_candles(start_price=1.10, drift=0.0002)}
    css = calculate_css_from_candles(candles, config=CSSConfig(ignore_future=True, timeframe="H1")).dropna()
    assert not css.empty
    assert np.allclose(css["EUR"], -css["USD"], atol=1e-10, equal_nan=False)
    for currency in ("GBP", "CHF", "JPY", "AUD", "CAD", "NZD"):
        assert np.allclose(css[currency], 0.0, atol=1e-12, equal_nan=False)


def test_trending_up_pair_tends_to_positive_base_css() -> None:
    candles = {"EURUSD": _make_candles(start_price=1.09, drift=0.0003, noise=0.00002)}
    css = calculate_css_from_candles(candles, config=CSSConfig(ignore_future=True, timeframe="H1")).dropna()
    assert css["EUR"].mean() > 0
    assert css["USD"].mean() < 0


def test_feature_builder_generates_full_ranks_per_timestamp() -> None:
    candles = {
        "EURUSD": _make_candles(start_price=1.08, drift=0.00015),
        "GBPUSD": _make_candles(start_price=1.25, drift=0.00010),
        "USDJPY": _make_candles(start_price=145.0, drift=-0.010),
        "AUDUSD": _make_candles(start_price=0.66, drift=0.00012),
    }
    css = calculate_css_from_candles(candles, config=CSSConfig(ignore_future=True, timeframe="H1")).dropna()
    features = build_single_tf_features(css, timeframe="H1")
    assert not features.empty

    ranks = features.groupby("timestamp")["rank_desc"].apply(lambda s: set(s.to_list()))
    for rank_set in ranks:
        assert rank_set == set(range(1, 9))
