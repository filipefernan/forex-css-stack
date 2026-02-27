from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from forex_css.constants import CURRENCIES, currency_occurrences, parse_symbol


@dataclass(frozen=True)
class CSSConfig:
    lwma_period: int = 21
    atr_period: int = 100
    atr_shift_bars: int = 10
    ignore_future: bool = True
    add_sunday_to_monday: bool = True
    timeframe: str | None = None


def lwma(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be > 0")
    weights = np.arange(1, period + 1, dtype=float)
    divisor = weights.sum()
    return series.rolling(period).apply(lambda x: float(np.dot(x, weights) / divisor), raw=True)


def calc_tma_with_future(close: pd.Series, half_window: int = 20) -> pd.Series:
    values = close.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        weighted_sum = values[i] * (half_window + 1)
        weighted_total = float(half_window + 1)

        for j in range(1, half_window + 1):
            w = half_window + 1 - j
            left_idx = i - j
            if left_idx >= 0:
                weighted_sum += values[left_idx] * w
                weighted_total += w

            right_idx = i + j
            if right_idx < n:
                weighted_sum += values[right_idx] * w
                weighted_total += w

        out[i] = weighted_sum / weighted_total if weighted_total else np.nan

    return pd.Series(out, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr_1 = high - low
    tr_2 = (high - prev_close).abs()
    tr_3 = (low - prev_close).abs()
    return pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high=high, low=low, close=close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _apply_sunday_adjustment(series: pd.Series, timeframe: str | None, enabled: bool) -> pd.Series:
    if not enabled or timeframe is None or timeframe.upper() != "D1":
        return series
    sunday_mask = series.index.weekday == 6
    if not sunday_mask.any():
        return series
    return series.where(~sunday_mask, series.shift(1))


def get_slope(candles: pd.DataFrame, config: CSSConfig | None = None) -> pd.Series:
    cfg = config or CSSConfig()
    required = ("high", "low", "close")
    missing = [col for col in required if col not in candles.columns]
    if missing:
        raise ValueError(f"Candles missing required columns: {missing}")

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)

    if cfg.ignore_future:
        tma_like = lwma(close, period=cfg.lwma_period)
        prev = (tma_like.shift(1) * 231.0 + close * 20.0) / 251.0
    else:
        tma_like = calc_tma_with_future(close, half_window=cfg.lwma_period - 1)
        prev = tma_like.shift(1)

    atr = atr_wilder(high=high, low=low, close=close, period=cfg.atr_period)
    atr = atr.shift(cfg.atr_shift_bars) / 10.0

    slope = (tma_like - prev) / atr
    slope = _apply_sunday_adjustment(
        series=slope,
        timeframe=cfg.timeframe,
        enabled=cfg.add_sunday_to_monday,
    )
    slope.name = "slope"
    return slope.replace([np.inf, -np.inf], np.nan)


def calc_css_from_slopes(slopes_by_symbol: Mapping[str, pd.Series]) -> pd.DataFrame:
    if not slopes_by_symbol:
        raise ValueError("slopes_by_symbol is empty")

    normalized = {symbol.upper(): series.sort_index() for symbol, series in slopes_by_symbol.items()}
    slopes_frame = pd.concat(normalized, axis=1, join="inner").dropna(how="any")
    symbols = list(normalized.keys())
    counts = currency_occurrences(symbols)

    css = pd.DataFrame(0.0, index=slopes_frame.index, columns=list(CURRENCIES))
    for symbol in symbols:
        base, quote = parse_symbol(symbol)
        slope = slopes_frame[symbol]
        css[base] += slope
        css[quote] -= slope

    for currency in CURRENCIES:
        occ = counts.get(currency, 0)
        if occ > 0:
            css[currency] = css[currency] / occ
        else:
            css[currency] = 0.0

    return css


def calculate_css_from_candles(
    candles_by_symbol: Mapping[str, pd.DataFrame],
    config: CSSConfig | None = None,
) -> pd.DataFrame:
    cfg = config or CSSConfig()
    slopes: dict[str, pd.Series] = {}

    for symbol, candles in candles_by_symbol.items():
        slopes[symbol.upper()] = get_slope(candles, config=cfg)

    return calc_css_from_slopes(slopes_by_symbol=slopes)
