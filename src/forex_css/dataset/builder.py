from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from forex_css.constants import CURRENCIES, parse_symbol
from forex_css.data.loaders import load_pairs_from_data_root


@dataclass(frozen=True)
class DecisionConfig:
    mode: str = "daily"  # daily | hourly
    timezone: str = "America/Bahia"
    decision_time: str = "21:00"
    hourly_step: int = 1


@dataclass(frozen=True)
class CostConfig:
    spread_bps: float = 1.5
    slippage_bps: float = 0.5

    @property
    def roundtrip_cost_return(self) -> float:
        # Cost in decimal return terms.
        return (self.spread_bps + (2.0 * self.slippage_bps)) / 10000.0


def _parse_hour_minute(text: str) -> tuple[int, int]:
    parts = text.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {text!r}. Use HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time value: {text!r}")
    return hour, minute


def build_decision_timestamps(index_utc: pd.DatetimeIndex, config: DecisionConfig) -> pd.DatetimeIndex:
    if index_utc.tz is None:
        raise ValueError("index_utc must be timezone-aware UTC index")
    if index_utc.empty:
        raise ValueError("index_utc is empty")

    local = index_utc.tz_convert(config.timezone)
    hour, minute = _parse_hour_minute(config.decision_time)
    mode = config.mode.lower()

    if mode == "daily":
        mask = (local.hour == hour) & (local.minute == minute)
    elif mode == "hourly":
        if config.hourly_step <= 0:
            raise ValueError("hourly_step must be > 0")
        mask = (local.minute == minute) & ((local.hour % config.hourly_step) == 0)
    else:
        raise ValueError(f"Unsupported decision mode: {config.mode}. Use daily or hourly.")

    selected = index_utc[mask]
    selected = selected[~selected.duplicated()].sort_values()
    return selected


def build_decision_grid(decision_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    if decision_timestamps.empty:
        raise ValueError("No decision timestamps selected.")
    index = pd.MultiIndex.from_product(
        [decision_timestamps, list(CURRENCIES)],
        names=["timestamp", "currency"],
    )
    grid = index.to_frame(index=False)
    return grid


def load_feature_frames(feature_root: str | Path, timeframes: Iterable[str]) -> dict[str, pd.DataFrame]:
    root = Path(feature_root)
    out: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        tf_clean = tf.strip().upper()
        path = root / tf_clean / "currency_features.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["currency"] = frame["currency"].astype(str).str.upper()
        out[tf_clean] = frame.sort_values(["currency", "timestamp"]).reset_index(drop=True)
    return out


def _align_tf_features_asof(decision_grid: pd.DataFrame, tf_frame: pd.DataFrame, tf: str) -> pd.DataFrame:
    tf_clean = tf.upper()
    cols_to_drop = [c for c in ("timeframe",) if c in tf_frame.columns]
    frame = tf_frame.drop(columns=cols_to_drop).copy()
    value_cols = [c for c in frame.columns if c not in ("timestamp", "currency")]
    prefixed = {c: f"{tf_clean.lower()}_{c}" for c in value_cols}

    results: list[pd.DataFrame] = []
    for currency in CURRENCIES:
        left = (
            decision_grid[decision_grid["currency"] == currency][["timestamp", "currency"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        right = (
            frame[frame["currency"] == currency]
            .drop(columns=["currency"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        merged = pd.merge_asof(
            left=left,
            right=right,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.rename(columns=prefixed)
        results.append(merged)

    out = pd.concat(results, axis=0, ignore_index=True)
    out = out.sort_values(["timestamp", "currency"]).reset_index(drop=True)
    return out


def merge_multi_timeframe_features(
    decision_grid: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged = decision_grid.sort_values(["timestamp", "currency"]).reset_index(drop=True)
    for tf, frame in feature_frames.items():
        aligned = _align_tf_features_asof(decision_grid=merged, tf_frame=frame, tf=tf)
        merged = merged.merge(aligned, on=["timestamp", "currency"], how="left")
    return merged


def add_congruence_features(dataset: pd.DataFrame, timeframes: Iterable[str]) -> pd.DataFrame:
    out = dataset.copy()
    sign_cols = [f"{tf.strip().lower()}_css_sign" for tf in timeframes]
    missing = [c for c in sign_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing sign columns for congruence: {missing}")

    signs = out[sign_cols].fillna(0.0)
    pos_count = (signs > 0).sum(axis=1)
    neg_count = (signs < 0).sum(axis=1)

    out["tf_pos_count"] = pos_count.astype(int)
    out["tf_neg_count"] = neg_count.astype(int)
    out["tf_congruence_count"] = np.maximum(pos_count, neg_count).astype(int)
    out["tf_congruence_direction"] = np.where(pos_count > neg_count, 1, np.where(neg_count > pos_count, -1, 0))
    out["tf_all_agree"] = ((pos_count == len(sign_cols)) | (neg_count == len(sign_cols))).astype(int)

    css_cols = [f"{tf.strip().lower()}_css" for tf in timeframes if f"{tf.strip().lower()}_css" in out.columns]
    if css_cols:
        out["tf_css_mean"] = out[css_cols].mean(axis=1)
        out["tf_css_abs_mean"] = out[css_cols].abs().mean(axis=1)
        out["tf_css_std"] = out[css_cols].std(axis=1, ddof=0)

    return out


def _pair_returns_for_horizon(
    close_series: pd.Series,
    decision_timestamps: pd.DatetimeIndex,
    horizon_hours: int,
) -> pd.Series:
    if close_series.empty:
        return pd.Series(index=decision_timestamps, dtype=float)

    prices = close_series.sort_index().dropna().to_frame(name="close").reset_index()
    prices = prices.rename(columns={prices.columns[0]: "px_time"})

    left = pd.DataFrame({"timestamp": decision_timestamps})
    left["exit_timestamp"] = left["timestamp"] + pd.to_timedelta(horizon_hours, unit="h")

    entry = pd.merge_asof(
        left=left[["timestamp"]].sort_values("timestamp"),
        right=prices[["px_time", "close"]].sort_values("px_time"),
        left_on="timestamp",
        right_on="px_time",
        direction="backward",
        allow_exact_matches=True,
    ).rename(columns={"close": "entry_close"})

    exit_ = pd.merge_asof(
        left=left[["exit_timestamp"]].sort_values("exit_timestamp"),
        right=prices[["px_time", "close"]].sort_values("px_time"),
        left_on="exit_timestamp",
        right_on="px_time",
        direction="backward",
        allow_exact_matches=True,
    ).rename(columns={"close": "exit_close"})

    merged = pd.concat([left.sort_values("timestamp").reset_index(drop=True), entry["entry_close"], exit_["exit_close"]], axis=1)
    gross = (merged["exit_close"] / merged["entry_close"]) - 1.0
    return pd.Series(gross.to_numpy(dtype=float), index=merged["timestamp"])


def _currency_related_pairs(pairs: Iterable[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {currency: [] for currency in CURRENCIES}
    for pair in pairs:
        base, quote = parse_symbol(pair)
        mapping[base].append(pair.upper())
        mapping[quote].append(pair.upper())
    return mapping


def compute_basket_targets(
    decision_timestamps: pd.DatetimeIndex,
    candles_by_pair: dict[str, pd.DataFrame],
    horizons_hours: Iterable[int],
    cost_config: CostConfig,
) -> pd.DataFrame:
    if decision_timestamps.empty:
        raise ValueError("decision_timestamps is empty")

    pairs = [p.upper() for p in candles_by_pair]
    related = _currency_related_pairs(pairs)
    out = build_decision_grid(decision_timestamps).sort_values(["timestamp", "currency"]).reset_index(drop=True)

    for horizon in horizons_hours:
        if horizon <= 0:
            raise ValueError("All horizons must be > 0")

        pair_returns: dict[str, pd.Series] = {}
        for pair, candles in candles_by_pair.items():
            pair_returns[pair.upper()] = _pair_returns_for_horizon(
                close_series=candles["close"].astype(float),
                decision_timestamps=decision_timestamps,
                horizon_hours=horizon,
            )

        strong_vals: list[float] = []
        weak_vals: list[float] = []
        for row in out.itertuples(index=False):
            ts = row.timestamp
            currency = row.currency
            legs_strong: list[float] = []
            legs_weak: list[float] = []
            for pair in related[currency]:
                gross_pair_ret = pair_returns[pair].get(ts, np.nan)
                if np.isnan(gross_pair_ret):
                    continue
                base, quote = parse_symbol(pair)
                direction_strong = 1.0 if currency == base else -1.0
                direction_weak = -direction_strong
                legs_strong.append((direction_strong * gross_pair_ret) - cost_config.roundtrip_cost_return)
                legs_weak.append((direction_weak * gross_pair_ret) - cost_config.roundtrip_cost_return)

            strong_vals.append(float(np.mean(legs_strong)) if legs_strong else np.nan)
            weak_vals.append(float(np.mean(legs_weak)) if legs_weak else np.nan)

        out[f"target_ret_h{horizon}_strong"] = strong_vals
        out[f"target_ret_h{horizon}_weak"] = weak_vals
        out[f"target_cls_h{horizon}_strong"] = (out[f"target_ret_h{horizon}_strong"] > 0).astype("Int64")
        out[f"target_cls_h{horizon}_weak"] = (out[f"target_ret_h{horizon}_weak"] > 0).astype("Int64")

    return out


def build_dataset(
    feature_root: str | Path,
    timeframes: Iterable[str],
    decision_config: DecisionConfig,
    data_root: str | Path,
    source: str,
    pairs: Iterable[str],
    target_timeframe: str,
    horizons_hours: Iterable[int],
    cost_config: CostConfig,
    decision_anchor_timeframe: str | None = None,
) -> pd.DataFrame:
    tf_list = [tf.strip().upper() for tf in timeframes]
    if not tf_list:
        raise ValueError("timeframes is empty")

    features = load_feature_frames(feature_root=feature_root, timeframes=tf_list)
    anchor_tf = (decision_anchor_timeframe or tf_list[0]).strip().upper()
    if anchor_tf not in features:
        raise ValueError(f"Decision anchor timeframe {anchor_tf} not found in feature frames.")

    anchor_index = pd.DatetimeIndex(features[anchor_tf]["timestamp"].dropna().unique()).tz_convert("UTC").sort_values()
    decision_timestamps = build_decision_timestamps(anchor_index, config=decision_config)
    decision_grid = build_decision_grid(decision_timestamps)

    merged = merge_multi_timeframe_features(decision_grid=decision_grid, feature_frames=features)
    merged = add_congruence_features(merged, timeframes=tf_list)

    candles = load_pairs_from_data_root(
        data_root=data_root,
        source=source,
        pairs=pairs,
        timeframe=target_timeframe,
    )
    targets = compute_basket_targets(
        decision_timestamps=decision_timestamps,
        candles_by_pair=candles,
        horizons_hours=horizons_hours,
        cost_config=cost_config,
    )

    dataset = merged.merge(targets, on=["timestamp", "currency"], how="left")
    dataset = dataset.sort_values(["timestamp", "currency"]).reset_index(drop=True)
    return dataset
