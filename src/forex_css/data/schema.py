from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

REQUIRED_OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True)
class CandleSchemaConfig:
    timestamp_col: str = "timestamp"
    timezone: str = "UTC"
    allow_volume: bool = True


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    return [c.strip().lower() for c in columns]


def ensure_candle_schema(df: pd.DataFrame, config: CandleSchemaConfig | None = None) -> pd.DataFrame:
    cfg = config or CandleSchemaConfig()
    frame = df.copy()
    frame.columns = _normalize_columns(frame.columns)

    ts_col = cfg.timestamp_col.lower()
    if ts_col in frame.columns:
        frame[ts_col] = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
        frame = frame.dropna(subset=[ts_col]).set_index(ts_col)
    elif frame.index.name is None:
        raise ValueError(
            f"Missing timestamp column '{cfg.timestamp_col}'. "
            "Provide a timestamp column or an indexed DataFrame."
        )
    else:
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    if cfg.timezone.upper() != "UTC":
        frame.index = frame.index.tz_convert(cfg.timezone)
    else:
        frame.index = frame.index.tz_convert("UTC")

    missing = [col for col in REQUIRED_OHLC_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = list(REQUIRED_OHLC_COLUMNS) + [c for c in ("volume", "tick_volume", "spread") if c in frame.columns]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=list(REQUIRED_OHLC_COLUMNS)).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame
