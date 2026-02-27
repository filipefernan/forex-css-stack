from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

from forex_css.constants import parse_symbol
from forex_css.data.schema import ensure_candle_schema
from forex_css.utils.timeframe import get_timeframe_spec


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _fmt_td_datetime(dt: datetime) -> str:
    return _as_utc(dt).strftime("%Y-%m-%d %H:%M:%S")


def to_twelvedata_symbol(symbol: str) -> str:
    base, quote = parse_symbol(symbol)
    return f"{base}/{quote}"


def to_twelvedata_interval(timeframe: str) -> str:
    spec = get_timeframe_spec(timeframe)
    mapping = {
        "M15": "15min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1day",
        "W1": "1week",
    }
    if spec.code not in mapping:
        raise ValueError(f"Unsupported timeframe for Twelve Data: {timeframe}")
    return mapping[spec.code]


@dataclass(frozen=True)
class TwelveDataConfig:
    api_key: str
    timeout_seconds: int = 30
    max_points_per_call: int = 5000
    max_retries_per_call: int = 6
    rate_limit_wait_seconds: int = 65
    base_url: str = "https://api.twelvedata.com"


class TwelveDataClient:
    def __init__(self, config: TwelveDataConfig):
        self.config = config
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        for attempt in range(1, self.config.max_retries_per_call + 1):
            response = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("status") == "error":
                msg = payload.get("message", "Unknown Twelve Data API error")
                msg_lower = msg.lower()
                if "run out of api credits for the current minute" in msg_lower:
                    if attempt == self.config.max_retries_per_call:
                        raise RuntimeError(f"Twelve Data rate limit after {attempt} attempts: {msg}")
                    wait_s = self.config.rate_limit_wait_seconds
                    print(f"[twelvedata] rate limit hit; waiting {wait_s}s (attempt {attempt}/{self.config.max_retries_per_call})")
                    time.sleep(wait_s)
                    continue
                if "no data is available on the specified dates" in msg_lower:
                    raise RuntimeError(f"Twelve Data no data: {msg}")
                raise RuntimeError(f"Twelve Data error: {msg}")
            return payload
        raise RuntimeError("Unexpected retry loop exit for Twelve Data request.")

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        spec = get_timeframe_spec(timeframe)
        td_symbol = to_twelvedata_symbol(symbol)
        interval = to_twelvedata_interval(timeframe)
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if end_utc <= start_utc:
            raise ValueError("end must be after start")

        # Keep calls smaller than provider hard limit for predictable behavior.
        window = timedelta(seconds=spec.seconds * self.config.max_points_per_call)
        cursor = start_utc
        rows: list[dict[str, Any]] = []

        while cursor < end_utc:
            chunk_end = min(cursor + window, end_utc)
            params = {
                "symbol": td_symbol,
                "interval": interval,
                "start_date": _fmt_td_datetime(cursor),
                "end_date": _fmt_td_datetime(chunk_end),
                "outputsize": self.config.max_points_per_call,
                "order": "ASC",
                # Forex defaults to Australia/Sydney; force UTC for stable alignment.
                "timezone": "UTC",
                "apikey": self.config.api_key,
            }
            payload = self._get("/time_series", params=params)
            values = payload.get("values", [])
            if not values:
                cursor = chunk_end
                continue

            for bar in values:
                dt = pd.to_datetime(bar["datetime"], utc=True, errors="coerce")
                if pd.isna(dt):
                    continue
                # Conservative anti-leakage convention: treat provider stamp as bar open and shift to close.
                close_ts = dt + pd.Timedelta(seconds=spec.seconds)
                rows.append(
                    {
                        "timestamp": close_ts,
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(bar.get("volume", 0.0)),
                    }
                )

            last_dt = pd.to_datetime(values[-1]["datetime"], utc=True, errors="coerce")
            if pd.isna(last_dt):
                cursor = chunk_end
            else:
                cursor = (last_dt + pd.Timedelta(seconds=spec.seconds)).to_pydatetime().replace(tzinfo=UTC)

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(rows)
        frame = ensure_candle_schema(frame)
        return frame

    def download_symbol_timeframe_to_parquet(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        output_path: str | Path,
    ) -> Path:
        candles = self.fetch_candles(symbol=symbol, timeframe=timeframe, start=start, end=end)
        if candles.empty:
            raise RuntimeError(f"No candles fetched for {symbol} {timeframe} in requested date range.")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        candles.to_parquet(out)
        return out
