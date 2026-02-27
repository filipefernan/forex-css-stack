from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
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


def _to_oanda_time(dt: datetime) -> str:
    return _as_utc(dt).isoformat().replace("+00:00", "Z")


def to_oanda_instrument(symbol: str) -> str:
    base, quote = parse_symbol(symbol)
    return f"{base}_{quote}"


@dataclass(frozen=True)
class OandaConfig:
    token: str
    account_id: str | None = None
    environment: str = "practice"  # practice | live
    timeout_seconds: int = 30
    max_candles_per_call: int = 4500

    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"


class OandaClient:
    def __init__(self, config: OandaConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
            }
        )

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        price_component: str = "M",
        include_incomplete: bool = False,
    ) -> pd.DataFrame:
        spec = get_timeframe_spec(timeframe)
        instrument = to_oanda_instrument(symbol)
        start_utc = _as_utc(start)
        end_utc = _as_utc(end)
        if end_utc <= start_utc:
            raise ValueError("end must be after start")

        step = timedelta(seconds=spec.seconds * self.config.max_candles_per_call)
        cursor = start_utc
        rows: list[dict[str, Any]] = []

        while cursor < end_utc:
            chunk_end = min(cursor + step, end_utc)
            params = {
                "price": price_component,
                "granularity": spec.oanda_granularity,
                "from": _to_oanda_time(cursor),
                "to": _to_oanda_time(chunk_end),
            }
            payload = self._get(f"/v3/instruments/{instrument}/candles", params=params)
            candles = payload.get("candles", [])
            if not candles:
                cursor = chunk_end
                continue

            for candle in candles:
                if not include_incomplete and not candle.get("complete", False):
                    continue
                bucket = candle.get("mid") or candle.get("bid") or candle.get("ask")
                if not bucket:
                    continue
                open_time = pd.to_datetime(candle["time"], utc=True)
                close_time = open_time + pd.Timedelta(seconds=spec.seconds)
                rows.append(
                    {
                        "timestamp": close_time,
                        "open": float(bucket["o"]),
                        "high": float(bucket["h"]),
                        "low": float(bucket["l"]),
                        "close": float(bucket["c"]),
                        "volume": float(candle.get("volume", 0.0)),
                    }
                )

            last_open_time = pd.to_datetime(candles[-1]["time"], utc=True)
            cursor = (last_open_time + pd.Timedelta(seconds=spec.seconds)).to_pydatetime().replace(tzinfo=UTC)

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
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        candles.to_parquet(out)
        return out
