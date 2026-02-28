from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeSpec:
    code: str
    pandas_freq: str
    minutes: int
    oanda_granularity: str

    @property
    def seconds(self) -> int:
        return self.minutes * 60


TIMEFRAME_SPECS: dict[str, TimeframeSpec] = {
    "M15": TimeframeSpec(code="M15", pandas_freq="15min", minutes=15, oanda_granularity="M15"),
    "M30": TimeframeSpec(code="M30", pandas_freq="30min", minutes=30, oanda_granularity="M30"),
    "H1": TimeframeSpec(code="H1", pandas_freq="1h", minutes=60, oanda_granularity="H1"),
    "H4": TimeframeSpec(code="H4", pandas_freq="4h", minutes=240, oanda_granularity="H4"),
    "D1": TimeframeSpec(code="D1", pandas_freq="1D", minutes=24 * 60, oanda_granularity="D"),
    "W1": TimeframeSpec(code="W1", pandas_freq="1W", minutes=7 * 24 * 60, oanda_granularity="W"),
}


def get_timeframe_spec(timeframe: str) -> TimeframeSpec:
    key = timeframe.strip().upper()
    if key not in TIMEFRAME_SPECS:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported={tuple(TIMEFRAME_SPECS)}")
    return TIMEFRAME_SPECS[key]
