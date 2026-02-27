from __future__ import annotations

from collections import Counter
from typing import Iterable

CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "CHF", "JPY", "AUD", "CAD", "NZD")

# Default universe from the original MQ4 source.
DEFAULT_SYMBOLS_TO_WEIGH: tuple[str, ...] = (
    "AUDCAD",
    "AUDCHF",
    "AUDJPY",
    "AUDNZD",
    "AUDUSD",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "EURJPY",
    "EURNZD",
    "EURUSD",
    "GBPAUD",
    "GBPCAD",
    "GBPCHF",
    "GBPJPY",
    "GBPNZD",
    "GBPUSD",
    "NZDCHF",
    "NZDJPY",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
)


def parse_symbol(symbol: str) -> tuple[str, str]:
    clean = symbol.strip().upper()
    if len(clean) < 6:
        raise ValueError(f"Symbol must have at least 6 chars (got {symbol!r})")
    base = clean[:3]
    quote = clean[3:6]
    if base not in CURRENCIES or quote not in CURRENCIES:
        raise ValueError(f"Unsupported symbol {symbol!r}. Base/quote must be in {CURRENCIES}.")
    return base, quote


def currency_occurrences(symbols: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for symbol in symbols:
        base, quote = parse_symbol(symbol)
        counts[base] += 1
        counts[quote] += 1
    return counts
