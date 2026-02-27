from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from forex_css.data.providers.oanda import OandaClient, OandaConfig


def _parse_datetime(text: str) -> datetime:
    if len(text) == 10:
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download forex OHLC data (OANDA v20).")
    parser.add_argument("--provider", type=str, default="oanda", choices=["oanda"])
    parser.add_argument("--pairs", type=str, required=True, help="Comma-separated symbols, e.g. EURUSD,GBPUSD")
    parser.add_argument("--timeframes", type=str, required=True, help="Comma-separated TFs, e.g. H1,H4,D1")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD or ISO datetime")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD or ISO datetime")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", type=str, default="oanda")
    parser.add_argument("--oanda-environment", type=str, default="practice", choices=["practice", "live"])
    parser.add_argument("--oanda-token", type=str, default=None, help="Optional. If omitted, read OANDA_API_TOKEN.")
    parser.add_argument("--oanda-account-id", type=str, default=None, help="Optional. If omitted, read OANDA_ACCOUNT_ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.oanda_token or os.getenv("OANDA_API_TOKEN")
    account_id = args.oanda_account_id or os.getenv("OANDA_ACCOUNT_ID")
    if not token:
        raise ValueError("Missing OANDA token. Set --oanda-token or OANDA_API_TOKEN.")

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if not pairs or not tfs:
        raise ValueError("Provide non-empty --pairs and --timeframes")

    client = OandaClient(
        OandaConfig(
            token=token,
            account_id=account_id,
            environment=args.oanda_environment,
        )
    )

    for pair in pairs:
        for tf in tfs:
            output_path = args.data_root / args.source / pair / f"{tf}.parquet"
            saved = client.download_symbol_timeframe_to_parquet(
                symbol=pair,
                timeframe=tf,
                start=start,
                end=end,
                output_path=output_path,
            )
            print(f"Saved {pair} {tf}: {saved}")


if __name__ == "__main__":
    main()
