from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import time

from forex_css.data.providers.oanda import OandaClient, OandaConfig
from forex_css.data.providers.twelvedata import TwelveDataClient, TwelveDataConfig


def _parse_datetime(text: str) -> datetime:
    if len(text) == 10:
        dt = datetime.fromisoformat(text)
        return dt.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download forex OHLC data (OANDA or Twelve Data).")
    parser.add_argument("--provider", type=str, default="twelvedata", choices=["oanda", "twelvedata"])
    parser.add_argument("--pairs", type=str, required=True, help="Comma-separated symbols, e.g. EURUSD,GBPUSD")
    parser.add_argument("--timeframes", type=str, required=True, help="Comma-separated TFs, e.g. H1,H4,D1")
    parser.add_argument("--start", type=str, required=True, help="YYYY-MM-DD or ISO datetime")
    parser.add_argument("--end", type=str, required=True, help="YYYY-MM-DD or ISO datetime")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", type=str, default="twelvedata")
    parser.add_argument("--oanda-environment", type=str, default="practice", choices=["practice", "live"])
    parser.add_argument("--oanda-token", type=str, default=None, help="Optional. If omitted, read OANDA_API_TOKEN.")
    parser.add_argument("--oanda-account-id", type=str, default=None, help="Optional. If omitted, read OANDA_ACCOUNT_ID.")
    parser.add_argument(
        "--twelvedata-api-key",
        type=str,
        default=None,
        help="Optional. If omitted, read TWELVEDATA_API_KEY.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue downloading other pair/timeframe on error.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Sleep seconds between symbol-timeframe requests (useful for free-tier rate limits).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional CSV log path. Default: <data-root>/<source>/_logs/download_<timestamp>.csv",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Default: same folder as --log-path with _summary.json suffix.",
    )
    return parser.parse_args()


def _write_csv_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_time_utc",
        "provider",
        "pair",
        "timeframe",
        "status",
        "duration_sec",
        "output_path",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if not pairs or not tfs:
        raise ValueError("Provide non-empty --pairs and --timeframes")

    started_at = datetime.now(UTC)
    default_log_path = args.data_root / args.source / "_logs" / f"download_{started_at:%Y%m%d_%H%M%S}.csv"
    log_path = args.log_path or default_log_path
    default_summary_path = log_path.with_name(log_path.stem + "_summary.json")
    summary_path = args.summary_path or default_summary_path
    log_rows: list[dict[str, str]] = []

    if args.provider == "oanda":
        token = args.oanda_token or os.getenv("OANDA_API_TOKEN")
        account_id = args.oanda_account_id or os.getenv("OANDA_ACCOUNT_ID")
        if not token:
            raise ValueError("Missing OANDA token. Set --oanda-token or OANDA_API_TOKEN.")
        client: OandaClient | TwelveDataClient = OandaClient(
            OandaConfig(
                token=token,
                account_id=account_id,
                environment=args.oanda_environment,
            )
        )
    else:
        api_key = args.twelvedata_api_key or os.getenv("TWELVEDATA_API_KEY")
        if not api_key:
            raise ValueError("Missing Twelve Data API key. Set --twelvedata-api-key or TWELVEDATA_API_KEY.")
        client = TwelveDataClient(
            TwelveDataConfig(
                api_key=api_key,
            )
        )

    total = len(pairs) * len(tfs)
    done = 0
    errors: list[str] = []

    for pair in pairs:
        for tf in tfs:
            done += 1
            output_path = args.data_root / args.source / pair / f"{tf}.parquet"
            t0 = time.perf_counter()
            try:
                saved = client.download_symbol_timeframe_to_parquet(
                    symbol=pair,
                    timeframe=tf,
                    start=start,
                    end=end,
                    output_path=output_path,
                )
                elapsed = time.perf_counter() - t0
                print(f"[{done}/{total}] Saved {pair} {tf}: {saved}")
                log_rows.append(
                    {
                        "event_time_utc": datetime.now(UTC).isoformat(),
                        "provider": args.provider,
                        "pair": pair,
                        "timeframe": tf,
                        "status": "success",
                        "duration_sec": f"{elapsed:.3f}",
                        "output_path": str(saved),
                        "error": "",
                    }
                )
            except Exception as exc:
                msg = f"[{done}/{total}] ERROR {pair} {tf}: {exc}"
                elapsed = time.perf_counter() - t0
                log_rows.append(
                    {
                        "event_time_utc": datetime.now(UTC).isoformat(),
                        "provider": args.provider,
                        "pair": pair,
                        "timeframe": tf,
                        "status": "error",
                        "duration_sec": f"{elapsed:.3f}",
                        "output_path": str(output_path),
                        "error": str(exc),
                    }
                )
                if args.continue_on_error:
                    print(msg)
                    errors.append(msg)
                else:
                    _write_csv_log(log_path, log_rows)
                    raise
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

    success = total - len(errors)
    _write_csv_log(log_path, log_rows)
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "provider": args.provider,
        "source": args.source,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_requests": total,
        "success_requests": success,
        "failed_requests": len(errors),
        "log_path": str(log_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Download completed. Success={success} Failed={len(errors)}")
    print(f"CSV log: {log_path}")
    print(f"Summary: {summary_path}")
    if errors:
        print("Failed items:")
        for line in errors:
            print(line)


if __name__ == "__main__":
    main()
