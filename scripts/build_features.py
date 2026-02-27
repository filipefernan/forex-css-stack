from __future__ import annotations

import argparse
from pathlib import Path

from forex_css.constants import DEFAULT_SYMBOLS_TO_WEIGH
from forex_css.data.loaders import load_pairs_from_data_root
from forex_css.features.single_tf import build_single_tf_features
from forex_css.indicators.css import CSSConfig, calculate_css_from_candles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Currency Slope Strength features for one timeframe.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"), help="Base directory with raw candles.")
    parser.add_argument("--source", type=str, default="local", help="Source folder under data-root.")
    parser.add_argument("--timeframe", type=str, default="D1", help="Single timeframe (M15/H1/H4/D1/W1).")
    parser.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help="Optional comma-separated list to build multiple timeframes at once.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=",".join(DEFAULT_SYMBOLS_TO_WEIGH),
        help="Comma-separated list like EURUSD,GBPUSD,USDJPY",
    )
    parser.add_argument(
        "--output-css",
        type=Path,
        default=None,
        help="Optional output path for raw CSS wide frame (parquet).",
    )
    parser.add_argument(
        "--output-features",
        type=Path,
        default=None,
        help="Optional output path for long features frame (parquet).",
    )
    parser.add_argument("--level-cross-value", type=float, default=0.20)
    parser.add_argument("--output-root", type=Path, default=Path("data/features"), help="Base folder for outputs.")
    parser.add_argument(
        "--ignore-future",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, use LWMA path compatible with non-repainting mode.",
    )
    return parser.parse_args()


def _build_for_timeframe(args: argparse.Namespace, pairs: list[str], timeframe: str) -> None:
    candles = load_pairs_from_data_root(
        data_root=args.data_root,
        source=args.source,
        pairs=pairs,
        timeframe=timeframe,
    )
    css_config = CSSConfig(ignore_future=args.ignore_future, timeframe=timeframe)
    css = calculate_css_from_candles(candles_by_symbol=candles, config=css_config)
    features = build_single_tf_features(
        css_frame=css,
        timeframe=timeframe,
        level_cross_value=args.level_cross_value,
    )

    output_css = args.output_css or args.output_root / timeframe.upper() / "currency_strength.parquet"
    output_features = args.output_features or args.output_root / timeframe.upper() / "currency_features.parquet"
    output_css.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    css.to_parquet(output_css)
    features.to_parquet(output_features, index=False)

    print(f"[{timeframe}] Saved CSS to: {output_css}")
    print(f"[{timeframe}] Saved features to: {output_features}")
    print(f"[{timeframe}] Rows CSS={len(css)}, features={len(features)}")


def main() -> None:
    args = parse_args()
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        raise ValueError("Empty --pairs list")

    timeframes = [args.timeframe.strip().upper()]
    if args.timeframes:
        if args.output_css is not None or args.output_features is not None:
            raise ValueError("--output-css/--output-features cannot be used together with --timeframes")
        timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    for tf in timeframes:
        _build_for_timeframe(args=args, pairs=pairs, timeframe=tf)


if __name__ == "__main__":
    main()
