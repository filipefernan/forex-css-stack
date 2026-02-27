from __future__ import annotations

import argparse
from pathlib import Path

from forex_css.constants import DEFAULT_SYMBOLS_TO_WEIGH
from forex_css.dataset.builder import CostConfig, DecisionConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build modeling dataset with multi-timeframe features + basket targets.")
    parser.add_argument("--feature-root", type=Path, default=Path("data/features"))
    parser.add_argument("--timeframes", type=str, default="H1,H4,D1", help="Comma-separated feature timeframes.")
    parser.add_argument("--decision-mode", type=str, default="daily", choices=["daily", "hourly"])
    parser.add_argument("--decision-time", type=str, default="21:00", help="HH:MM in --timezone.")
    parser.add_argument("--timezone", type=str, default="America/Bahia")
    parser.add_argument("--hourly-step", type=int, default=1, help="Used when --decision-mode=hourly.")
    parser.add_argument("--decision-anchor-timeframe", type=str, default="H1")

    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", type=str, default="twelvedata")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_SYMBOLS_TO_WEIGH))
    parser.add_argument("--target-timeframe", type=str, default="H1")
    parser.add_argument("--horizons-hours", type=str, default="1,4,8,24")
    parser.add_argument("--spread-bps", type=float, default=1.5)
    parser.add_argument("--slippage-bps", type=float, default=0.5)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/daily_21brt.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeframes = [tf.strip().upper() for tf in args.timeframes.split(",") if tf.strip()]
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    horizons = [int(x.strip()) for x in args.horizons_hours.split(",") if x.strip()]
    if not timeframes or not pairs or not horizons:
        raise ValueError("Provide non-empty --timeframes, --pairs and --horizons-hours")

    decision_cfg = DecisionConfig(
        mode=args.decision_mode,
        timezone=args.timezone,
        decision_time=args.decision_time,
        hourly_step=args.hourly_step,
    )
    cost_cfg = CostConfig(spread_bps=args.spread_bps, slippage_bps=args.slippage_bps)

    dataset = build_dataset(
        feature_root=args.feature_root,
        timeframes=timeframes,
        decision_config=decision_cfg,
        data_root=args.data_root,
        source=args.source,
        pairs=pairs,
        target_timeframe=args.target_timeframe,
        horizons_hours=horizons,
        cost_config=cost_cfg,
        decision_anchor_timeframe=args.decision_anchor_timeframe,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)
    print(f"Saved dataset: {args.output}")
    print(f"Rows={len(dataset)} | timestamps={dataset['timestamp'].nunique()} | currencies={dataset['currency'].nunique()}")
    print(f"Targets: {[c for c in dataset.columns if c.startswith('target_ret_')]}")


if __name__ == "__main__":
    main()
