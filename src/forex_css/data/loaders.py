from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from forex_css.data.schema import CandleSchemaConfig, ensure_candle_schema


def load_candle_file(path: str | Path, schema: CandleSchemaConfig | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    elif suffix == ".csv":
        frame = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix!r}. Use CSV or Parquet.")

    return ensure_candle_schema(frame, schema)


def resolve_pair_file(data_root: str | Path, source: str, pair: str, timeframe: str) -> Path:
    root = Path(data_root)
    pair_clean = pair.upper()
    tf_clean = timeframe.upper()

    parquet_path = root / source / pair_clean / f"{tf_clean}.parquet"
    csv_path = root / source / pair_clean / f"{tf_clean}.csv"

    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        f"Could not find file for pair={pair_clean}, tf={tf_clean} at {parquet_path} or {csv_path}"
    )


def load_pairs_from_data_root(
    data_root: str | Path,
    source: str,
    pairs: Iterable[str],
    timeframe: str,
    schema: CandleSchemaConfig | None = None,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        path = resolve_pair_file(data_root=data_root, source=source, pair=pair, timeframe=timeframe)
        output[pair.upper()] = load_candle_file(path, schema=schema)
    return output
