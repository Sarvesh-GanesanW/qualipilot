"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def tidy_pandas() -> pd.DataFrame:
    """Clean dataframe, no quality issues."""
    return pd.DataFrame(
        {
            "id": range(100),
            "score": [i * 0.5 for i in range(100)],
            "label": ["a", "b"] * 50,
        }
    )


@pytest.fixture
def dirty_pandas() -> pd.DataFrame:
    """Dataframe with nulls, duplicates, outliers, range violations."""
    data = {
        "id": list(range(1, 11)),
        # null in id#5, outlier on id#10
        "amount": [
            10.0,
            12.0,
            11.5,
            None,
            10_000.0,
            9.8,
            10.1,
            10.2,
            9.9,
            10.0,
        ],
        "category": ["x", "x", "y", "y", "z", "z", "z", "z", "z", "z"],
        # duplicate row: id 6 and 7 share everything
    }
    df = pd.DataFrame(data)
    return pd.concat([df, df.iloc[[5]]], ignore_index=True)


@pytest.fixture
def dirty_polars(dirty_pandas: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(dirty_pandas)


@pytest.fixture
def stale_timestamps_pandas() -> pd.DataFrame:
    now = datetime.now(UTC)
    return pd.DataFrame(
        {
            "event_ts": [
                now - timedelta(hours=48),
                now - timedelta(hours=72),
            ],
            "value": [1, 2],
        }
    )


@pytest.fixture
def tmp_csv(tmp_path: Path, dirty_pandas: pd.DataFrame) -> Path:
    path = tmp_path / "dirty.csv"
    dirty_pandas.to_csv(path, index=False)
    return path
