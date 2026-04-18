"""Hypothesis-based property tests.

Goal: no matter the shape of the input, checks return a valid
``CheckResult`` and never raise.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from datapilot.checks import CheckContext, MissingValuesCheck
from datapilot.engines import PolarsEngine
from datapilot.models.config import CheckConfig

_INT_OR_NONE = st.one_of(
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.none(),
)
_FLOAT_OR_NONE = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False),
    st.none(),
)


@settings(max_examples=25, deadline=None)
@given(
    ints=st.lists(_INT_OR_NONE, min_size=0, max_size=50),
    floats=st.lists(_FLOAT_OR_NONE, min_size=0, max_size=50),
)
def test_missing_check_tolerates_random_shapes(
    ints: list[int | None], floats: list[float | None]
) -> None:
    n = min(len(ints), len(floats))
    df = pd.DataFrame({"a": ints[:n], "b": floats[:n]})
    ctx = CheckContext(
        engine=PolarsEngine.from_any(df), config=CheckConfig()
    )
    result = MissingValuesCheck().run(ctx)
    assert result.error is None
    assert result.severity in {"ok", "warn", "error"}
