from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qualipilot import DataQualityChecker, QualipilotConfig, register_check
from qualipilot.engines.pandas_engine import PandasEngine
from qualipilot.engines.polars_engine import PolarsEngine
from qualipilot.models.config import (
    CheckConfig,
    ComparisonRule,
    ForeignKeyRule,
    QualityContract,
    SchemaPolicy,
)
from qualipilot.models.results import (
    CardinalityPayload,
    CheckResult,
    DatasetStats,
    DataTypesPayload,
    DuplicatesPayload,
    MissingValuesPayload,
    OutliersPayload,
    QualityReport,
    RangesPayload,
)


def _config(contract: QualityContract) -> QualipilotConfig:
    return QualipilotConfig(
        engine="pandas",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
            rule_packs=[contract],
        ),
    )


def test_contract_counts_comparisons_foreign_keys_and_schema() -> None:
    orders = pd.DataFrame(
        {"id": [1, 2, 3], "end": [2, 1, None], "start": [1, 2, 3]}
    )
    customers = pd.DataFrame({"customer": [1, 4]})
    contract = QualityContract(
        name="orders-v1",
        comparisons=[
            ComparisonRule(
                name="time",
                left_column="end",
                operator="ge",
                right_column="start",
            )
        ],
        foreign_keys=[
            ForeignKeyRule(
                name="customer",
                columns=["id"],
                reference="customers",
                reference_columns=["customer"],
            )
        ],
        schema_policy=SchemaPolicy(
            columns={"id": "int64", "end": "float64", "start": "int64"}
        ),
    )
    result = next(
        item
        for item in DataQualityChecker(
            orders, _config(contract), references={"customers": customers}
        )
        .run(include_llm=False)
        .results
        if item.name == "quality_contract"
    )
    assert result.severity == "error"
    assert result.typed_payload().violation_count == 3


def test_contract_nulls_can_be_violations() -> None:
    contract = QualityContract(
        name="strict",
        comparisons=[
            ComparisonRule(
                name="cmp",
                left_column="a",
                operator="eq",
                right_column="b",
                nulls_pass=False,
            )
        ],
    )
    report = DataQualityChecker(
        pd.DataFrame({"a": [None], "b": [None]}), _config(contract)
    ).run(include_llm=False)
    assert (
        next(
            item for item in report.results if item.name == "quality_contract"
        ).severity
        == "error"
    )


def test_duckdb_foreign_key_stays_in_the_connection() -> None:
    pytest.importorskip("duckdb")
    contract = QualityContract(
        name="keys",
        foreign_keys=[
            ForeignKeyRule(
                name="customer",
                columns=["customer_id"],
                reference="customers",
                reference_columns=["id"],
            )
        ],
    )
    config = _config(contract).model_copy(update={"engine": "duckdb"})
    report = DataQualityChecker(
        pd.DataFrame({"customer_id": [1, 2]}),
        config,
        references={"customers": pd.DataFrame({"id": [1]})},
    ).run(include_llm=False)
    result = next(
        item for item in report.results if item.name == "quality_contract"
    )
    assert result.typed_payload().violation_count == 1


def test_registered_check_and_metrics_are_safe() -> None:
    name = "test_registered_contract_check"
    register_check(name, lambda _: ("warn", {"count": 1}))
    config = QualipilotConfig(checks=CheckConfig(registered_checks=[name]))
    report = DataQualityChecker(pd.DataFrame({"a": [1]}), config).run(
        include_llm=False
    )
    assert name in {result.name for result in report.results}
    text = QualityReport(
        dataset=DatasetStats(
            row_count=0, column_count=0, columns=[], dtypes={}, engine="x"
        ),
        results=[
            CheckResult(name='a"\\\n', severity="ok", duration_seconds=0)
        ],
    ).prometheus_text()
    assert '\\"' in text
    assert "\\\\" in text
    assert "\\n" in text


def test_contract_rejects_ambiguous_operand() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ComparisonRule(name="bad", left_column="a", operator="eq")


def test_opt_in_nested_values_are_deterministic_without_mutation() -> None:
    frame = pd.DataFrame({"id": [1, 2], "nested": [{"b": 1, "a": 2}, None]})
    config = QualipilotConfig(
        engine="pandas", checks=CheckConfig(allow_nested=True)
    )
    report = DataQualityChecker(frame, config).run(include_llm=False)
    assert frame.loc[0, "nested"] == {"b": 1, "a": 2}
    assert report.dataset.dtypes["nested"] == "object"


def test_direct_engines_normalize_nested_maps_without_mutating_pandas() -> (
    None
):
    frame = pd.DataFrame({"nested": [{"z": 1, "a": 2}, None]})
    pandas = PandasEngine(frame, allow_nested=True)
    assert pandas._df["nested"].tolist() == ['{"a":2,"z":1}', None]
    assert frame.loc[0, "nested"] == {"z": 1, "a": 2}
    polars = PolarsEngine.from_any(frame, allow_nested=True)
    assert polars._df["nested"].to_list()[0] == '{"a":2,"z":1}'


def test_builtin_payloads_have_typed_views() -> None:
    report = DataQualityChecker(
        pd.DataFrame({"value": [1.0, None, 1.0, 100.0]}),
        QualipilotConfig(engine="pandas"),
    ).run(include_llm=False)
    payloads = {item.name: item.typed_payload() for item in report.results}
    assert isinstance(payloads["missing_values"], MissingValuesPayload)
    assert isinstance(payloads["duplicates"], DuplicatesPayload)
    assert isinstance(payloads["data_types"], DataTypesPayload)
    assert isinstance(payloads["outliers"], OutliersPayload)
    assert isinstance(payloads["ranges"], RangesPayload)
    assert isinstance(payloads["cardinality"], CardinalityPayload)


def _contract_config_for(
    engine: str, contract: QualityContract
) -> QualipilotConfig:
    return QualipilotConfig(
        engine=engine,  # type: ignore[arg-type]
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
            rule_packs=[contract],
        ),
    )


def _contract_violation_count(
    engine: str,
    frame: pd.DataFrame,
    rule: ComparisonRule | ForeignKeyRule,
    references: dict[str, pd.DataFrame] | None = None,
) -> int:
    contract = QualityContract(
        name="contract",
        comparisons=[rule] if isinstance(rule, ComparisonRule) else [],
        foreign_keys=[rule] if isinstance(rule, ForeignKeyRule) else [],
    )
    report = DataQualityChecker(
        frame,
        _contract_config_for(engine, contract),
        references=references,
    ).run(include_llm=False)
    return (
        next(
            item for item in report.results if item.name == "quality_contract"
        )
        .typed_payload()
        .violation_count
    )


@pytest.mark.parametrize("engine", ["pandas", "polars", "dask", "duckdb"])
@pytest.mark.parametrize(
    ("operator", "column_expected", "literal_expected"),
    [
        ("eq", 1, 1),
        ("ne", 1, 1),
        ("gt", 2, 1),
        ("ge", 1, 0),
        ("lt", 1, 2),
        ("le", 0, 1),
    ],
)
def test_contract_comparison_operators_are_native_and_consistent(
    engine: str,
    operator: str,
    column_expected: int,
    literal_expected: int,
) -> None:
    frame = pd.DataFrame({"left": [1, 2], "right": [1, 3]})
    assert (
        _contract_violation_count(
            engine,
            frame,
            ComparisonRule(
                name="column",
                left_column="left",
                operator=operator,
                right_column="right",  # type: ignore[arg-type]
            ),
        )
        == column_expected
    )
    assert (
        _contract_violation_count(
            engine,
            frame,
            ComparisonRule(
                name="literal",
                left_column="left",
                operator=operator,
                right_value=1,  # type: ignore[arg-type]
            ),
        )
        == literal_expected
    )


@pytest.mark.parametrize("engine", ["pandas", "polars", "dask", "duckdb"])
@pytest.mark.parametrize(("nulls_pass", "expected"), [(True, 1), (False, 3)])
def test_contract_comparisons_treat_null_and_nan_by_explicit_policy(
    engine: str, nulls_pass: bool, expected: int
) -> None:
    frame = pd.DataFrame(
        {
            "left": [1.0, 2.0, None, float("nan")],
            "right": [1.0, 3.0, None, float("nan")],
        }
    )
    assert (
        _contract_violation_count(
            engine,
            frame,
            ComparisonRule(
                name="compare",
                left_column="left",
                operator="eq",
                right_column="right",
                nulls_pass=nulls_pass,
            ),
        )
        == expected
    )


@pytest.mark.parametrize("engine", ["pandas", "polars", "dask", "duckdb"])
@pytest.mark.parametrize(("nulls_pass", "expected"), [(True, 1), (False, 3)])
def test_composite_foreign_keys_handle_nulls_duplicates_and_indices(
    engine: str, nulls_pass: bool, expected: int
) -> None:
    source = pd.DataFrame(
        {"left key": [1, 1, None, 2], 'right"key': [1, None, 2, 2]},
        index=[10, 7, 99, 3],
    )
    reference = pd.DataFrame(
        {"ref.left": [1, 1, None], "ref`right": [1, 1, 2]},
        index=[42, 1, 6],
    )
    assert (
        _contract_violation_count(
            engine,
            source,
            ForeignKeyRule(
                name="foreign",
                columns=["left key", 'right"key'],
                reference="reference",
                reference_columns=["ref.left", "ref`right"],
                nulls_pass=nulls_pass,
            ),
            {"reference": reference},
        )
        == expected
    )


def test_duckdb_relation_references_require_and_use_caller_connection() -> (
    None
):
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    try:
        relation = connection.sql('SELECT 1 AS "x""z" UNION ALL SELECT 2')
        rule = ForeignKeyRule(
            name="quoted",
            columns=['x"z'],
            reference="reference",
            reference_columns=['k"y'],
        )
        contract = QualityContract(name="quoted-pack", foreign_keys=[rule])
        with pytest.raises(ValueError, match="caller-owned duckdb_connection"):
            DataQualityChecker(
                relation,
                _contract_config_for("duckdb", contract),
                references={"reference": pd.DataFrame({'k"y': [1]})},
            )
        checker = DataQualityChecker(
            relation,
            _contract_config_for("duckdb", contract),
            duckdb_connection=connection,
            references={"reference": pd.DataFrame({'k"y': [1]})},
        )
        report = checker.run(include_llm=False)
        assert (
            next(
                item
                for item in report.results
                if item.name == "quality_contract"
            )
            .typed_payload()
            .violation_count
            == 1
        )
        checker.close()
        assert connection.sql("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_contract_config_rejects_ambiguous_or_padded_names() -> None:
    with pytest.raises(ValueError, match="blank or padded"):
        ComparisonRule(
            name=" bad", left_column="a", operator="eq", right_value=1
        )
    with pytest.raises(ValueError, match="duplicates"):
        ForeignKeyRule(
            name="key",
            columns=["a", "a"],
            reference="ref",
            reference_columns=["a", "b"],
        )
    with pytest.raises(ValueError, match="rule names must be unique"):
        QualityContract(
            name="pack",
            comparisons=[
                ComparisonRule(
                    name="same", left_column="a", operator="eq", right_value=1
                )
            ],
            foreign_keys=[
                ForeignKeyRule(
                    name="same",
                    columns=["a"],
                    reference="ref",
                    reference_columns=["a"],
                )
            ],
        )
    with pytest.raises(ValueError, match="reserved"):
        QualityContract(
            name="pack",
            comparisons=[
                ComparisonRule(
                    name="schema",
                    left_column="a",
                    operator="eq",
                    right_value=1,
                )
            ],
        )
    with pytest.raises(ValueError, match="shadow"):
        CheckConfig(registered_checks=["duplicates"])


def test_reference_initialisation_closes_owned_primary_and_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qualipilot.checker as checker_module

    closed: list[str] = []

    class _Engine:
        name = "pandas"

        def close(self) -> None:
            closed.append("closed")

    calls = 0

    def build(_: object, **__: object) -> _Engine:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TypeError("bad reference")
        return _Engine()

    monkeypatch.setattr(checker_module, "build_engine", build)
    with pytest.raises(TypeError, match="bad reference"):
        DataQualityChecker(
            pd.DataFrame({"a": [1]}), references={"reference": object()}
        )
    assert closed == ["closed"]


def test_dask_nested_normalisation_rebuilds_object_type_provenance() -> None:
    dask = pytest.importorskip("dask")
    import dask.dataframe as dd

    from qualipilot.engines.dask_engine import DaskEngine

    frame = pd.DataFrame({"nested": [{"b": 1, "a": 2}, "scalar", None]})
    with dask.config.set({"dataframe.convert-string": False}):
        engine = DaskEngine(
            dd.from_pandas(frame, npartitions=2), allow_nested=True
        )
    assert engine.nested_normalized_columns == ["nested"]
    assert engine._object_families == {"nested": {"string"}}
    assert engine._df.compute()["nested"].tolist() == [
        '{"a":2,"b":1}',
        "scalar",
        None,
    ]


def test_contract_reports_missing_columns_and_references_as_findings() -> None:
    contract = QualityContract(
        name="missing",
        comparisons=[
            ComparisonRule(
                name="comparison",
                left_column="absent",
                operator="eq",
                right_value=1,
            )
        ],
        foreign_keys=[
            ForeignKeyRule(
                name="foreign",
                columns=["present"],
                reference="absent-reference",
                reference_columns=["id"],
            )
        ],
    )
    report = DataQualityChecker(
        pd.DataFrame({"present": [1]}),
        _contract_config_for("pandas", contract),
    ).run(include_llm=False)
    findings = (
        next(
            item for item in report.results if item.name == "quality_contract"
        )
        .typed_payload()
        .findings
    )
    assert findings[0].missing_columns == ["absent"]
    assert findings[1].missing_reference is True


def test_contract_comparison_rejects_nonfinite_literal() -> None:
    with pytest.raises(ValueError, match="finite"):
        ComparisonRule(
            name="finite",
            left_column="a",
            operator="eq",
            right_value=float("nan"),
        )


@pytest.mark.parametrize(
    "engine_type",
    ["pandas", "polars", "dask", "duckdb"],
)
def test_nested_jsonl_is_opt_in_and_preserves_scalars(
    tmp_path: Path, engine_type: str
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":1,"nested":{"b":1,"a":2},"items":[2,1],"plain":"value"}',
                '{"id":2,"nested":null,"items":null,"plain":"other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    engines = {
        "pandas": PandasEngine,
        "polars": PolarsEngine,
        "dask": __import__(
            "qualipilot.engines.dask_engine", fromlist=["DaskEngine"]
        ).DaskEngine,
        "duckdb": __import__(
            "qualipilot.engines.duckdb_engine", fromlist=["DuckDBEngine"]
        ).DuckDBEngine,
    }
    engine_class = engines[engine_type]
    with pytest.raises(ValueError, match="only scalar"):
        engine_class.from_any(path)
    engine = engine_class.from_any(path, allow_nested=True)
    assert engine.row_count() == 2
    assert engine.nested_normalized_columns == ["nested", "items"]
    if engine_type == "pandas":
        assert engine._df["plain"].tolist() == ["value", "other"]
        assert engine._df["nested"].tolist() == ['{"a":2,"b":1}', None]
    elif engine_type == "polars":
        assert engine._df["plain"].to_list() == ["value", "other"]
        assert engine._df["items"].to_list() == ["[2,1]", None]


def test_polars_object_normalisation_preserves_noncontainer_scalars() -> None:
    import polars as pl

    values = [{"b": 1, "a": 2}, "plain", [2, 1], None]
    frame = pl.DataFrame({"nested": pl.Series(values, dtype=pl.Object)})
    engine = PolarsEngine(frame, allow_nested=True)
    assert frame["nested"].to_list() == values
    assert engine.nested_normalized_columns == ["nested"]
    assert engine._df["nested"].to_list() == [
        '{"a":2,"b":1}',
        "plain",
        "[2,1]",
        None,
    ]


def test_nested_jsonl_opt_in_keeps_json_safety_guards(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"nested":{"key":1,"key":2}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object keys"):
        PandasEngine.from_any(path, allow_nested=True)
    path.write_text('{"nested":[NaN]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        PandasEngine.from_any(path, allow_nested=True)


def test_nested_file_checker_report_records_normalisation_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"id":1,"nested":[1,2]}\n', encoding="utf-8")
    report = DataQualityChecker(
        path,
        QualipilotConfig(
            engine="pandas", checks=CheckConfig(allow_nested=True)
        ),
    ).run(include_llm=False)
    assert report.dataset.row_count == 1
    assert report.dataset.nested_normalized_columns == ["nested"]


def test_linkage_payload_variants_are_concrete() -> None:
    from qualipilot.models.results import (
        LinkageResultPayload,
        LinkageSkippedPayload,
    )

    skipped = CheckResult(
        name="linkage",
        severity="ok",
        duration_seconds=0,
        payload={"skipped": True},
    )
    assert isinstance(skipped.typed_payload(), LinkageSkippedPayload)
    result = CheckResult(
        name="linkage",
        severity="ok",
        duration_seconds=0,
        payload={
            "candidate_pairs": 1,
            "matched_pairs": 1,
            "match_threshold_probability": 0.9,
            "clusters": 1,
            "timings_ms": {"fit": 1.0},
            "lambda": 0.1,
            "fit_status": "ok",
            "fit_reason": None,
            "fit_warnings": [],
            "duplicate_clusters": 1,
            "records_in_duplicate_groups": 2,
        },
    )
    typed = result.typed_payload()
    assert isinstance(typed, LinkageResultPayload)
    assert typed.lambda_ == 0.1


def test_failed_and_extension_payload_views() -> None:
    from qualipilot.models.results import FailedPayload, GenericPayload

    failed = CheckResult(
        name="duplicates",
        severity="error",
        status="failed",
        duration_seconds=0,
        payload={},
        error="bad",
    )
    assert isinstance(failed.typed_payload(), FailedPayload)
    extension = CheckResult(
        name="app", severity="ok", duration_seconds=0, payload={"x": 1}
    )
    assert isinstance(extension.typed_payload(), GenericPayload)


def test_public_iceberg_config_rejects_before_engine_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qualipilot.checker as checker_module
    from qualipilot.models.config import IcebergTableConfig

    monkeypatch.setattr(
        checker_module,
        "build_engine",
        lambda *args, **kwargs: pytest.fail("engine built"),
    )
    with pytest.raises(ValueError, match="from_iceberg"):
        DataQualityChecker(
            pd.DataFrame({"a": [1]}),
            QualipilotConfig(
                engine="spark",
                iceberg=IcebergTableConfig(identifier="catalog.db.table"),
            ),
        )
