"""Real local-catalog coverage for the optional Iceberg runtime."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qualipilot import DataQualityChecker
from qualipilot.models.config import (
    CheckConfig,
    IcebergTableConfig,
    QualipilotConfig,
)

pytestmark = pytest.mark.integration
_RUNTIME = "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.1"


@pytest.fixture
def iceberg_spark(  # type: ignore[no-untyped-def]
    tmp_path: Path, record_testsuite_property: object
):
    pyspark = pytest.importorskip("pyspark")
    if not pyspark.__version__.startswith("3.5."):
        pytest.skip("Iceberg integration is pinned to Spark 3.5")
    from pyspark.sql import SparkSession

    from qualipilot import __version__

    record_testsuite_property("qualipilot_package_version", __version__)
    record_testsuite_property("pyspark_version", pyspark.__version__)
    record_testsuite_property("iceberg_runtime", _RUNTIME)
    record_testsuite_property(
        "source_revision", os.environ.get("GITHUB_SHA", "working-tree")
    )
    session = (
        SparkSession.builder.master("local[1]")
        .appName("qualipilot-iceberg-integration")
        .config(
            "spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog"
        )
        .config("spark.sql.catalog.local.type", "hadoop")
        .config(
            "spark.sql.catalog.local.warehouse", str(tmp_path / "warehouse")
        )
        .config("spark.jars.packages", _RUNTIME)
        .getOrCreate()
    )
    yield session
    session.stop()


def _config(**options: object) -> QualipilotConfig:
    return QualipilotConfig(
        engine="spark",
        iceberg=IcebergTableConfig(identifier="local.db.events", **options),
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
        ),
    )


def test_from_iceberg_binds_rows_and_metadata_to_one_snapshot(
    iceberg_spark: object,
) -> None:
    spark = iceberg_spark
    spark.sql(
        "CREATE TABLE local.db.events (id BIGINT, name STRING) USING iceberg"
    )
    spark.sql("INSERT INTO local.db.events VALUES (1, 'first')")
    first = int(
        spark.sql("SELECT snapshot_id FROM local.db.events.snapshots").first()[
            0
        ]
    )
    spark.sql("INSERT INTO local.db.events VALUES (2, 'second')")
    config = _config(
        max_manifest_count=0,
        expected_schema_id=0,
        expected_current_spec_id=0,
        expected_ancestor_snapshot_id=first,
        file_probe_max_files=10,
        file_probe_require_complete=True,
    )
    checker = DataQualityChecker.from_iceberg(
        "local.db.events", spark_session=spark, config=config
    )
    report = checker.run(include_llm=False)
    iceberg_result = next(
        result for result in report.results if result.name == "iceberg_table"
    )
    payload = iceberg_result.payload
    assert iceberg_result.typed_payload().file_probe is not None
    assert report.dataset.row_count == 2
    assert report.dataset.source_version == str(
        payload["captured_snapshot_id"]
    )
    assert payload["current_snapshot_id"] == payload["captured_snapshot_id"]
    assert payload["ancestor_ok"] is True
    assert payload["file_probe"]["complete"] is True
    assert payload["file_probe"]["checked"] >= 1
    assert payload["manifest_count"] > 0
    assert (
        next(
            result
            for result in report.results
            if result.name == "iceberg_table"
        ).severity
        == "error"
    )


def test_from_iceberg_validates_snapshot_and_schema_contract(
    iceberg_spark: object,
) -> None:
    spark = iceberg_spark
    spark.sql("CREATE TABLE local.db.events (id BIGINT) USING iceberg")
    spark.sql("INSERT INTO local.db.events VALUES (1)")
    snapshot = int(
        spark.sql("SELECT snapshot_id FROM local.db.events.snapshots").first()[
            0
        ]
    )
    spark.sql("ALTER TABLE local.db.events ADD COLUMN name STRING")
    spark.sql("INSERT INTO local.db.events VALUES (2, 'second')")
    report = DataQualityChecker.from_iceberg(
        "local.db.events",
        spark_session=spark,
        snapshot_id=snapshot,
        config=_config(expected_schema_id=999),
    ).run(include_llm=False)
    payload = next(
        result for result in report.results if result.name == "iceberg_table"
    ).payload
    assert payload["captured_snapshot_id"] == snapshot
    assert payload["captured_schema_id"] != 999
    assert payload["current_snapshot_id"] != snapshot


def test_iceberg_partition_spec_and_snapshot_lineage_policies(
    iceberg_spark: object,
) -> None:
    spark = iceberg_spark
    spark.sql(
        "CREATE TABLE local.db.events (id BIGINT) USING iceberg "
        "PARTITIONED BY (bucket(2, id))"
    )
    spark.sql("INSERT INTO local.db.events VALUES (1)")
    first = int(
        spark.sql("SELECT snapshot_id FROM local.db.events.snapshots").first()[
            0
        ]
    )
    spark.sql("INSERT INTO local.db.events VALUES (2)")
    report = DataQualityChecker.from_iceberg(
        "local.db.events",
        spark_session=spark,
        config=_config(
            allowed_spec_ids=[999], expected_ancestor_snapshot_id=first
        ),
    ).run(include_llm=False)
    payload = next(
        result for result in report.results if result.name == "iceberg_table"
    ).payload
    assert payload["active_spec_ids"] == [0]
    assert payload["ancestor_ok"] is True
    assert (
        next(
            result
            for result in report.results
            if result.name == "iceberg_table"
        ).severity
        == "error"
    )


def test_iceberg_native_spec_evolution_and_rollback(
    iceberg_spark: object,
) -> None:
    spark = iceberg_spark
    spark.sql("CREATE TABLE local.db.events (id BIGINT) USING iceberg")
    spark.sql("INSERT INTO local.db.events VALUES (1)")
    first = int(
        spark.sql("SELECT snapshot_id FROM local.db.events.snapshots").first()[
            0
        ]
    )
    table = spark._jvm.org.apache.iceberg.spark.Spark3Util.loadIcebergTable(
        spark._jsparkSession, "local.db.events"
    )
    table.updateSpec().addField("id").commit()
    table.refresh()
    assert int(table.spec().specId()) == 1
    spark.catalog.refreshTable("local.db.events")
    spark.sql("INSERT INTO local.db.events VALUES (2)")
    evolved = int(
        spark.sql(
            "SELECT snapshot_id FROM local.db.events.snapshots "
            "ORDER BY committed_at DESC"
        ).first()[0]
    )
    table.refresh()
    table.manageSnapshots().rollbackTo(first).commit()
    table.refresh()
    spark.catalog.refreshTable("local.db.events")
    report = DataQualityChecker.from_iceberg(
        "local.db.events",
        spark_session=spark,
        config=_config(
            expected_ancestor_snapshot_id=evolved,
            require_current_snapshot=True,
        ),
    ).run(include_llm=False)
    payload = next(
        result for result in report.results if result.name == "iceberg_table"
    ).payload
    assert payload["current_snapshot_id"] != evolved
    assert payload["ancestor_ok"] is False


def test_iceberg_bounded_probe_warns_or_fails_when_incomplete(
    iceberg_spark: object,
) -> None:
    spark = iceberg_spark
    spark.sql("CREATE TABLE local.db.events (id BIGINT) USING iceberg")
    spark.sql("INSERT INTO local.db.events VALUES (1)")
    spark.sql("INSERT INTO local.db.events VALUES (2)")
    for strict, severity in [(False, "warn"), (True, "error")]:
        report = DataQualityChecker.from_iceberg(
            "local.db.events",
            spark_session=spark,
            config=_config(
                file_probe_max_files=1, file_probe_require_complete=strict
            ),
        ).run(include_llm=False)
        result = next(r for r in report.results if r.name == "iceberg_table")
        probe = result.payload["file_probe"]
        assert probe["checked"] == 1
        assert probe["complete"] is False
        assert result.severity == severity


def test_iceberg_factory_rejects_never_written_table(
    iceberg_spark: object,
) -> None:
    spark = iceberg_spark
    spark.sql("CREATE TABLE local.db.events (id BIGINT) USING iceberg")
    with pytest.raises(ValueError, match="no snapshot"):
        DataQualityChecker.from_iceberg(
            "local.db.events", spark_session=spark, config=_config()
        )


@pytest.mark.parametrize("mutation", ["delete", "truncate"])
def test_iceberg_probe_detects_local_file_mutation(
    iceberg_spark: object, tmp_path: Path, mutation: str
) -> None:
    from qualipilot.checks.base import CheckContext
    from qualipilot.checks.iceberg import IcebergTableCheck

    spark = iceberg_spark
    spark.sql("CREATE TABLE local.db.events (id BIGINT) USING iceberg")
    spark.sql("INSERT INTO local.db.events VALUES (1)")
    checker = DataQualityChecker.from_iceberg(
        "local.db.events",
        spark_session=spark,
        config=_config(file_probe_max_files=10),
    )
    path = Path(
        spark.read.format("iceberg")
        .option("snapshot-id", checker._iceberg_snapshot_id)
        .load("local.db.events.files")
        .select("file_path")
        .first()[0]
    )
    assert path.resolve().is_relative_to(tmp_path)
    if mutation == "delete":
        path.unlink()
    else:
        path.write_bytes(b"x")
    ctx = CheckContext(
        engine=checker.engine,
        config=checker.config.checks,
        iceberg=checker.config.iceberg,
        iceberg_snapshot_id=checker._iceberg_snapshot_id,
        iceberg_table=checker._iceberg_table,
        iceberg_schema=checker._iceberg_schema,
    )
    result = IcebergTableCheck().run(ctx)
    probe = result.payload["file_probe"]
    assert result.severity == "error"
    assert probe["checked"] <= 10
    assert probe["missing" if mutation == "delete" else "size_mismatches"] == 1
    assert str(path) not in str(result.payload)
