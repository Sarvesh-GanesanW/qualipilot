"""Read-only Iceberg metadata checks through the caller's Spark session."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.config import SchemaPolicy
from qualipilot.models.results import Severity


def _quoted_table(identifier: str, suffix: str = "") -> str:
    return ".".join(f"`{part}`" for part in identifier.split(".")) + suffix


class IcebergTableCheck(Check):
    """Inspect one captured Iceberg snapshot without mutating its table."""

    name = "iceberg_table"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        config = ctx.iceberg
        if config is None:
            return "ok", {"identifier": "", "metadata_complete": False}
        if ctx.engine.name != "spark" or ctx.iceberg_table is None:
            raise ValueError("iceberg metadata checks require from_iceberg")
        captured = ctx.iceberg_snapshot_id
        if captured is None:
            raise ValueError("Iceberg check requires a captured snapshot")
        engine: Any = ctx.engine
        spark = engine._df.sparkSession
        table: Any = ctx.iceberg_table
        files = _snapshot_metadata(spark, config.identifier, "files", captured)
        manifests = _snapshot_metadata(
            spark, config.identifier, "manifests", captured
        )
        file_counts = files.selectExpr(
            "COUNT(*) AS total_files",
            "SUM(CASE WHEN content = 0 THEN 1 ELSE 0 END) AS data_files",
            "SUM(CASE WHEN content <> 0 THEN 1 ELSE 0 END) AS delete_files",
        ).first()
        data_files = int(file_counts["data_files"] or 0)
        delete_files = int(file_counts["delete_files"] or 0)
        total_files = int(file_counts["total_files"] or 0)
        active_spec_ids = sorted(
            int(row["spec_id"])
            for row in files.select("spec_id").distinct().collect()
        )
        manifest_count = int(manifests.count())
        captured_snapshot = table.snapshot(captured)
        if captured_snapshot is None:
            raise ValueError(f"captured snapshot {captured} is unavailable")
        committed_at = datetime.fromtimestamp(
            int(captured_snapshot.timestampMillis()) / 1000, tz=UTC
        )
        age_hours = (datetime.now(UTC) - committed_at).total_seconds() / 3600
        table.refresh()
        current_snapshot = table.currentSnapshot()
        current_id = (
            None
            if current_snapshot is None
            else int(current_snapshot.snapshotId())
        )
        schema = ctx.iceberg_schema or {}
        schema_id = int(captured_snapshot.schemaId())
        spec_id = int(table.spec().specId())
        dtype_mismatches = [
            {
                "column": column,
                "expected": expected,
                "actual": schema.get(column),
            }
            for column, expected in config.expected_dtypes.items()
            if schema.get(column) != expected
        ]
        schema_finding = _schema_finding(schema, config.schema_policy)
        ancestor_ok = _is_ancestor(
            table, captured, config.expected_ancestor_snapshot_id
        )
        probe = _probe_files(spark, files, config.file_probe_max_files)
        violations = sum(
            (
                int(
                    config.max_snapshot_age_hours is not None
                    and age_hours > config.max_snapshot_age_hours
                ),
                int(
                    config.max_data_files is not None
                    and data_files > config.max_data_files
                ),
                int(
                    config.max_delete_files is not None
                    and delete_files > config.max_delete_files
                ),
                int(
                    config.max_active_partition_specs is not None
                    and len(active_spec_ids)
                    > config.max_active_partition_specs
                ),
                int(
                    config.max_manifest_count is not None
                    and manifest_count > config.max_manifest_count
                ),
                int(
                    config.expected_snapshot_id is not None
                    and captured != config.expected_snapshot_id
                ),
                int(
                    config.require_current_snapshot and current_id != captured
                ),
                int(
                    config.expected_schema_id is not None
                    and schema_id != config.expected_schema_id
                ),
                int(
                    config.expected_current_spec_id is not None
                    and spec_id != config.expected_current_spec_id
                ),
                int(
                    bool(config.allowed_spec_ids)
                    and any(
                        value not in config.allowed_spec_ids
                        for value in active_spec_ids
                    )
                ),
                int(
                    config.expected_ancestor_snapshot_id is not None
                    and not ancestor_ok
                ),
                len(dtype_mismatches),
                schema_finding["violation_count"],
                probe["missing"] + probe["size_mismatches"] + probe["errors"],
                int(
                    config.file_probe_require_complete
                    and not probe["complete"]
                ),
            )
        )
        severity: Severity = "error" if violations else "ok"
        if (
            severity == "ok"
            and probe["performed"]
            and not probe["complete"]
            and not config.file_probe_require_complete
        ):
            severity = "warn"
        return severity, {
            "identifier": config.identifier,
            "metadata_complete": True,
            "captured_snapshot_id": captured,
            "current_snapshot_id": current_id,
            "history_count": int(
                spark.sql(
                    "SELECT COUNT(*) AS n FROM "
                    f"{_quoted_table(config.identifier)}.history"
                ).first()["n"]
            ),
            "snapshot_count": int(
                spark.sql(
                    "SELECT COUNT(*) AS n FROM "
                    f"{_quoted_table(config.identifier)}.snapshots"
                ).first()["n"]
            ),
            "snapshot_age_hours": age_hours,
            "data_files": data_files,
            "delete_files": delete_files,
            "total_files": total_files,
            "active_partition_spec_count": len(active_spec_ids),
            "active_spec_ids": active_spec_ids,
            "manifest_count": manifest_count,
            "captured_schema_id": schema_id,
            "current_spec_id": spec_id,
            "schema": schema,
            "dtype_mismatches": dtype_mismatches,
            "schema_policy": schema_finding,
            "ancestor_ok": ancestor_ok,
            "file_probe": probe,
            "file_probe_note": (
                "existence and size only; this is not Parquet or "
                "checksum integrity"
            ),
        }


def _snapshot_metadata(
    spark: Any, identifier: str, name: str, snapshot_id: int
) -> Any:
    return (
        spark.read.format("iceberg")
        .option("snapshot-id", snapshot_id)
        .load(f"{identifier}.{name}")
    )


def _schema_finding(
    actual: dict[str, str], policy: SchemaPolicy | None
) -> dict[str, Any]:
    if policy is None:
        return {
            "violation_count": 0,
            "additions": [],
            "removals": [],
            "type_changes": [],
        }
    expected = policy.columns
    additions = sorted(set(actual) - set(expected))
    removals = sorted(set(expected) - set(actual))
    changes = [
        {
            "column": column,
            "expected": expected[column],
            "actual": actual[column],
        }
        for column in sorted(set(actual) & set(expected))
        if actual[column].casefold() != expected[column].casefold()
    ]
    violations = (
        (0 if policy.allow_additions else len(additions))
        + (0 if policy.allow_removals else len(removals))
        + (0 if policy.allow_type_changes else len(changes))
    )
    return {
        "violation_count": violations,
        "additions": additions,
        "removals": removals,
        "type_changes": changes,
    }


def _is_ancestor(
    table: Any, captured: int, ancestor: int | None
) -> bool | None:
    if ancestor is None:
        return None
    current: int | None = captured
    seen: set[int] = set()
    while current is not None and current not in seen:
        if current == ancestor:
            return True
        seen.add(current)
        snapshot = table.snapshot(current)
        if snapshot is None:
            return False
        parent = snapshot.parentId()
        current = None if parent is None else int(parent)
    return False


def _probe_files(spark: Any, files: Any, maximum: int) -> dict[str, Any]:
    if not maximum:
        return {
            "performed": False,
            "checked": 0,
            "missing": 0,
            "size_mismatches": 0,
            "errors": 0,
            "complete": False,
        }
    rows = (
        files.select("file_path", "file_size_in_bytes")
        .limit(maximum + 1)
        .collect()
    )
    checked = min(len(rows), maximum)
    missing = size_mismatches = errors = 0
    jvm = spark._jvm
    hadoop = spark._jsc.hadoopConfiguration()
    for row in rows[:maximum]:
        try:
            path = jvm.org.apache.hadoop.fs.Path(row["file_path"])
            fs = path.getFileSystem(hadoop)
            if not fs.exists(path):
                missing += 1
            elif int(fs.getFileStatus(path).getLen()) != int(
                row["file_size_in_bytes"]
            ):
                size_mismatches += 1
        except Exception:
            errors += 1
    return {
        "performed": True,
        "checked": checked,
        "missing": missing,
        "size_mismatches": size_mismatches,
        "errors": errors,
        "complete": len(rows) <= maximum,
    }
