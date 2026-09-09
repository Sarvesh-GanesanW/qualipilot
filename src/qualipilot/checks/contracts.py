"""Portable quality-contract rules with no expression evaluation."""

from __future__ import annotations

from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.config import ComparisonRule, ForeignKeyRule
from qualipilot.models.results import Severity


class QualityContractCheck(Check):
    """Evaluate declared comparisons, reference keys, and schema baselines."""

    name = "quality_contract"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for pack in ctx.config.rule_packs:
            findings.extend(
                _schema_findings(ctx, pack.name, pack.schema_policy)
            )
            findings.extend(
                _comparison_finding(ctx, pack.name, rule)
                for rule in pack.comparisons
            )
            findings.extend(
                _foreign_key_finding(ctx, pack.name, rule)
                for rule in pack.foreign_keys
            )
        violations = sum(int(item["violation_count"]) for item in findings)
        return ("error" if violations else "ok"), {
            "schema_version": "1.0",
            "rule_packs": [pack.name for pack in ctx.config.rule_packs],
            "findings": findings,
            "violation_count": violations,
        }


def _schema_findings(
    ctx: CheckContext, pack: str, policy: Any
) -> list[dict[str, Any]]:
    if policy is None:
        return []
    actual = ctx.dtypes if ctx.dtypes is not None else ctx.engine.dtypes()
    expected = policy.columns
    additions = sorted(set(actual) - set(expected))
    removals = sorted(set(expected) - set(actual))
    changed = sorted(
        column
        for column in set(actual) & set(expected)
        if actual[column].casefold() != expected[column].casefold()
    )
    count = (
        (0 if policy.allow_additions else len(additions))
        + (0 if policy.allow_removals else len(removals))
        + (0 if policy.allow_type_changes else len(changed))
    )
    return [
        {
            "pack": pack,
            "rule": "schema",
            "kind": "schema",
            "violation_count": count,
            "additions": additions,
            "removals": removals,
            "type_changes": [
                {
                    "column": column,
                    "expected": expected[column],
                    "actual": actual[column],
                }
                for column in changed
            ],
        }
    ]


def _comparison_finding(
    ctx: CheckContext, pack: str, rule: ComparisonRule
) -> dict[str, Any]:
    columns = set(ctx.columns or ctx.engine.columns())
    required = {rule.left_column, rule.right_column}
    missing = sorted(
        column for column in required if column and column not in columns
    )
    count = _comparison_count(ctx.engine, rule) if not missing else 1
    return {
        "pack": pack,
        "rule": rule.name,
        "kind": "comparison",
        "violation_count": count,
        "missing_columns": missing,
        "nulls_pass": rule.nulls_pass,
    }


def _foreign_key_finding(
    ctx: CheckContext, pack: str, rule: ForeignKeyRule
) -> dict[str, Any]:
    reference = (ctx.references or {}).get(rule.reference)
    count = (
        1
        if reference is None
        else _foreign_key_count(ctx.engine, reference, rule)
    )
    return {
        "pack": pack,
        "rule": rule.name,
        "kind": "foreign_key",
        "reference": rule.reference,
        "violation_count": count,
        "missing_reference": reference is None,
        "nulls_pass": rule.nulls_pass,
    }


def _comparison_count(  # noqa: C901, PLR0915
    engine: Any, rule: ComparisonRule
) -> int:
    left: Any
    right: Any
    comparison: Any
    invalid: Any
    if engine.name == "spark":
        from pyspark.sql import functions as f

        left = _spark_column(rule.left_column)
        right = (
            _spark_column(rule.right_column)
            if rule.right_column
            else f.lit(rule.right_value)
        )
        comparison = {
            "eq": left == right,
            "ne": left != right,
            "gt": left > right,
            "ge": left >= right,
            "lt": left < right,
            "le": left <= right,
        }[rule.operator]
        invalid = ~f.coalesce(comparison, f.lit(False))
        if rule.nulls_pass:
            invalid = invalid & _spark_present(engine, rule.left_column, left)
            if rule.right_column:
                invalid = invalid & _spark_present(
                    engine, rule.right_column, right
                )
        return int(engine._df.filter(invalid).count())
    if engine.name == "polars":
        import polars as pl

        left = pl.col(rule.left_column)
        right = (
            pl.col(rule.right_column)
            if rule.right_column
            else pl.lit(rule.right_value)
        )
        comparison = {
            "eq": left == right,
            "ne": left != right,
            "gt": left > right,
            "ge": left >= right,
            "lt": left < right,
            "le": left <= right,
        }[rule.operator]
        invalid = (~comparison).fill_null(not rule.nulls_pass)
        if rule.nulls_pass:
            invalid &= _polars_present(engine, rule.left_column, left)
            if rule.right_column:
                invalid &= _polars_present(engine, rule.right_column, right)
        return int(engine._df.select(invalid.sum()).item())
    if engine.name == "dask":
        left = engine._df[rule.left_column]
        right = (
            engine._df[rule.right_column]
            if rule.right_column
            else rule.right_value
        )
        comparison = {
            "eq": left == right,
            "ne": left != right,
            "gt": left > right,
            "ge": left >= right,
            "lt": left < right,
            "le": left <= right,
        }[rule.operator]
        invalid = ~comparison.fillna(False)
        if rule.nulls_pass:
            left_present = ~left.isna()
            right_present = ~right.isna() if rule.right_column else True
            invalid &= left_present & right_present
        return int(invalid.sum().compute())
    if engine.name == "duckdb":
        from qualipilot.engines._duckdb_sql import quote_identifier

        left = quote_identifier(rule.left_column)
        right = (
            quote_identifier(rule.right_column) if rule.right_column else "?"
        )
        operator = {
            "eq": "=",
            "ne": "!=",
            "gt": ">",
            "ge": ">=",
            "lt": "<",
            "le": "<=",
        }[rule.operator]
        expression = f"{left} {operator} {right}"
        left_missing = engine._missing_sql(rule.left_column)
        right_missing = (
            engine._missing_sql(rule.right_column)
            if rule.right_column
            else f"{right} IS NULL"
        )
        nulls = f"{left_missing} OR {right_missing}"
        where = f"NOT COALESCE({expression}, FALSE)"
        if rule.nulls_pass:
            where = f"({where}) AND NOT ({nulls})"
        params = (
            []
            if rule.right_column
            else [rule.right_value] * (2 if rule.nulls_pass else 1)
        )
        return int(
            engine._scalar(
                "SELECT COUNT(*) FROM "
                f"{quote_identifier(engine._view)} WHERE {where}",
                params,
            )
        )
    left = engine._df[rule.left_column]
    right = (
        engine._df[rule.right_column]
        if rule.right_column
        else rule.right_value
    )
    comparison = {
        "eq": left == right,
        "ne": left != right,
        "gt": left > right,
        "ge": left >= right,
        "lt": left < right,
        "le": left <= right,
    }[rule.operator]
    invalid = ~comparison.fillna(False)
    if rule.nulls_pass:
        right_missing = getattr(right, "notna", lambda: True)()
        invalid &= left.notna() & right_missing
    return int(invalid.sum())


def _spark_present(engine: Any, name: str, column: Any) -> Any:
    from pyspark.sql import functions as f

    present = column.isNotNull()
    return (
        ~f.isnan(column) & present
        if engine.dtype_family(name) == "float"
        else present
    )


def _polars_present(engine: Any, name: str, column: Any) -> Any:
    present = column.is_not_null()
    return (
        column.fill_nan(None).is_not_null()
        if engine.dtype_family(name) == "float"
        else present
    )


def _spark_column(name: str) -> Any:
    """Return a Spark column while treating dots and backticks as data."""
    from pyspark.sql import functions as f

    return f.col(f"`{name.replace('`', '``')}`")


def _foreign_key_count(  # noqa: C901, PLR0912, PLR0915
    engine: Any, reference: Any, rule: ForeignKeyRule
) -> int:
    non_null: Any
    if engine.name != reference.name:
        raise ValueError("reference frame must use the same resolved engine")
    if engine.name == "spark":
        from pyspark.sql import functions as f

        source_names = [
            f"_qp_contract_source_{index}"
            for index in range(len(rule.columns))
        ]
        target_names = [
            f"_qp_contract_reference_{index}"
            for index in range(len(rule.columns))
        ]
        source = engine._df.select(
            *[
                _spark_column(column).alias(alias)
                for column, alias in zip(
                    rule.columns, source_names, strict=True
                )
            ]
        ).alias("source")
        reference_present = _spark_present(
            reference,
            rule.reference_columns[0],
            _spark_column(rule.reference_columns[0]),
        )
        for column in rule.reference_columns[1:]:
            reference_present = reference_present & _spark_present(
                reference, column, _spark_column(column)
            )
        target = (
            reference._df.filter(reference_present)
            .select(
                *[
                    _spark_column(column).alias(alias)
                    for column, alias in zip(
                        rule.reference_columns, target_names, strict=True
                    )
                ]
            )
            .dropDuplicates()
            .alias("target")
        )
        join = None
        for source_name, target_name in zip(
            source_names, target_names, strict=True
        ):
            part = f.col(f"source.{source_name}") == f.col(
                f"target.{target_name}"
            )
            join = part if join is None else join & part
        missing = source.join(target, join, "left_anti")
        if rule.nulls_pass:
            non_null = _spark_present(
                engine, rule.columns[0], f.col(source_names[0])
            )
            for column, name in zip(
                rule.columns[1:], source_names[1:], strict=True
            ):
                non_null = non_null & _spark_present(
                    engine, column, f.col(name)
                )
            missing = missing.filter(non_null)
        return int(missing.count())
    if engine.name == "polars":
        import polars as pl

        right = reference._df.select(
            *[
                pl.col(reference_column)
                .cast(engine._df.schema[column])
                .alias(column)
                for column, reference_column in zip(
                    rule.columns, rule.reference_columns, strict=True
                )
            ]
        ).unique()
        missing = engine._df.join(right, on=rule.columns, how="anti")
        if rule.nulls_pass:
            missing = missing.filter(
                pl.all_horizontal(
                    [pl.col(c).is_not_null() for c in rule.columns]
                )
            )
        return int(missing.height)
    if engine.name == "dask":
        target = reference._df[rule.reference_columns].rename(
            columns=dict(
                zip(rule.reference_columns, rule.columns, strict=True)
            )
        )
        for column in rule.columns:
            target[column] = target[column].astype(engine._df.dtypes[column])
        target = target.drop_duplicates()
        source = engine._df.assign(_qp_row=engine._df.index)
        joined = source.merge(
            target, on=rule.columns, how="left", indicator=True
        )
        missing = joined["_merge"] == "left_only"
        missing |= joined[rule.columns].isna().any(axis=1)
        if rule.nulls_pass:
            missing &= ~joined[rule.columns].isna().any(axis=1)
        return int(missing.sum().compute())
    if engine.name == "pandas":
        target = (
            reference._df[rule.reference_columns]
            .rename(
                columns=dict(
                    zip(rule.reference_columns, rule.columns, strict=True)
                )
            )
            .drop_duplicates()
        )
        source = engine._df.assign(_qp_row=range(len(engine._df)))
        joined = source.merge(
            target, on=rule.columns, how="left", indicator=True
        )
        missing = joined["_merge"] == "left_only"
        missing |= joined[rule.columns].isna().any(axis=1)
        if rule.nulls_pass:
            missing &= ~joined[rule.columns].isna().any(axis=1)
        return int(missing.sum())
    if engine.name == "duckdb":
        from qualipilot.engines._duckdb_sql import quote_identifier

        source = (
            engine._relation
            if engine._relation is not None
            else engine._con.sql(
                f"SELECT * FROM {quote_identifier(engine._view)}"
            )
        ).set_alias("source")
        target_relation = reference._relation
        if target_relation is None:
            if reference._con is None:
                raise ValueError("DuckDB reference has no queryable relation")
            target_relation = reference._con.sql(
                f"SELECT * FROM {quote_identifier(reference._view)}"
            )
        target = target_relation.set_alias("reference")
        condition = " AND ".join(
            "source."
            f"{quote_identifier(left)} = reference.{quote_identifier(right)}"
            for left, right in zip(
                rule.columns, rule.reference_columns, strict=True
            )
        )
        missing = source.join(target, condition, "anti")
        if rule.nulls_pass:
            non_null = " AND ".join(
                f"{quote_identifier(column)} IS NOT NULL"
                for column in rule.columns
            )
            missing = missing.filter(non_null)
        return int(missing.aggregate("count(*) AS count").fetchone()[0])
    raise ValueError(f"unsupported engine: {engine.name}")
