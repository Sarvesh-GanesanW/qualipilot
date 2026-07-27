"""Deterministic record consolidation after deduplication."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby
from typing import Any, Literal

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from qualipilot.engines.base import reject_nested_columns

MergeStrategy = Literal[
    "survivor",
    "first_non_null",
    "most_frequent",
    "latest",
]
AuditAction = Literal["retained", "filled", "replaced"]


class SurvivorSortKey(BaseModel):
    """One ordered key used to choose a cluster survivor."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    column: str = Field(min_length=1)
    descending: bool = False

    @field_validator("column")
    @classmethod
    def _strip_column(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sort key column must not be blank")
        return value


class MergeRule(BaseModel):
    """How one non-ID column is populated on the survivor."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    strategy: MergeStrategy = "survivor"
    order_by: str | None = None

    @field_validator("order_by")
    @classmethod
    def _strip_order_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("order_by must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_order_by(self) -> MergeRule:
        if self.strategy == "latest" and self.order_by is None:
            raise ValueError("latest merge rules require order_by")
        if self.strategy != "latest" and self.order_by is not None:
            raise ValueError("order_by is only valid for latest merge rules")
        return self


class ConsolidationConfig(BaseModel):
    """Strict survivor selection and field merge rules."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    sort_keys: tuple[SurvivorSortKey, ...] = ()
    completeness_columns: tuple[str, ...] = ()
    merge_rules: dict[str, MergeRule] = Field(default_factory=dict)

    @field_validator("completeness_columns")
    @classmethod
    def _normalise_completeness_columns(
        cls,
        columns: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalised = tuple(column.strip() for column in columns)
        if any(not column for column in normalised):
            raise ValueError("completeness columns must not be blank")
        if len(set(normalised)) != len(normalised):
            raise ValueError("completeness columns must be unique")
        return normalised

    @field_validator("merge_rules", mode="before")
    @classmethod
    def _normalise_merge_columns(cls, rules: Any) -> Any:
        if not isinstance(rules, Mapping):
            return rules
        normalised: dict[str, Any] = {}
        for raw_column, rule in rules.items():
            if not isinstance(raw_column, str):
                raise ValueError("merge rule columns must be strings")
            column = raw_column.strip()
            if not column:
                raise ValueError("merge rule columns must not be blank")
            if column in normalised:
                raise ValueError("merge rule columns must be unique")
            normalised[column] = rule
        return normalised

    @model_validator(mode="after")
    def _validate_sort_keys(self) -> ConsolidationConfig:
        columns = [key.column for key in self.sort_keys]
        if len(set(columns)) != len(columns):
            raise ValueError("sort key columns must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ConsolidationAudit:
    """Metadata-only record of a conflict or survivor field change."""

    cluster_id: int
    survivor_id: object
    column: str
    strategy: MergeStrategy
    action: AuditAction
    donor_id: object
    distinct_value_count: int
    conflicting_source_ids: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Consolidated records, lineage, and metadata-safe field audit."""

    frame: pl.DataFrame
    lineage: dict[object, object]
    audit: tuple[ConsolidationAudit, ...]

    @property
    def removed_count(self) -> int:
        """Return how many source records consolidation removed."""
        return len(self.lineage) - self.frame.height

    def summary(self) -> dict[str, int]:
        """Return compact consolidation counts."""
        return {
            "input_count": len(self.lineage),
            "output_count": self.frame.height,
            "removed_count": self.removed_count,
            "cluster_count": self.frame.height,
            "conflict_count": sum(
                entry.distinct_value_count > 1 for entry in self.audit
            ),
            "change_count": sum(
                entry.action != "retained" for entry in self.audit
            ),
        }


_SURVIVOR_RULE = MergeRule()


def consolidate_records(
    frame: pl.DataFrame,
    *,
    id_column: str,
    clusters: Mapping[object, int],
    config: ConsolidationConfig,
) -> ConsolidationResult:
    """Choose one record per cluster and apply explicit field merge rules."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if not isinstance(config, ConsolidationConfig):
        raise TypeError("config must be a ConsolidationConfig")
    if not isinstance(clusters, Mapping):
        raise TypeError("clusters must be a mapping")

    _validate_input(frame, id_column, clusters, config)
    if frame.is_empty():
        return ConsolidationResult(frame.clone(), {}, ())

    record_ids = frame.get_column(id_column).to_list()
    cluster_ids = [clusters[record_id] for record_id in record_ids]
    ordered_cluster_ids = sorted(set(cluster_ids))
    if len(ordered_cluster_ids) == len(record_ids):
        return _singleton_result(
            frame,
            record_ids=record_ids,
            cluster_ids=cluster_ids,
        )

    completeness_column = _unused_name(
        frame.columns,
        "__consolidation_completeness__",
    )
    cluster_column = _unused_name(
        [*frame.columns, completeness_column],
        "__consolidation_cluster__",
    )
    cluster_order = {
        cluster_id: position
        for position, cluster_id in enumerate(ordered_cluster_ids)
    }
    ranked = _rank_records(
        frame.with_columns(
            pl.Series(
                cluster_column,
                [cluster_order[cluster_id] for cluster_id in cluster_ids],
                dtype=pl.UInt64,
            )
        ),
        id_column=id_column,
        cluster_column=cluster_column,
        config=config,
        completeness_column=completeness_column,
    )
    latest_order_ranks = _latest_order_ranks(frame, config)

    rows: list[dict[str, Any]] = []
    audit: list[ConsolidationAudit] = []
    survivors: dict[int, object] = {}
    ranked_groups = groupby(
        ranked.iter_rows(named=True),
        key=lambda row: row[cluster_column],
    )
    for cluster_position, group in ranked_groups:
        cluster_id = ordered_cluster_ids[cluster_position]
        ranked_rows = list(group)
        survivor = ranked_rows[0]
        survivor_id = survivor[id_column]
        survivors[cluster_id] = survivor_id
        if len(ranked_rows) == 1:
            rows.append({column: survivor[column] for column in frame.columns})
            continue
        consolidated = {column: survivor[column] for column in frame.columns}

        for column in frame.columns:
            if column == id_column:
                continue
            rule = config.merge_rules.get(column, _SURVIVOR_RULE)
            dtype = frame.schema[column]
            present = [
                row
                for row in ranked_rows
                if not _value_is_filter_missing(row[column], dtype)
            ]
            value_groups = _group_values(
                (row[id_column], row[column]) for row in present
            )
            donor_id, value = _select_value(
                ranked_rows,
                present,
                value_groups,
                id_column=id_column,
                column=column,
                rule=rule,
                schema=frame.schema,
                latest_order_ranks=latest_order_ranks,
            )
            previous = survivor[column]
            changed = not _values_equal(
                previous,
                value,
                dtype,
            )
            if changed:
                consolidated[column] = value

            distinct_count = len(value_groups)
            if changed or distinct_count > 1:
                audit.append(
                    ConsolidationAudit(
                        cluster_id=cluster_id,
                        survivor_id=survivor_id,
                        column=column,
                        strategy=rule.strategy,
                        action=_audit_action(
                            previous,
                            changed=changed,
                            dtype=dtype,
                        ),
                        donor_id=donor_id,
                        distinct_value_count=distinct_count,
                        conflicting_source_ids=(
                            tuple(row[id_column] for row in present)
                            if distinct_count > 1
                            else ()
                        ),
                    )
                )
            del present, value_groups
        rows.append(consolidated)
        del ranked_rows

    del latest_order_ranks, ranked_groups, ranked
    output = _build_output(rows, frame.schema)
    return ConsolidationResult(
        output,
        {
            record_id: survivors[clusters[record_id]]
            for record_id in record_ids
        },
        tuple(audit),
    )


def _singleton_result(
    frame: pl.DataFrame,
    *,
    record_ids: list[object],
    cluster_ids: list[int],
) -> ConsolidationResult:
    output = frame[
        sorted(range(len(record_ids)), key=cluster_ids.__getitem__),
        :,
    ]
    return ConsolidationResult(
        output,
        {record_id: record_id for record_id in record_ids},
        (),
    )


def _validate_input(
    frame: pl.DataFrame,
    id_column: str,
    clusters: Mapping[object, int],
    config: ConsolidationConfig,
) -> None:
    record_ids = _validate_consolidation_frame(frame, id_column, config)
    _validate_clusters(record_ids, clusters)


def _validate_consolidation_frame(
    frame: pl.DataFrame,
    id_column: str,
    config: ConsolidationConfig,
) -> pl.Series:
    if not id_column or not id_column.strip():
        raise ValueError("id_column must not be blank")
    if id_column not in frame.columns:
        raise ValueError(f"frame is missing ID column {id_column!r}")

    unsupported = [
        column
        for column, dtype in frame.schema.items()
        if dtype.is_nested() or dtype == pl.Object
    ]
    reject_nested_columns(unsupported)

    configured_columns = {
        *(key.column for key in config.sort_keys),
        *config.completeness_columns,
        *config.merge_rules,
        *(
            rule.order_by
            for rule in config.merge_rules.values()
            if rule.order_by is not None
        ),
    }
    missing = sorted(configured_columns - set(frame.columns))
    if missing:
        raise ValueError(f"frame is missing configured columns: {missing}")
    protected_columns = {
        *(key.column for key in config.sort_keys),
        *config.completeness_columns,
        *config.merge_rules,
    }
    if id_column in protected_columns:
        raise ValueError("the ID column is protected from consolidation rules")

    record_ids = frame.get_column(id_column)
    missing_id_count = frame.select(
        _is_missing(id_column, record_ids.dtype).sum()
    ).item()
    if missing_id_count:
        raise ValueError("record IDs must not be null, NaN, or blank")
    if record_ids.dtype.is_float() and record_ids.is_infinite().any():
        raise ValueError("record IDs must be finite")
    if record_ids.n_unique() != record_ids.len():
        raise ValueError("record IDs must be unique")
    _require_orderable(record_ids, id_column)

    order_columns = {
        *(key.column for key in config.sort_keys),
        *(
            rule.order_by
            for rule in config.merge_rules.values()
            if rule.order_by is not None
        ),
    }
    for column in order_columns:
        _require_orderable(frame.get_column(column), column)
    for column, rule in config.merge_rules.items():
        if rule.strategy != "latest" or rule.order_by is None:
            continue
        unclear = frame.filter(
            _is_present(column, frame.schema[column])
            & _is_missing(rule.order_by, frame.schema[rule.order_by])
        )
        if not unclear.is_empty():
            ids = unclear.get_column(id_column).to_list()
            raise ValueError(
                f"latest merge for {column!r} requires non-missing "
                f"{rule.order_by!r} values for source IDs: {ids}"
            )
    return record_ids


def _validate_clusters(
    record_ids: pl.Series,
    clusters: Mapping[object, int],
) -> None:
    ids = record_ids.to_list()
    if ids and not clusters:
        raise ValueError(
            "clusters must cover every record; link-mode results cannot "
            "be consolidated"
        )
    try:
        input_ids = set(ids)
        cluster_ids = set(clusters)
    except TypeError as exc:
        raise ValueError("record and cluster-map IDs must be scalar") from exc
    missing_ids = input_ids - cluster_ids
    unknown_ids = cluster_ids - input_ids
    if missing_ids or unknown_ids or len(clusters) != len(ids):
        raise ValueError(
            "clusters must map every input ID exactly once "
            f"(missing={len(missing_ids)}, unknown={len(unknown_ids)})"
        )
    if any(
        type(cluster_id) is not int or cluster_id < 0
        for cluster_id in clusters.values()
    ):
        raise ValueError("cluster IDs must be non-negative integers")


def _rank_records(
    frame: pl.DataFrame,
    *,
    id_column: str,
    cluster_column: str,
    config: ConsolidationConfig,
    completeness_column: str,
) -> pl.DataFrame:
    columns: list[str | pl.Expr] = [
        cluster_column,
        *[
            _null_missing_values(key.column, frame.schema[key.column])
            for key in config.sort_keys
        ],
    ]
    descending = [
        False,
        *(key.descending for key in config.sort_keys),
    ]
    if config.completeness_columns:
        completeness = pl.sum_horizontal(
            *[
                _is_present(column, frame.schema[column]).cast(pl.UInt32)
                for column in config.completeness_columns
            ]
        ).alias(completeness_column)
        frame = frame.with_columns(completeness)
        columns.append(completeness_column)
        descending.append(True)
    columns.append(id_column)
    descending.append(False)
    return frame.sort(
        columns,
        descending=descending,
        nulls_last=True,
        maintain_order=True,
    )


def _select_value(
    ranked: list[dict[str, Any]],
    present: list[dict[str, Any]],
    groups: list[list[tuple[object, Any]]],
    *,
    id_column: str,
    column: str,
    rule: MergeRule,
    schema: pl.Schema,
    latest_order_ranks: Mapping[str, dict[Any, int]],
) -> tuple[object, Any]:
    survivor = ranked[0]
    if rule.strategy == "survivor" or not present:
        return survivor[id_column], survivor[column]
    if rule.strategy == "first_non_null":
        donor = present[0]
        return donor[id_column], donor[column]
    if rule.strategy == "most_frequent":
        donor_id, value = max(groups, key=len)[0]
        return donor_id, value

    order_by = rule.order_by
    if order_by is None:  # guarded by MergeRule validation
        raise AssertionError("latest merge rule is missing order_by")
    unclear = [
        row[id_column]
        for row in present
        if _value_is_filter_missing(row[order_by], schema[order_by])
    ]
    if unclear:
        raise ValueError(
            f"latest merge for {column!r} requires non-missing "
            f"{order_by!r} values for source IDs: {unclear}"
        )
    ranks = latest_order_ranks[order_by]
    donor = max(present, key=lambda row: ranks[row[order_by]])
    return donor[id_column], donor[column]


def _latest_order_ranks(
    frame: pl.DataFrame,
    config: ConsolidationConfig,
) -> dict[str, dict[Any, int]]:
    order_columns = {
        rule.order_by
        for rule in config.merge_rules.values()
        if rule.strategy == "latest" and rule.order_by is not None
    }
    ranks: dict[str, dict[Any, int]] = {}
    for column in order_columns:
        ordered = (
            frame.filter(_is_present(column, frame.schema[column]))
            .select(column)
            .unique(maintain_order=True)
            .sort(column)
            .get_column(column)
            .to_list()
        )
        ranks[column] = {
            value: position for position, value in enumerate(ordered)
        }
    return ranks


def _value_is_filter_missing(value: Any, dtype: pl.DataType) -> bool:
    if value is None:
        return True
    if dtype.is_float():
        return isinstance(value, float) and math.isnan(value)
    if dtype.base_type() in {pl.String, pl.Categorical, pl.Enum}:
        return isinstance(value, str) and not value.strip()
    return False


def _group_values(
    rows: Iterable[tuple[object, Any]],
) -> list[list[tuple[object, Any]]]:
    groups: dict[Any, list[tuple[object, Any]]] = {}
    for source_id, value in rows:
        try:
            groups.setdefault(value, []).append((source_id, value))
        except TypeError as exc:
            raise ValueError(
                "consolidation values must be hashable scalars"
            ) from exc
    return list(groups.values())


def _build_output(
    rows: list[dict[str, Any]],
    schema: pl.Schema,
) -> pl.DataFrame:
    columns: list[pl.Series] = []
    for column, dtype in schema.items():
        try:
            columns.append(
                pl.Series(
                    column,
                    [row[column] for row in rows],
                    dtype=dtype,
                    strict=True,
                )
            )
        except Exception as exc:
            raise ValueError(
                f"consolidation cannot preserve dtype {dtype!s} "
                f"for column {column!r}"
            ) from exc
    output = pl.DataFrame(columns)
    if output.schema != schema:
        raise ValueError("consolidation could not preserve the input schema")
    return output


def _null_missing_values(column: str, dtype: pl.DataType) -> pl.Expr:
    return (
        pl.when(_is_missing(column, dtype))
        .then(None)
        .otherwise(pl.col(column))
    )


def _is_present(column: str, dtype: pl.DataType) -> pl.Expr:
    return ~_is_missing(column, dtype)


def _is_missing(column: str, dtype: pl.DataType) -> pl.Expr:
    expression = pl.col(column).is_null()
    if dtype.is_float():
        expression = expression | pl.col(column).is_nan()
    elif dtype.base_type() in {pl.String, pl.Categorical, pl.Enum}:
        expression = expression | pl.col(column).cast(
            pl.String
        ).str.strip_chars().eq("")
    return expression


def _value_is_missing(value: Any, dtype: pl.DataType) -> bool:
    if value is None:
        return True
    if dtype.is_float():
        return isinstance(value, float) and math.isnan(value)
    if dtype.is_decimal():
        return isinstance(value, Decimal) and value.is_nan()
    if dtype.base_type() in {pl.String, pl.Categorical, pl.Enum}:
        return isinstance(value, str) and not value.strip()
    return False


def _values_equal(
    left: Any,
    right: Any,
    dtype: pl.DataType,
) -> bool:
    if _value_is_missing(left, dtype):
        return _value_is_missing(right, dtype)
    if _value_is_missing(right, dtype):
        return False
    return bool(left == right)


def _audit_action(
    previous: Any,
    *,
    changed: bool,
    dtype: pl.DataType,
) -> AuditAction:
    if not changed:
        return "retained"
    if _value_is_missing(previous, dtype):
        return "filled"
    return "replaced"


def _require_orderable(series: pl.Series, column: str) -> None:
    try:
        pl.DataFrame({column: series}).filter(
            _is_present(column, series.dtype)
        ).get_column(column).sort()
    except Exception as exc:
        raise ValueError(f"column {column!r} must be orderable") from exc


def _unused_name(columns: list[str], preferred: str) -> str:
    name = preferred
    while name in columns:
        name = f"_{name}"
    return name
