from typing import assert_type

from qualipilot.models.results import (
    CheckResult,
    DuplicatesPayload,
    JSONValue,
    TypedPayload,
)


def test_static_payload_api() -> None:
    result = CheckResult(
        name="duplicates",
        severity="ok",
        duration_seconds=0,
        payload={"total_duplicate_rows": 1},
    )
    duplicate = result.payload_as(DuplicatesPayload)
    assert_type(duplicate, DuplicatesPayload)
    assert_type(duplicate.total_duplicate_rows, int)
    payload = result.typed_payload()
    assert_type(payload, TypedPayload)
    if isinstance(payload, DuplicatesPayload):
        assert_type(payload.sample, list[dict[str, JSONValue]])


def test_dataset_contract_payload_is_concrete_and_rejects_bad_shape() -> None:
    from pydantic import ValidationError

    from qualipilot.models.results import DatasetContractPayload

    payload = DatasetContractPayload.model_validate(
        {
            "row_count": 1,
            "min_rows": 1,
            "missing_required_columns": [],
            "dtype_mismatches": [
                {"column": "id", "expected": "int", "actual": "string"}
            ],
        }
    )
    assert payload.dtype_mismatches[0].expected == "int"
    with __import__("pytest").raises(ValidationError):
        DatasetContractPayload.model_validate(
            {
                "row_count": 1,
                "min_rows": 1,
                "missing_required_columns": [],
                "dtype_mismatches": [{"column": "id"}],
            }
        )
