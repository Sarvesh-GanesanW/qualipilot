"""Configuration boundary regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qualipilot.linking.comparisons import ExactMatch
from qualipilot.linking.config import LinkConfig
from qualipilot.models.config import (
    CheckConfig,
    ColumnRange,
    LLMConfig,
    QualipilotConfig,
)


def test_environment_overrides_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "qualipilot.yaml"
    path.write_text(
        "engine: polars\nchecks:\n  sample_size: 7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QUALIPILOT_ENGINE", "pandas")
    monkeypatch.setenv("QUALIPILOT_CHECKS__SAMPLE_SIZE", "0")

    config = QualipilotConfig.from_file(path)

    assert config.engine == "pandas"
    assert config.checks.sample_size == 0


def test_partial_environment_values_merge_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "qualipilot.yaml"
    path.write_text(
        "llm:\n  model: configured-model\n  region: us-east-1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QUALIPILOT_LLM__PROVIDER", "bedrock")

    config = QualipilotConfig.from_file(path)

    assert config.llm.provider == "bedrock"
    assert config.llm.model == "configured-model"


def test_config_file_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "qualipilot.yaml"
    path.write_text(
        "checks:\n  freshnes: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="freshnes"):
        QualipilotConfig.from_file(path)


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".json", '{"engine":"polars","engine":"pandas"}'),
        (".yaml", "checks:\n  min_rows: 1\n  min_rows: 2\n"),
    ],
)
def test_config_file_rejects_duplicate_keys(
    tmp_path: Path,
    suffix: str,
    content: str,
) -> None:
    path = tmp_path / f"qualipilot{suffix}"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate config key"):
        QualipilotConfig.from_file(path)


def test_ranges_reject_non_finite_bounds() -> None:
    with pytest.raises(ValidationError):
        ColumnRange(min=float("nan"), max=10)
    with pytest.raises(ValidationError):
        ColumnRange(min=0, max=float("inf"))


def test_check_column_names_are_normalized() -> None:
    config = CheckConfig(
        required_columns=[" id "],
        expected_dtypes={" amount ": " Float64 "},
        column_ranges={" amount ": ColumnRange(min=0, max=1)},
    )

    assert config.required_columns == ["id"]
    assert config.expected_dtypes == {"amount": "Float64"}
    assert config.column_ranges == {"amount": ColumnRange(min=0, max=1)}


def test_severity_overrides_parse_from_config_file(tmp_path: Path) -> None:
    path = tmp_path / "qualipilot.yaml"
    path.write_text(
        "checks:\n"
        "  severity_overrides:\n"
        "    missing_values: error\n"
        "    ranges: warn\n",
        encoding="utf-8",
    )

    config = QualipilotConfig.from_file(path)

    assert config.checks.severity_overrides == {
        "missing_values": "error",
        "ranges": "warn",
    }


@pytest.mark.parametrize(
    "severity_overrides",
    [
        {"custom_check": "warn"},
        {"missing_values": "critical"},
        {"missing_values": "WARN"},
    ],
)
def test_severity_overrides_reject_unknown_names_and_values(
    severity_overrides: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="severity_overrides"):
        CheckConfig.model_validate({"severity_overrides": severity_overrides})


@pytest.mark.parametrize(
    "values",
    [
        {"required_columns": [" "]},
        {"required_columns": ["id", " id "]},
        {"freshness_columns": ["Event_Time", "event_time"]},
        {"duplicate_subset": ["ID", "id"]},
        {"expected_dtypes": {"id": "int", " id ": "str"}},
        {"expected_dtypes": {"ID": "int", "id": "int"}},
        {
            "column_ranges": {
                "amount": ColumnRange(min=0, max=1),
                " amount ": ColumnRange(min=0, max=1),
            }
        },
        {
            "column_ranges": {
                "Amount": ColumnRange(min=0, max=1),
                "amount": ColumnRange(min=0, max=1),
            }
        },
    ],
)
def test_check_column_names_reject_blanks_and_normalized_duplicates(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match=r"empty|duplicates"):
        CheckConfig(**values)


def test_api_key_is_hidden_and_survives_file_loading(tmp_path: Path) -> None:
    path = tmp_path / "qualipilot.yaml"
    path.write_text(
        "llm:\n"
        "  provider: openai\n"
        "  model: test-model\n"
        "  base_url: https://api.example.com/v1\n"
        "  api_key: production-secret\n",
        encoding="utf-8",
    )

    config = QualipilotConfig.from_file(path)

    assert config.llm.api_key is not None
    assert config.llm.api_key.get_secret_value() == "production-secret"
    assert "production-secret" not in repr(config.llm)
    assert "production-secret" not in config.model_dump_json()


def test_connection_name_selects_gz_provider() -> None:
    config = LLMConfig(connection_name=" TestDataQuality ")

    assert config.provider == "gz"
    assert config.connection_name == "TestDataQuality"
    assert config.model == ""


def test_gz_provider_can_be_explicit() -> None:
    config = LLMConfig(
        provider="gz",
        connection_name="TestDataQuality",
    )

    assert config.provider == "gz"


@pytest.mark.parametrize("connection_name", ["", "   "])
def test_gz_connection_name_rejects_blanks(connection_name: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        LLMConfig(connection_name=connection_name)


def test_gz_provider_requires_connection_name() -> None:
    with pytest.raises(ValidationError, match="requires connection_name"):
        LLMConfig(provider="gz")


@pytest.mark.parametrize("provider", ["none", "bedrock", "openai"])
def test_connection_name_rejects_direct_provider(provider: str) -> None:
    with pytest.raises(ValidationError, match="direct provider"):
        LLMConfig(
            provider=provider,  # type: ignore[arg-type]
            connection_name="TestDataQuality",
        )


@pytest.mark.parametrize(
    "direct_setting",
    [
        {"model": "direct-model"},
        {"api_key": "direct-secret"},
        {"base_url": "https://api.example.com/v1"},
        {"region": "eu-west-1"},
        {"aws_profile": "direct-profile"},
    ],
)
def test_connection_name_rejects_direct_provider_settings(
    direct_setting: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="cannot be combined"):
        LLMConfig(
            connection_name="TestDataQuality",
            **direct_setting,
        )


@pytest.mark.parametrize("provider", ["bedrock", "ollama", "openai"])
def test_enabled_provider_requires_explicit_model(provider: str) -> None:
    with pytest.raises(ValidationError, match="requires an explicit model"):
        LLMConfig(provider=provider)  # type: ignore[arg-type]


def test_remote_openai_endpoint_requires_tls() -> None:
    with pytest.raises(ValidationError, match="must use https"):
        LLMConfig(
            provider="openai",
            model="test-model",
            base_url="http://api.example.com/v1",
        )

    config = LLMConfig(
        provider="openai",
        model="test-model",
        base_url="http://api.example.com/v1",
        allow_insecure_http=True,
    )
    assert config.allow_insecure_http


def test_remote_ollama_endpoint_requires_tls() -> None:
    with pytest.raises(ValidationError, match="must use https"):
        LLMConfig(
            provider="ollama",
            model="local-model",
            base_url="http://ollama.example.com",
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:secret@example.com/v1",
        "https://example.com/v1;token=secret",
        "https://example.com/v1?token=secret",
        "https://example.com/v1#secret",
    ],
)
def test_llm_base_url_rejects_embedded_credentials(
    base_url: str,
) -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        LLMConfig(
            provider="openai",
            model="test-model",
            base_url=base_url,
        )


def test_malformed_api_key_is_hidden_in_validation_errors() -> None:
    secret = "should-never-appear"

    with pytest.raises(ValidationError) as raised:
        LLMConfig.model_validate({"api_key": {"password": secret}})

    assert secret not in str(raised.value)


def test_embedded_linkage_rejects_two_table_mode() -> None:
    link = LinkConfig(
        mode="link",
        unique_id_column="id",
        comparisons=[ExactMatch(column="email")],
    )

    with pytest.raises(ValidationError, match="dedupe mode only"):
        CheckConfig(linkage=link)


def test_linkage_rejects_incompatible_engine() -> None:
    link = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="email")],
    )

    with pytest.raises(ValidationError, match="polars or pandas"):
        QualipilotConfig(engine="duckdb", checks=CheckConfig(linkage=link))
