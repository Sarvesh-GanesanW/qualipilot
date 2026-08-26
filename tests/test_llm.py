"""LLM provider tests.

We mock network IO end-to-end: ``httpx.Client`` for ollama/openai,
``moto`` for bedrock. This keeps the suite runnable offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from qualipilot.llm import build_provider
from qualipilot.models.config import LLMConfig


def test_null_provider_returns_empty() -> None:
    provider = build_provider(LLMConfig(provider="none"))
    assert provider.generate(system="x", user="y") == ""


def test_ollama_provider_strips_v1_suffix() -> None:
    cfg = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )

    provider = build_provider(cfg)
    assert provider._base == "http://localhost:11434"


def test_ollama_round_trip_is_mocked() -> None:
    cfg = LLMConfig(provider="ollama", model="llama3.2")
    provider = build_provider(cfg)

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"message": {"content": "ok"}}

    with patch("httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        out = provider.generate(system="s", user="u")

    assert out == "ok"


def test_openai_provider_sends_bearer() -> None:
    cfg = LLMConfig(
        provider="openai",
        base_url="https://api.example.com",
        api_key="sk-test",
        model="gpt-4o-mini",
    )
    provider = build_provider(cfg)

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "hi"}}]
    }

    with patch("httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        out = provider.generate(system="s", user="u")

    assert out == "hi"
    # bearer token must flow through
    kwargs = client.post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test"


def test_openai_does_not_retry_permanent_http_errors() -> None:
    cfg = LLMConfig(
        provider="openai",
        base_url="https://api.example.com",
        model="test-model",
        retries=0,
    )
    provider = build_provider(cfg)
    request = httpx.Request("POST", "https://api.example.com")
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=response,
    )
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = error

    with patch("httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        with pytest.raises(httpx.HTTPStatusError):
            provider.generate(system="s", user="u")

    client.post.assert_called_once()


def test_bedrock_provider_happy_path() -> None:
    boto3 = pytest.importorskip("boto3")
    _ = boto3
    cfg = LLMConfig(
        provider="bedrock",
        model="provider.test-model-v1",
        retries=4,
    )

    fake_response = {
        "output": {"message": {"content": [{"text": "done"}]}},
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
        },
    }
    with patch(
        "boto3.Session.client",
        return_value=MagicMock(converse=MagicMock(return_value=fake_response)),
    ) as client_method:
        provider = build_provider(cfg)
        assert provider.generate(system="s", user="u") == "done"
    boto_config = client_method.call_args.kwargs["config"]
    assert boto_config.retries["total_max_attempts"] == 5


def test_bedrock_rejects_unexpected_response() -> None:
    pytest.importorskip("boto3")
    cfg = LLMConfig(provider="bedrock", model="provider.test-model-v1")

    with patch(
        "boto3.Session.client",
        return_value=MagicMock(converse=MagicMock(return_value={})),
    ):
        provider = build_provider(cfg)
        with pytest.raises(RuntimeError, match="unexpected bedrock response"):
            provider.generate(system="s", user="u")


def test_bedrock_collects_text_after_reasoning_blocks() -> None:
    pytest.importorskip("boto3")
    cfg = LLMConfig(provider="bedrock", model="provider.reasoning-model-v1")
    response = {
        "output": {
            "message": {
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "hidden"}
                        }
                    },
                    {"text": "first"},
                    {"text": "second"},
                ]
            }
        }
    }

    with patch(
        "boto3.Session.client",
        return_value=MagicMock(converse=MagicMock(return_value=response)),
    ):
        provider = build_provider(cfg)
        assert provider.generate(system="s", user="u") == "first\nsecond"


def test_gz_openai_uses_typed_getter_and_raw_model_fallback() -> None:
    secret = "gz-openai-secret"
    connections = MagicMock()
    connections.conn = {
        "TestDataQuality": {
            "type": "openai",
            "model": "gpt-from-connection",
        }
    }
    connections.getOpenAIDetails.return_value = {
        "apiKey": secret,
        "baseUrl": "https://api.example.com/v1",
    }
    config = LLMConfig(connection_name="TestDataQuality")
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "checked"}}]
    }

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ) as import_module,
        patch("httpx.Client") as client_cls,
    ):
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        provider = build_provider(config)
        result = provider.generate(system="system", user="summary")

    assert result == "checked"
    import_module.assert_called_once_with("GZ.CONNECTIONS")
    connections.getOpenAIDetails.assert_called_once_with("TestDataQuality")
    request = client.post.call_args
    assert request.args[0] == "https://api.example.com/v1/chat/completions"
    assert request.kwargs["json"]["model"] == "gpt-from-connection"
    assert request.kwargs["headers"]["Authorization"] == f"Bearer {secret}"
    assert secret not in config.model_dump_json()
    assert secret not in repr(provider)
    assert secret not in repr(provider._delegate._cfg)


def test_gz_azure_openai_uses_deployment_endpoint_and_api_key_header() -> None:
    connections = MagicMock()
    connections.conn = {"Azure": {"type": "azureopenai"}}
    connections.getAzureOpenAIDetails.return_value = {
        "apiKey": "azure-secret",
        "endpoint": "https://resource.openai.azure.com",
        "deploymentName": "quality model",
        "apiVersion": "2024-10-21",
    }
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "checked"}}]
    }

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        patch("httpx.Client") as client_cls,
    ):
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        provider = build_provider(LLMConfig(connection_name="Azure"))
        result = provider.generate(system="system", user="summary")

    request = client.post.call_args
    assert result == "checked"
    assert request.args[0] == (
        "https://resource.openai.azure.com/openai/deployments/"
        "quality%20model/chat/completions?api-version=2024-10-21"
    )
    assert request.kwargs["headers"]["api-key"] == "azure-secret"
    assert "Authorization" not in request.kwargs["headers"]
    assert "model" not in request.kwargs["json"]


def test_gz_bedrock_uses_connection_credentials() -> None:
    pytest.importorskip("boto3")
    secret = "gz-bedrock-secret"
    connections = MagicMock()
    connections.conn = {
        "TestDataQuality": {
            "type": "awsbedrock",
            "model": "anthropic.test-model-v1",
        }
    }
    connections.getAwsBedrockDetails.return_value = {
        "accessKey": "access-key",
        "secretAccessKey": secret,
        "sessionToken": "session-token",
        "region": "eu-west-1",
    }
    response = {"output": {"message": {"content": [{"text": "checked"}]}}}
    runtime = MagicMock(converse=MagicMock(return_value=response))
    session = MagicMock()
    session.client.return_value = runtime
    config = LLMConfig(connection_name="TestDataQuality")

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        patch("boto3.Session", return_value=session) as session_class,
    ):
        provider = build_provider(config)
        result = provider.generate(system="system", user="summary")

    assert result == "checked"
    connections.getAwsBedrockDetails.assert_called_once_with("TestDataQuality")
    assert session_class.call_args.kwargs == {
        "region_name": "eu-west-1",
        "aws_access_key_id": "access-key",
        "aws_secret_access_key": secret,
        "aws_session_token": "session-token",
    }
    assert secret not in config.model_dump_json()
    assert secret not in repr(provider)


def test_gz_gemini_keeps_api_key_out_of_url() -> None:
    secret = "gz-gemini-secret"
    connections = MagicMock()
    connections.conn = {"Gemini": {"type": "gemini"}}
    connections.getGeminiDetails.return_value = {
        "apiKey": secret,
        "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-test",
    }
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "checked"}]}}]
    }

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        patch("httpx.Client") as client_cls,
    ):
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        provider = build_provider(LLMConfig(connection_name="Gemini"))
        result = provider.generate(system="system", user="summary")

    request = client.post.call_args
    assert result == "checked"
    assert secret not in request.args[0]
    assert request.kwargs["headers"]["x-goog-api-key"] == secret


@pytest.mark.parametrize(
    ("connection_type", "getter_name"),
    [
        ("openai", "getOpenAIDetails"),
        ("gemini", "getGeminiDetails"),
    ],
)
def test_gz_requires_an_explicit_model(
    connection_type: str,
    getter_name: str,
) -> None:
    connections = MagicMock()
    connections.conn = {"NoModel": {"type": connection_type}}
    getattr(connections, getter_name).return_value = {"apiKey": "secret"}

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        pytest.raises(
            ValueError,
            match="GZ connection 'NoModel' requires model",
        ),
    ):
        build_provider(LLMConfig(connection_name="NoModel"))


def test_gz_huggingface_requires_an_explicit_endpoint() -> None:
    connections = MagicMock()
    connections.conn = {
        "HuggingFace": {
            "type": "huggingface",
            "model": "org/model",
        }
    }
    connections.getHuggingFaceDetails.return_value = {"apiKey": "secret"}

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        pytest.raises(
            ValueError,
            match="requires inferenceEndpointUrl or inferenceBaseUrl",
        ),
    ):
        build_provider(LLMConfig(connection_name="HuggingFace"))


@pytest.mark.parametrize(
    (
        "connection_type",
        "getter_name",
        "details",
        "raw_fields",
        "response_data",
        "expected_url",
        "expected_headers",
        "expected_payload",
    ),
    [
        (
            "anthropic",
            "getAnthropicDetails",
            {
                "apiKey": "native-secret",
                "baseUrl": "https://api.anthropic.com",
                "model": "claude-test",
                "anthropicVersion": "2023-06-01",
            },
            {},
            {"content": [{"text": "checked"}]},
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": "native-secret",
                "anthropic-version": "2023-06-01",
            },
            {
                "model": "claude-test",
                "max_tokens": 99,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": "summary"}],
                "system": "system",
            },
        ),
        (
            "cohere",
            "getCohereDetails",
            {
                "apiKey": "native-secret",
                "baseUrl": "https://api.cohere.com",
                "model": "command-test",
            },
            {},
            {"message": {"content": [{"text": "checked"}]}},
            "https://api.cohere.com/v2/chat",
            {"Authorization": "Bearer native-secret"},
            {
                "model": "command-test",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "summary"},
                ],
                "max_tokens": 99,
                "temperature": 0.3,
            },
        ),
        (
            "huggingface",
            "getHuggingFaceDetails",
            {"apiKey": "native-secret"},
            {
                "model": "org/test model",
                "inferenceBaseUrl": "https://inference.example.com",
            },
            [{"generated_text": "checked"}],
            "https://inference.example.com/models/org%2Ftest%20model",
            {"Authorization": "Bearer native-secret"},
            {
                "inputs": "system\n\nsummary",
                "parameters": {
                    "max_new_tokens": 99,
                    "return_full_text": False,
                    "temperature": 0.3,
                },
            },
        ),
    ],
)
def test_gz_native_http_connectors_generate_offline(
    connection_type: str,
    getter_name: str,
    details: dict[str, object],
    raw_fields: dict[str, str],
    response_data: object,
    expected_url: str,
    expected_headers: dict[str, str],
    expected_payload: dict[str, object],
) -> None:
    connections = MagicMock()
    connections.conn = {"Native": {"type": connection_type, **raw_fields}}
    getattr(connections, getter_name).return_value = details
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = response_data

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        patch("httpx.Client") as client_cls,
    ):
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_response
        provider = build_provider(
            LLMConfig(
                connection_name="Native",
                max_tokens=99,
                temperature=0.3,
            )
        )
        result = provider.generate(system="system", user="summary")

    request = client.post.call_args
    assert result == "checked"
    assert request.args[0] == expected_url
    assert request.kwargs["json"] == expected_payload
    assert (
        request.kwargs["headers"] | expected_headers
        == request.kwargs["headers"]
    )
    assert "native-secret" not in request.args[0]


@pytest.mark.parametrize(
    ("raw_type", "getter_name", "canonical_type", "base_url"),
    [
        (
            "claude",
            "getAnthropicDetails",
            "anthropic",
            "https://api.anthropic.com",
        ),
        (
            "google_gemini",
            "getGeminiDetails",
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
    ],
)
def test_gz_spark_aliases_use_canonical_http_route(
    raw_type: str,
    getter_name: str,
    canonical_type: str,
    base_url: str,
) -> None:
    connections = MagicMock()
    connections.conn = {"Alias": {"type": raw_type}}
    getter = getattr(connections, getter_name)
    getter.return_value = {
        "apiKey": "secret",
        "baseUrl": base_url,
        "model": "test-model",
    }

    with patch(
        "qualipilot.llm.gz.import_module",
        return_value=connections,
    ):
        provider = build_provider(LLMConfig(connection_name="Alias"))

    getter.assert_called_once_with("Alias")
    assert provider._delegate._connection_type == canonical_type


def test_gz_rejects_insecure_remote_endpoint() -> None:
    connections = MagicMock()
    connections.conn = {"Remote": {"type": "openai", "model": "test"}}
    connections.getOpenAIDetails.return_value = {
        "apiKey": "secret",
        "baseUrl": "http://api.example.com/v1",
    }

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        pytest.raises(ValueError, match="must use https"),
    ):
        build_provider(LLMConfig(connection_name="Remote"))


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ({}, "was not found"),
        ({"Bad": {"type": "snowflake"}}, "unsupported LLM type"),
    ],
)
def test_gz_connection_errors_are_actionable(
    registry: dict[str, dict[str, str]],
    message: str,
) -> None:
    connections = MagicMock()
    connections.conn = registry

    with (
        patch(
            "qualipilot.llm.gz.import_module",
            return_value=connections,
        ),
        pytest.raises(ValueError, match=message),
    ):
        build_provider(LLMConfig(connection_name="Bad"))


def test_unknown_provider_raises() -> None:
    # pydantic guards the happy path via Literal; we exercise the fallback
    # by handing build_provider a config-like object with an unknown name
    class _Fake:
        provider = "gibberish"
        model = ""
        region = "us-east-1"
        aws_profile = None
        base_url = ""
        api_key = None
        max_tokens = 100
        temperature = 0.0
        timeout_seconds = 1.0
        retries = 0
        system_prompt = ""

    with pytest.raises(ValueError, match="unknown llm provider"):
        build_provider(_Fake())  # type: ignore[arg-type]
