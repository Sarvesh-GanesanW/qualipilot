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
