"""LLM provider tests.

We mock network IO end-to-end: ``httpx.Client`` for ollama/openai,
``moto`` for bedrock. This keeps the suite runnable offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from datapilot.llm import build_provider
from datapilot.models.config import LLMConfig


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


@pytest.mark.integration
def test_bedrock_provider_happy_path() -> None:
    boto3 = pytest.importorskip("boto3")
    _ = boto3
    cfg = LLMConfig(
        provider="bedrock",
        model="provider.model-3-5-haiku-20241022-v1:0",
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
    ):
        provider = build_provider(cfg)
        assert provider.generate(system="s", user="u") == "done"


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
