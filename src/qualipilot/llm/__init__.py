"""Pluggable LLM provider layer.

The orchestrator talks to the ``LLMProvider`` ABC; concrete providers
are loaded lazily so optional deps like ``boto3`` only import when
that provider is selected.
"""

from __future__ import annotations

from qualipilot.llm.base import LLMProvider
from qualipilot.models.config import LLMConfig

__all__ = ["LLMProvider", "build_provider"]


def build_provider(cfg: LLMConfig) -> LLMProvider:
    """Instantiate the provider declared in ``cfg``.

    Args:
        cfg: Runtime LLM configuration.

    Returns:
        A ready-to-call ``LLMProvider`` instance.

    Raises:
        ValueError: If the provider name is unknown.
        ImportError: If a required optional dependency is missing.
    """
    name = cfg.provider
    if name == "none":
        from qualipilot.llm.null_provider import NullProvider

        return NullProvider()
    if name == "bedrock":
        from qualipilot.llm.bedrock import BedrockProvider

        return BedrockProvider(cfg)
    if name == "ollama":
        from qualipilot.llm.ollama import OllamaProvider

        return OllamaProvider(cfg)
    if name == "openai":
        from qualipilot.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg)
    raise ValueError(f"unknown llm provider: {name!r}")
