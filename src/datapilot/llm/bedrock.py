"""AWS Bedrock provider using the Converse API.

We use Converse (not InvokeModel) because it unifies request shape
across Provider, Meta, Mistral, etc. — so switching models is a
config change, not a code change.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from datapilot.llm.base import LLMProvider
from datapilot.models.config import LLMConfig

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """Bedrock Converse API provider."""

    name = "bedrock"

    def __init__(self, cfg: LLMConfig) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for BedrockProvider; "
                "install with `pip install data-pilot-checker[bedrock]`"
            ) from exc

        self._cfg = cfg
        self._model_id = (
            cfg.model or "provider.model-3-5-haiku-20241022-v1:0"
        )
        session_kwargs: dict[str, Any] = {"region_name": cfg.region}
        if cfg.aws_profile:
            session_kwargs["profile_name"] = cfg.aws_profile
        session = boto3.Session(**session_kwargs)

        # adaptive retries + keep-alive pooling trims cold-start cost
        boto_config = BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            read_timeout=cfg.timeout_seconds,
            connect_timeout=10,
        )
        self._client = session.client("bedrock-runtime", config=boto_config)

    def generate(self, *, system: str, user: str) -> str:
        messages = [
            {"role": "user", "content": [{"text": user}]},
        ]
        system_blocks = [{"text": system}] if system else []

        return self._converse_with_retry(messages, system_blocks)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(
            (RuntimeError, ConnectionError, TimeoutError)
        ),
    )
    def _converse_with_retry(
        self,
        messages: list[dict[str, Any]],
        system_blocks: list[dict[str, Any]],
    ) -> str:
        response = self._client.converse(
            modelId=self._model_id,
            messages=messages,
            system=system_blocks,
            inferenceConfig={
                "maxTokens": self._cfg.max_tokens,
                "temperature": self._cfg.temperature,
            },
        )
        self._log_usage(response)
        try:
            return response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"unexpected bedrock response shape: {response!r}"
            ) from exc

    def _log_usage(self, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        if usage:
            logger.info(
                "bedrock usage model=%s in=%s out=%s total=%s",
                self._model_id,
                usage.get("inputTokens"),
                usage.get("outputTokens"),
                usage.get("totalTokens"),
            )
