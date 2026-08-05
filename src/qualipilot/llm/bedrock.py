"""AWS Bedrock provider using the Converse API.

We use Converse (not InvokeModel) because it unifies request shape
across supported text models, so switching models is a config change.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from qualipilot.llm.base import LLMProvider
from qualipilot.models.config import LLMConfig

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """Bedrock Converse API provider."""

    name = "bedrock"

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        aws_session_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for BedrockProvider; "
                "install with `pip install qualipilot[bedrock]`"
            ) from exc

        self._cfg = cfg
        self._model_id = cfg.model
        session_kwargs: dict[str, Any] = {"region_name": cfg.region}
        if cfg.aws_profile:
            session_kwargs["profile_name"] = cfg.aws_profile
        if aws_session_kwargs:
            session_kwargs.update(aws_session_kwargs)
        session = boto3.Session(**session_kwargs)

        # adaptive retries + keep-alive pooling trims cold-start cost
        boto_config = BotoConfig(
            retries={
                "total_max_attempts": cfg.retries + 1,
                "mode": "adaptive",
            },
            read_timeout=cfg.timeout_seconds,
            connect_timeout=10,
        )
        self._client = session.client("bedrock-runtime", config=boto_config)

    def generate(self, *, system: str, user: str) -> str:
        messages = [
            {"role": "user", "content": [{"text": user}]},
        ]
        system_blocks = [{"text": system}] if system else []

        return self._converse(messages, system_blocks)

    def _converse(
        self,
        messages: list[dict[str, Any]],
        system_blocks: list[dict[str, Any]],
    ) -> str:
        request: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self._cfg.max_tokens,
                "temperature": self._cfg.temperature,
            },
        }
        if system_blocks:
            request["system"] = system_blocks
        response = self._client.converse(**request)
        self._log_usage(response)
        try:
            content = response["output"]["message"]["content"]
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and block["text"]
            ]
            if not texts:
                raise ValueError("response content is empty")
            return "\n".join(texts)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("unexpected bedrock response shape") from exc

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
