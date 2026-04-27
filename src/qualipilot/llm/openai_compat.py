"""OpenAI-compatible provider.

Works against any server that implements the Chat Completions API:
OpenAI itself, Azure OpenAI, vLLM, LiteLLM proxy, LocalAI, etc.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from qualipilot.llm.base import LLMProvider
from qualipilot.models.config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        self._base = cfg.base_url.rstrip("/")
        self._model = cfg.model or "gpt-4o-mini"
        if not cfg.api_key:
            # some open-source servers still require a bearer token
            # even when they do not validate it
            logger.warning(
                "openai-compat provider initialised without api key"
            )

    def generate(self, *, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        return self._post(payload)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException)
        ),
    )
    def _post(self, payload: dict[str, Any]) -> str:
        url = f"{self._base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"

        with httpx.Client(timeout=self._cfg.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"unexpected openai-compat response: {data!r}"
            ) from exc
