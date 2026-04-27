"""Ollama provider using Ollama's native ``/api/chat`` endpoint.

We hit the native endpoint rather than the OpenAI-compat one because
it returns streaming tokens more reliably and does not require a
fake API key.
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


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        base = cfg.base_url.rstrip("/")
        # strip trailing /v1 that users copy-paste from OpenAI configs
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._base = base
        self._model = cfg.model or "llama3.2:latest"

    def generate(self, *, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": self._cfg.temperature,
                "num_predict": self._cfg.max_tokens,
            },
        }
        return self._chat(payload)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException)
        ),
    )
    def _chat(self, payload: dict[str, Any]) -> str:
        url = f"{self._base}/api/chat"
        with httpx.Client(timeout=self._cfg.timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content", "")
        if not content:
            raise RuntimeError(f"empty response from ollama: {data!r}")
        return str(content)
