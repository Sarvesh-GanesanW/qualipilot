"""Ollama provider using Ollama's native ``/api/chat`` endpoint.

We hit the native endpoint rather than the OpenAI-compat one because
it returns streaming tokens more reliably and does not require a
fake API key.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from qualipilot.llm._http import is_retryable_http_error
from qualipilot.llm.base import LLMProvider
from qualipilot.models.config import LLMConfig


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, cfg: LLMConfig) -> None:
        self._cfg = cfg
        base = cfg.base_url.rstrip("/")
        # strip trailing /v1 that users copy-paste from OpenAI configs
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._base = base
        self._model = cfg.model

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
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self._cfg.retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=15),
            retry=retry_if_exception(is_retryable_http_error),
        )
        return str(retrying(self._chat, payload))

    def _chat(self, payload: dict[str, Any]) -> str:
        url = f"{self._base}/api/chat"
        with httpx.Client(timeout=self._cfg.timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str) or not content:
            raise RuntimeError("empty response from ollama")
        return content
