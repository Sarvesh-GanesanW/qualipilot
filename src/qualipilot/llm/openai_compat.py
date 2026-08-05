"""OpenAI-compatible provider.

Works against any server that implements the Chat Completions API:
OpenAI itself, Azure OpenAI, vLLM, LiteLLM proxy, LocalAI, etc.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
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

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        completion_url: str | None = None,
        auth_header: str = "Authorization",
        auth_scheme: str = "Bearer",
        extra_headers: Mapping[str, str] | None = None,
        include_model: bool = True,
    ) -> None:
        self._cfg = cfg
        self._base = cfg.base_url.rstrip("/")
        self._completion_url = completion_url
        self._auth_header = auth_header
        self._auth_scheme = auth_scheme
        self._extra_headers = dict(extra_headers or {})
        self._include_model = include_model
        self._model = cfg.model
        if not cfg.api_key:
            # some open-source servers still require a bearer token
            # even when they do not validate it
            logger.warning(
                "openai-compat provider initialised without api key"
            )

    def generate(self, *, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._include_model:
            payload["model"] = self._model
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self._cfg.retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=15),
            retry=retry_if_exception(is_retryable_http_error),
        )
        return str(retrying(self._post, payload))

    def _post(self, payload: dict[str, Any]) -> str:
        url = self._completion_url or f"{self._base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        if self._cfg.api_key:
            secret = self._cfg.api_key.get_secret_value()
            headers[self._auth_header] = (
                f"{self._auth_scheme} {secret}"
                if self._auth_scheme
                else secret
            )

        with httpx.Client(timeout=self._cfg.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content:
                raise ValueError("response content is empty")
            return content
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "unexpected openai-compatible response shape"
            ) from exc
