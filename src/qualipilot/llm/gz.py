"""LLM provider backed by a Ground Zero connection name."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Literal
from urllib.parse import quote, urlencode

import httpx
from pydantic import SecretStr
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from qualipilot.llm._http import is_retryable_http_error
from qualipilot.llm.base import LLMProvider
from qualipilot.llm.bedrock import BedrockProvider
from qualipilot.llm.openai_compat import OpenAICompatProvider
from qualipilot.models.config import LLMConfig

_GETTERS = {
    "anthropic": "getAnthropicDetails",
    "awsbedrock": "getAwsBedrockDetails",
    "azureopenai": "getAzureOpenAIDetails",
    "claude": "getAnthropicDetails",
    "cohere": "getCohereDetails",
    "fireworks": "getFireworksAIDetails",
    "fireworksai": "getFireworksAIDetails",
    "gemini": "getGeminiDetails",
    "googleai": "getGeminiDetails",
    "googlegemini": "getGeminiDetails",
    "huggingface": "getHuggingFaceDetails",
    "openai": "getOpenAIDetails",
    "togetherai": "getTogetherAIDetails",
}

_CANONICAL_TYPES = {
    "claude": "anthropic",
    "googleai": "gemini",
    "googlegemini": "gemini",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "cohere": "command-a-03-2025",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "togetherai": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

_OPENAI_COMPATIBLE = {
    "azureopenai",
    "fireworks",
    "fireworksai",
    "openai",
    "togetherai",
}


class GZConnectionProvider(LLMProvider):
    """Resolve one GZ connection and delegate without retaining its details."""

    name = "gz"

    def __init__(self, cfg: LLMConfig) -> None:
        connection_name = cfg.connection_name
        if connection_name is None:
            raise ValueError("gz requires connection_name")

        try:
            connections = import_module("GZ.CONNECTIONS")
        except ImportError as exc:
            raise ImportError(
                "GZ.CONNECTIONS is required for provider='gz'; install "
                "gz-base-rte-spark or gz-base-rte-duckdb"
            ) from exc

        registry = getattr(connections, "conn", None)
        if not isinstance(registry, Mapping):
            raise RuntimeError("GZ.CONNECTIONS.conn is not available")
        raw_connection = registry.get(connection_name)
        if not isinstance(raw_connection, Mapping):
            raise ValueError(
                f"GZ connection '{connection_name}' was not found"
            )

        raw_type = str(raw_connection.get("type") or "").strip()
        connection_type = _normalize_type(raw_type)
        getter_name = _GETTERS.get(connection_type)
        if getter_name is None:
            supported = ", ".join(sorted(_GETTERS))
            raise ValueError(
                f"GZ connection '{connection_name}' has unsupported LLM "
                f"type '{raw_type}'. Supported types: {supported}"
            )
        getter = getattr(connections, getter_name, None)
        if not callable(getter):
            raise RuntimeError(
                f"GZ.CONNECTIONS does not provide {getter_name}"
            )
        details = getter(connection_name)
        if not isinstance(details, Mapping):
            raise RuntimeError(f"{getter_name} did not return a mapping")
        connection_type = _CANONICAL_TYPES.get(
            connection_type, connection_type
        )

        self._delegate = _build_delegate(
            cfg,
            connection_name,
            connection_type,
            details,
            raw_connection,
        )

    def generate(self, *, system: str, user: str) -> str:
        return self._delegate.generate(system=system, user=user)


def _normalize_type(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _build_delegate(
    cfg: LLMConfig,
    connection_name: str,
    connection_type: str,
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
) -> LLMProvider:
    if connection_type in _OPENAI_COMPATIBLE:
        return _build_openai_delegate(
            cfg,
            connection_name,
            connection_type,
            details,
            raw_connection,
        )
    if connection_type == "awsbedrock":
        return _build_bedrock_delegate(
            cfg,
            connection_name,
            details,
            raw_connection,
        )
    return _build_native_http_delegate(
        cfg,
        connection_name,
        connection_type,
        details,
        raw_connection,
    )


def _build_openai_delegate(
    cfg: LLMConfig,
    connection_name: str,
    connection_type: str,
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
) -> OpenAICompatProvider:
    api_key = _secret(details.get("apiKey"), "apiKey", connection_name)
    if connection_type == "azureopenai":
        endpoint = _public_text(
            details.get("endpoint") or raw_connection.get("endpoint"),
            "endpoint",
            connection_name,
        )
        deployment = _public_text(
            details.get("deploymentName")
            or raw_connection.get("deploymentName"),
            "deploymentName",
            connection_name,
        )
        api_version = _header_text(
            details.get("apiVersion")
            or raw_connection.get("apiVersion")
            or "2024-10-21",
            "apiVersion",
            connection_name,
        )
        delegate_cfg = _delegate_config(
            cfg,
            provider="openai",
            model=deployment,
            base_url=endpoint,
            api_key=api_key,
        )
        base_url = delegate_cfg.base_url.rstrip("/")
        current_api = api_version in {"preview", "v1"}
        if current_api:
            completion_url = f"{base_url}/openai/v1/chat/completions"
            if api_version == "preview":
                completion_url += "?" + urlencode({"api-version": "preview"})
        else:
            encoded_deployment = quote(deployment, safe="")
            completion_url = (
                f"{base_url}/openai/deployments/{encoded_deployment}"
                f"/chat/completions?{urlencode({'api-version': api_version})}"
            )
        return OpenAICompatProvider(
            delegate_cfg,
            completion_url=completion_url,
            auth_header="api-key",
            auth_scheme="",
            include_model=current_api,
        )

    defaults = {
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "fireworksai": "https://api.fireworks.ai/inference/v1",
        "openai": "https://api.openai.com/v1",
        "togetherai": "https://api.together.xyz/v1",
    }
    model = _model(
        details,
        raw_connection,
        _DEFAULT_MODELS.get(connection_type),
        connection_name,
    )
    base_url = _public_text(
        details.get("baseUrl")
        or raw_connection.get("baseUrl")
        or defaults[connection_type],
        "baseUrl",
        connection_name,
    )
    delegate_cfg = _delegate_config(
        cfg,
        provider="openai",
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    extra_headers: dict[str, str] = {}
    organization = details.get("organizationId") or raw_connection.get(
        "organizationId"
    )
    if organization:
        extra_headers["OpenAI-Organization"] = _header_text(
            organization,
            "organizationId",
            connection_name,
        )
    return OpenAICompatProvider(
        delegate_cfg,
        extra_headers=extra_headers,
    )


def _build_bedrock_delegate(
    cfg: LLMConfig,
    connection_name: str,
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
) -> BedrockProvider:
    model = _model(details, raw_connection, None, connection_name)
    region = _public_text(
        details.get("region") or raw_connection.get("region") or "us-east-1",
        "region",
        connection_name,
    )
    access_key = details.get("accessKey")
    secret_key = details.get("secretAccessKey")
    session_token = details.get("sessionToken")
    if bool(access_key) != bool(secret_key):
        raise ValueError(
            f"GZ connection '{connection_name}' must provide both accessKey "
            "and secretAccessKey"
        )
    if session_token and not access_key:
        raise ValueError(
            f"GZ connection '{connection_name}' cannot use sessionToken "
            "without accessKey and secretAccessKey"
        )

    session_kwargs: dict[str, str] = {}
    if access_key:
        session_kwargs["aws_access_key_id"] = _secret(
            access_key, "accessKey", connection_name
        )
        session_kwargs["aws_secret_access_key"] = _secret(
            secret_key, "secretAccessKey", connection_name
        )
    if session_token:
        session_kwargs["aws_session_token"] = _secret(
            session_token, "sessionToken", connection_name
        )
    delegate_cfg = _delegate_config(
        cfg,
        provider="bedrock",
        model=model,
        region=region,
    )
    return BedrockProvider(
        delegate_cfg,
        aws_session_kwargs=session_kwargs,
    )


def _build_native_http_delegate(
    cfg: LLMConfig,
    connection_name: str,
    connection_type: str,
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
) -> _GZNativeHTTPProvider:
    model = _model(
        details,
        raw_connection,
        _DEFAULT_MODELS.get(connection_type),
        connection_name,
    )
    api_key = _secret(details.get("apiKey"), "apiKey", connection_name)
    defaults = {
        "anthropic": "https://api.anthropic.com",
        "cohere": "https://api.cohere.com",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
    }
    if connection_type == "huggingface":
        endpoint = details.get("inferenceEndpointUrl") or raw_connection.get(
            "inferenceEndpointUrl"
        )
        if endpoint:
            request_url = _validated_base_url(
                cfg,
                model,
                _public_text(
                    endpoint, "inferenceEndpointUrl", connection_name
                ),
            )
        else:
            inference_base = (
                details.get("inferenceBaseUrl")
                or raw_connection.get("inferenceBaseUrl")
                or "https://api-inference.huggingface.co"
            )
            base_url = _validated_base_url(
                cfg,
                model,
                _public_text(
                    inference_base,
                    "inferenceBaseUrl",
                    connection_name,
                ),
            )
            request_url = f"{base_url}/models/{quote(model, safe='')}"
        api_version = ""
    else:
        base_url = _validated_base_url(
            cfg,
            model,
            _public_text(
                details.get("baseUrl")
                or raw_connection.get("baseUrl")
                or defaults[connection_type],
                "baseUrl",
                connection_name,
            ),
        )
        paths = {
            "anthropic": "/v1/messages",
            "cohere": "/v2/chat",
            "gemini": (f"/{_gemini_model_resource(model)}:generateContent"),
        }
        request_url = base_url + paths[connection_type]
        api_version = ""
        if connection_type == "anthropic":
            api_version = _header_text(
                details.get("anthropicVersion")
                or raw_connection.get("anthropicVersion")
                or "2023-06-01",
                "anthropicVersion",
                connection_name,
            )
    return _GZNativeHTTPProvider(
        cfg,
        connection_type=connection_type,
        model=model,
        request_url=request_url,
        api_key=api_key,
        api_version=api_version,
    )


class _GZNativeHTTPProvider(LLMProvider):
    name = "gz_http"

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        connection_type: str,
        model: str,
        request_url: str,
        api_key: str,
        api_version: str,
    ) -> None:
        self._cfg = cfg
        self._connection_type = connection_type
        self._model = model
        self._request_url = request_url
        self._api_key = SecretStr(api_key)
        self._api_version = api_version

    def generate(self, *, system: str, user: str) -> str:
        payload = self._payload(system, user)
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self._cfg.retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=15),
            retry=retry_if_exception(is_retryable_http_error),
        )
        response = retrying(self._post, payload)
        return self._response_text(response)

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        if self._connection_type == "anthropic":
            payload: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._cfg.max_tokens,
                "temperature": self._cfg.temperature,
                "messages": [{"role": "user", "content": user}],
            }
            if system:
                payload["system"] = system
            return payload
        if self._connection_type == "cohere":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            return {
                "model": self._model,
                "messages": messages,
                "max_tokens": self._cfg.max_tokens,
                "temperature": self._cfg.temperature,
            }
        if self._connection_type == "gemini":
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "maxOutputTokens": self._cfg.max_tokens,
                    "temperature": self._cfg.temperature,
                },
            }
            if system:
                payload["system_instruction"] = {"parts": [{"text": system}]}
            return payload
        prompt = f"{system}\n\n{user}" if system else user
        return {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self._cfg.max_tokens,
                "return_full_text": False,
                "temperature": self._cfg.temperature,
            },
        }

    def _post(self, payload: dict[str, Any]) -> Any:
        api_key = self._api_key.get_secret_value()
        headers = {"Content-Type": "application/json"}
        if self._connection_type == "anthropic":
            headers.update(
                {
                    "anthropic-version": self._api_version,
                    "x-api-key": api_key,
                }
            )
        elif self._connection_type == "gemini":
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        with httpx.Client(timeout=self._cfg.timeout_seconds) as client:
            response = client.post(
                self._request_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    def _response_text(self, response: Any) -> str:
        text = ""
        if self._connection_type == "anthropic" and isinstance(
            response, Mapping
        ):
            text = _join_text(response.get("content"))
        elif self._connection_type == "cohere" and isinstance(
            response, Mapping
        ):
            message = response.get("message")
            if isinstance(message, Mapping):
                text = _join_text(message.get("content"))
        elif self._connection_type == "gemini" and isinstance(
            response, Mapping
        ):
            candidates = response.get("candidates")
            if isinstance(candidates, list) and candidates:
                candidate = candidates[0]
                if isinstance(candidate, Mapping):
                    content = candidate.get("content")
                    if isinstance(content, Mapping):
                        text = _join_text(content.get("parts"))
        else:
            item = (
                response[0]
                if isinstance(response, list) and response
                else response
            )
            if isinstance(item, Mapping):
                text = str(
                    item.get("generated_text")
                    or item.get("summary_text")
                    or item.get("translation_text")
                    or item.get("text")
                    or ""
                )
        if not text:
            raise RuntimeError(
                f"unexpected {self._connection_type} response shape"
            )
        return text


def _join_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "".join(
        str(item.get("text") or "")
        for item in value
        if isinstance(item, Mapping)
    )


def _gemini_model_resource(model: str) -> str:
    parts = model.split("/", 1)
    if len(parts) == 2 and parts[0] in {"models", "tunedModels"}:
        return f"{parts[0]}/{quote(parts[1], safe='')}"
    return f"models/{quote(model, safe='')}"


def _model(
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
    default: str | None,
    connection_name: str,
) -> str:
    return _public_text(
        details.get("model") or raw_connection.get("model") or default,
        "model",
        connection_name,
    )


def _public_text(value: Any, field: str, connection_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"GZ connection '{connection_name}' requires {field}")
    return text


def _header_text(value: Any, field: str, connection_name: str) -> str:
    text = _public_text(value, field, connection_name)
    if "\r" in text or "\n" in text:
        raise ValueError(
            f"GZ connection '{connection_name}' has invalid {field}"
        )
    return text


def _secret(value: Any, field: str, connection_name: str) -> str:
    text = str(value or "")
    if not text.strip() or "\r" in text or "\n" in text:
        raise ValueError(
            f"GZ connection '{connection_name}' requires a valid {field}"
        )
    return text


def _validated_base_url(cfg: LLMConfig, model: str, base_url: str) -> str:
    return _delegate_config(
        cfg,
        provider="openai",
        model=model,
        base_url=base_url,
    ).base_url.rstrip("/")


def _delegate_config(
    cfg: LLMConfig,
    *,
    provider: Literal["bedrock", "openai"],
    model: str,
    base_url: str = "http://localhost:11434/v1",
    api_key: str | None = None,
    region: str = "us-east-1",
) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        model=model,
        region=region,
        base_url=base_url,
        api_key=SecretStr(api_key) if api_key is not None else None,
        allow_insecure_http=cfg.allow_insecure_http,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout_seconds=cfg.timeout_seconds,
        retries=cfg.retries,
        system_prompt=cfg.system_prompt,
    )
