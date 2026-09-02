"""LLM provider backed by a Ground Zero connection name."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from importlib import import_module
from typing import Any, Literal
from urllib.parse import quote, urlencode, urlsplit

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
    "xai": "getXAIDetails",
}

_CANONICAL_TYPES = {
    "claude": "anthropic",
    "googleai": "gemini",
    "googlegemini": "gemini",
}

_OPENAI_COMPATIBLE = {
    "azureopenai",
    "fireworks",
    "fireworksai",
    "openai",
    "togetherai",
    "xai",
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
        self.resolved_provider = connection_type
        self.resolved_model = (
            _public_text(
                details.get("deploymentName")
                or raw_connection.get("deploymentName"),
                "deploymentName",
                connection_name,
            )
            if connection_type == "azureopenai"
            else _model(details, raw_connection, connection_name)
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
            or "v1",
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
        operation_path = urlsplit(base_url).path.rstrip("/")
        operation = (
            "responses"
            if operation_path.endswith("/responses")
            else (
                "chat/completions"
                if operation_path.endswith("/chat/completions")
                else None
            )
        )
        current_api = api_version in {
            "preview",
            "v1",
        } or operation_path.endswith("/openai/v1")
        if operation is not None:
            completion_url = base_url
        elif current_api:
            current_base = (
                base_url
                if base_url.endswith("/openai/v1")
                else f"{base_url}/openai/v1"
            )
            completion_url = f"{current_base}/chat/completions"
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
            include_model=current_api or operation == "responses",
        )

    defaults = {
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "fireworksai": "https://api.fireworks.ai/inference/v1",
        "openai": "https://api.openai.com/v1",
        "togetherai": "https://api.together.ai/v1",
        "xai": "https://api.x.ai/v1",
    }
    model = _model(
        details,
        raw_connection,
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
    project = details.get("projectId") or raw_connection.get("projectId")
    if project:
        extra_headers["OpenAI-Project"] = _header_text(
            project,
            "projectId",
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
    model = _model(details, raw_connection, connection_name)
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
    assume_role_arn = details.get("assumeRoleArn") or raw_connection.get(
        "assumeRoleArn"
    )
    if assume_role_arn:
        session_kwargs = _assume_role_credentials(
            region,
            session_kwargs,
            _header_text(
                assume_role_arn,
                "assumeRoleArn",
                connection_name,
            ),
            connection_name,
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


def _assume_role_credentials(
    region: str,
    session_kwargs: Mapping[str, str],
    role_arn: str,
    connection_name: str,
) -> dict[str, str]:
    import boto3

    session = boto3.Session(region_name=region, **session_kwargs)
    sts = session.client("sts")
    try:
        response = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName="qualipilot-gz",
        )
    finally:
        with suppress(Exception):
            sts.close()
    credentials = response.get("Credentials")
    if not isinstance(credentials, Mapping):
        raise RuntimeError("STS AssumeRole did not return credentials")
    return {
        "aws_access_key_id": _secret(
            credentials.get("AccessKeyId"),
            "AccessKeyId",
            connection_name,
        ),
        "aws_secret_access_key": _secret(
            credentials.get("SecretAccessKey"),
            "SecretAccessKey",
            connection_name,
        ),
        "aws_session_token": _secret(
            credentials.get("SessionToken"),
            "SessionToken",
            connection_name,
        ),
    }


def _build_native_http_delegate(
    cfg: LLMConfig,
    connection_name: str,
    connection_type: str,
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
) -> LLMProvider:
    model = _model(
        details,
        raw_connection,
        connection_name,
    )
    api_key = _secret(details.get("apiKey"), "apiKey", connection_name)
    defaults = {
        "anthropic": "https://api.anthropic.com",
        "cohere": "https://api.cohere.com",
        "gemini": "https://generativelanguage.googleapis.com/v1",
    }
    extra_headers: dict[str, str] = {}
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
            path = urlsplit(request_url).path.rstrip("/")
            uses_openai_compatibility = path.endswith(
                ("/responses", "/chat/completions")
            ) or not (
                "/models/" in path
                or path.endswith(("/generate", "/generate_stream"))
            )
            if uses_openai_compatibility and not (
                path.endswith(("/v1", "/responses", "/chat/completions"))
            ):
                request_url += "/v1"
        else:
            inference_base = _public_text(
                details.get("inferenceBaseUrl")
                or raw_connection.get("inferenceBaseUrl")
                or "https://router.huggingface.co/v1",
                "inferenceEndpointUrl or inferenceBaseUrl",
                connection_name,
            )
            request_url = _validated_base_url(
                cfg,
                model,
                inference_base,
            )
            uses_openai_compatibility = request_url.endswith("/v1")
            if not uses_openai_compatibility:
                request_url += f"/models/{quote(model, safe='')}"
        if uses_openai_compatibility:
            delegate_cfg = _delegate_config(
                cfg,
                provider="openai",
                model=model,
                base_url=request_url,
                api_key=api_key,
            )
            return OpenAICompatProvider(delegate_cfg)
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
            workspace_id = details.get(
                "anthropicWorkspaceId"
            ) or raw_connection.get("anthropicWorkspaceId")
            if workspace_id:
                extra_headers["anthropic-workspace-id"] = _header_text(
                    workspace_id,
                    "anthropicWorkspaceId",
                    connection_name,
                )
    return _GZNativeHTTPProvider(
        cfg,
        connection_type=connection_type,
        model=model,
        request_url=request_url,
        api_key=api_key,
        api_version=api_version,
        extra_headers=extra_headers,
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
        extra_headers: Mapping[str, str],
    ) -> None:
        self._cfg = cfg
        self._connection_type = connection_type
        self._model = model
        self._request_url = request_url
        self._api_key = SecretStr(api_key)
        self._api_version = api_version
        self._extra_headers = dict(extra_headers)

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
                "messages": [{"role": "user", "content": user}],
            }
            if _anthropic_supports_sampling(self._model):
                payload["temperature"] = self._cfg.temperature
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
        headers.update(self._extra_headers)
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


def _anthropic_supports_sampling(model: str) -> bool:
    parts = model.lower().split("-")
    if len(parts) < 3 or parts[0] != "claude":
        return True
    if parts[1] == "mythos":
        return False
    try:
        major = int(parts[2])
        minor = int(parts[3]) if len(parts) > 3 else 0
    except ValueError:
        return True
    return major < 5 and not (parts[1] == "opus" and major == 4 and minor >= 7)


def _gemini_model_resource(model: str) -> str:
    parts = model.split("/", 1)
    if len(parts) == 2 and parts[0] in {"models", "tunedModels"}:
        return f"{parts[0]}/{quote(parts[1], safe='')}"
    return f"models/{quote(model, safe='')}"


def _model(
    details: Mapping[str, Any],
    raw_connection: Mapping[str, Any],
    connection_name: str,
) -> str:
    return _public_text(
        details.get("model") or raw_connection.get("model"),
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
