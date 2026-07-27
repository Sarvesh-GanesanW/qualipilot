"""Shared behavior for HTTP-backed LLM providers."""

from __future__ import annotations

import httpx


def is_retryable_http_error(exc: BaseException) -> bool:
    """Retry transport failures, throttling, timeouts, and server errors."""
    if isinstance(exc, httpx.TransportError):
        return True
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    return exc.response.status_code in {408, 429} or (
        500 <= exc.response.status_code < 600
    )
