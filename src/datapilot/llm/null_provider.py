"""No-op provider used when ``provider='none'`` is configured."""

from __future__ import annotations

from datapilot.llm.base import LLMProvider


class NullProvider(LLMProvider):
    name = "null"

    def generate(self, *, system: str, user: str) -> str:
        # signature keeps the same shape so callers do not branch
        _ = system, user
        return ""
