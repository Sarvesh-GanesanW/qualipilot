"""Common protocol for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base every concrete provider implements."""

    name: str

    @abstractmethod
    def generate(self, *, system: str, user: str) -> str:
        """Return the model's answer as a single string.

        Args:
            system: System prompt / role prompt.
            user: User message content.

        Returns:
            Raw model text. Callers decide how to render it.
        """
