"""LLMProvider interface — the contract every adapter implements.

A provider takes an AgentConfig and streams an assistant response. v1 yields
text deltas only; the StreamEvent union is the seam for tool-call events later.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator

from agent.config import AgentConfig


@dataclass
class TextDelta:
    """A chunk of streamed assistant text."""

    text: str


# The streamed-event union. Today it's just text; tool-call events (for the
# future acting scope) will be added here, and the agent loop will match on type.
StreamEvent = TextDelta


@dataclass
class ProviderCapabilities:
    supports_tools: bool = False
    supports_streaming: bool = True
    context_window: int | None = None


class ProviderError(Exception):
    """Raised when a provider call fails (misconfig, auth, or network)."""


class LLMProvider(abc.ABC):
    """Base adapter. Subclasses wrap a concrete SDK/endpoint."""

    name: str = "base"
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def __init__(self, config: AgentConfig):
        self.config = config

    @abc.abstractmethod
    def stream_chat(
        self, *, system: str, messages: list[dict]
    ) -> AsyncIterator[StreamEvent]:
        """Stream the assistant reply as StreamEvent items.

        `messages` is OpenAI-style and starts with a user turn:
        [{"role": "user"|"assistant", "content": str}, ...].
        Implementations must raise ProviderError on failure.
        """
        raise NotImplementedError

    async def ping(self) -> None:
        """Validate config/connectivity with a minimal request; raise on failure."""
        got = False
        async for _ in self.stream_chat(
            system="", messages=[{"role": "user", "content": "ping"}]
        ):
            got = True
            break
        if not got:
            raise ProviderError("No response from provider")
