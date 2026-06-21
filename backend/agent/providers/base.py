"""LLMProvider interface — the contract every adapter implements.

A provider runs one assistant *turn* via `run`. With no tools it just streams
text (explain-only). With tools it runs the agentic loop: stream text, and each
time the model calls a tool, await `execute_tool` and feed the result back,
looping until the model produces a final answer. Each adapter hides its
provider's tool-calling wire format behind this one method.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Awaitable, Callable

from agent.config import AgentConfig


@dataclass
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema for the args."""

    name: str
    description: str
    parameters: dict


# Streams assistant text chunks to the caller.
OnText = Callable[[str], Awaitable[None]]
# Executes a tool call (name, args) and returns an observation string.
ExecuteTool = Callable[[str, dict], Awaitable[str]]


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
    async def run(
        self,
        *,
        system: str,
        messages: list[dict],
        on_text: OnText,
        tools: list[ToolSpec] | None = None,
        execute_tool: ExecuteTool | None = None,
    ) -> None:
        """Run one assistant turn.

        `messages` is OpenAI-style and starts with a user turn. Streams text via
        `on_text`. If `tools` are given, loops: on each tool call, await
        `execute_tool(name, args)` and feed the result back until the model
        finishes with a text answer. Must raise ProviderError on failure.
        """
        raise NotImplementedError

    async def ping(self) -> None:
        """Validate config/connectivity with a minimal request; raise on failure."""
        got = False

        async def mark(_text: str) -> None:
            nonlocal got
            got = True

        await self.run(system="", messages=[{"role": "user", "content": "ping"}], on_text=mark)
        if not got:
            raise ProviderError("No response from provider")
