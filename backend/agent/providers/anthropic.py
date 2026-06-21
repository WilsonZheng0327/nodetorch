"""Native Anthropic (Claude) adapter.

Uses the official `anthropic` SDK with streaming. Defaults to claude-opus-4-8.
"""
from __future__ import annotations

from typing import AsyncIterator

from agent.config import AgentConfig
from agent.providers.base import (
    LLMProvider,
    ProviderCapabilities,
    ProviderError,
    StreamEvent,
    TextDelta,
)
from agent.providers.registry import register

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    capabilities = ProviderCapabilities(
        supports_tools=True, supports_streaming=True, context_window=200_000
    )

    async def stream_chat(
        self, *, system: str, messages: list[dict]
    ) -> AsyncIterator[StreamEvent]:
        from anthropic import AsyncAnthropic, AnthropicError

        key = self.config.resolved_api_key()
        if not key:
            raise ProviderError("Anthropic requires an API key")

        client = AsyncAnthropic(api_key=key, base_url=self.config.base_url or None)
        model = self.config.model or DEFAULT_MODEL
        opts = self.config.options or {}
        max_tokens = opts.get("max_tokens") if isinstance(opts.get("max_tokens"), int) else 4096

        kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield TextDelta(text)
        except AnthropicError as e:
            raise ProviderError(str(e)) from e
        finally:
            await client.close()


register(
    "anthropic",
    AnthropicProvider,
    label="Anthropic (Claude) — native",
    fields=[
        {
            "key": "model",
            "label": "Model",
            "placeholder": f"{DEFAULT_MODEL}  •  claude-sonnet-4-6  •  claude-haiku-4-5",
            "required": False,
        },
        {
            "key": "api_key",
            "label": "API key",
            "placeholder": "sk-ant-…",
            "required": True,
            "secret": True,
        },
    ],
)
