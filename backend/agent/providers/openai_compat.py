"""OpenAI-compatible adapter.

One adapter for every endpoint that speaks the OpenAI Chat Completions API:
paid (OpenAI, OpenRouter, Groq, Together, DeepSeek) and local (Ollama at
http://localhost:11434/v1, LM Studio, vLLM, llama.cpp). Configured purely by
base_url + model + api_key.
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


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"
    capabilities = ProviderCapabilities(supports_tools=True, supports_streaming=True)

    def _client(self):
        from openai import AsyncOpenAI

        # Local servers (Ollama/LM Studio) accept any/no key, but the SDK requires
        # a non-empty string — use a placeholder when none is configured.
        api_key = self.config.resolved_api_key() or "sk-no-key-required"
        return AsyncOpenAI(base_url=self.config.base_url or None, api_key=api_key)

    async def stream_chat(
        self, *, system: str, messages: list[dict]
    ) -> AsyncIterator[StreamEvent]:
        if not self.config.model:
            raise ProviderError("No model configured")

        from openai import OpenAIError

        client = self._client()
        full_messages = (
            [{"role": "system", "content": system}] if system else []
        ) + messages

        kwargs: dict = {"model": self.config.model, "messages": full_messages, "stream": True}
        opts = self.config.options or {}
        if isinstance(opts.get("temperature"), (int, float)):
            kwargs["temperature"] = opts["temperature"]
        if isinstance(opts.get("max_tokens"), int):
            kwargs["max_tokens"] = opts["max_tokens"]

        try:
            stream = await client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                text = getattr(chunk.choices[0].delta, "content", None)
                if text:
                    yield TextDelta(text)
        except OpenAIError as e:
            raise ProviderError(str(e)) from e
        finally:
            await client.close()


register(
    "openai_compat",
    OpenAICompatProvider,
    label="OpenAI-compatible",
    fields=[
        {
            "key": "base_url",
            "label": "Base URL",
            "placeholder": "https://api.openai.com/v1  •  http://localhost:11434/v1 (Ollama)",
            "required": False,
        },
        {
            "key": "model",
            "label": "Model",
            "placeholder": "gpt-4o-mini  •  llama3.1  •  qwen2.5",
            "required": True,
        },
        {
            "key": "api_key",
            "label": "API key",
            "placeholder": "sk-…  (leave blank for local models)",
            "required": False,
            "secret": True,
        },
    ],
)
