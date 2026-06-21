"""Native Anthropic (Claude) adapter.

Uses the official `anthropic` SDK with streaming + tool use. Defaults to
claude-opus-4-8.
"""
from __future__ import annotations

from agent.config import AgentConfig
from agent.providers.base import (
    ExecuteTool,
    LLMProvider,
    OnText,
    ProviderCapabilities,
    ProviderError,
    ToolSpec,
)
from agent.providers.registry import register

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    capabilities = ProviderCapabilities(
        supports_tools=True, supports_streaming=True, context_window=200_000
    )

    async def run(
        self,
        *,
        system: str,
        messages: list[dict],
        on_text: OnText,
        tools: list[ToolSpec] | None = None,
        execute_tool: ExecuteTool | None = None,
    ) -> None:
        from anthropic import AsyncAnthropic, AnthropicError

        key = self.config.resolved_api_key()
        if not key:
            raise ProviderError("Anthropic requires an API key")

        client = AsyncAnthropic(api_key=key, base_url=self.config.base_url or None)
        model = self.config.model or DEFAULT_MODEL
        opts = self.config.options or {}
        max_tokens = opts.get("max_tokens") if isinstance(opts.get("max_tokens"), int) else 4096

        convo = list(messages)
        an_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in (tools or [])
        ]

        try:
            while True:
                kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": convo}
                if system:
                    kwargs["system"] = system
                if an_tools:
                    kwargs["tools"] = an_tools

                async with client.messages.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                            if event.delta.text:
                                await on_text(event.delta.text)
                    final = await stream.get_final_message()

                tool_uses = [b for b in final.content if b.type == "tool_use"]
                if not tool_uses:
                    return  # final answer streamed

                if execute_tool is None:
                    raise ProviderError("Model requested a tool but tool execution is unavailable")

                # Echo the assistant's content (incl. tool_use blocks), then the results.
                convo.append({"role": "assistant", "content": final.content})
                results = []
                for tu in tool_uses:
                    result = await execute_tool(tu.name, tu.input)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})
                convo.append({"role": "user", "content": results})
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
