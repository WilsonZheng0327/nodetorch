"""OpenAI-compatible adapter.

One adapter for every endpoint that speaks the OpenAI Chat Completions API:
paid (OpenAI, OpenRouter, Groq, Together, DeepSeek) and local (Ollama at
http://localhost:11434/v1, LM Studio, vLLM, llama.cpp). Configured purely by
base_url + model + api_key.
"""
from __future__ import annotations

import json
import logging

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

logger = logging.getLogger("nodetorch.agent")

# Substrings that mark a malformed-tool-call rejection (model emitted a tool call
# the provider couldn't parse/validate). Weaker models (e.g. Llama on Groq) hit
# these intermittently; we nudge and retry rather than surfacing the raw error.
_MALFORMED_TOOL_ERRORS = (
    "failed to call a function",
    "tool call validation failed",
    "not in request.tools",
    "invalid tool call",
)


def _is_malformed_tool_error(text: str) -> bool:
    low = text.lower()
    return any(s in low for s in _MALFORMED_TOOL_ERRORS)


def _failed_generation(e) -> str | None:
    """The raw (malformed) text the model produced — Groq returns it in
    error.failed_generation on a rejected tool call. Robust to where the SDK
    surfaces it (e.body, nested under 'error', or on the raw response)."""
    candidates: list = []
    body = getattr(e, "body", None)
    candidates.append(body)
    if isinstance(body, dict):
        candidates.append(body.get("error"))
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            j = resp.json()
            candidates.append(j)
            if isinstance(j, dict):
                candidates.append(j.get("error"))
        except Exception:
            pass
    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("failed_generation"), str):
            return c["failed_generation"]
    return None


class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"
    capabilities = ProviderCapabilities(supports_tools=True, supports_streaming=True)

    def _client(self):
        from openai import AsyncOpenAI

        # Local servers (Ollama/LM Studio) accept any/no key, but the SDK requires
        # a non-empty string — use a placeholder when none is configured.
        api_key = self.config.resolved_api_key() or "sk-no-key-required"
        return AsyncOpenAI(base_url=self.config.base_url or None, api_key=api_key)

    async def run(
        self,
        *,
        system: str,
        messages: list[dict],
        on_text: OnText,
        tools: list[ToolSpec] | None = None,
        execute_tool: ExecuteTool | None = None,
    ) -> None:
        if not self.config.model:
            raise ProviderError("No model configured")

        from openai import OpenAIError

        client = self._client()
        convo = ([{"role": "system", "content": system}] if system else []) + list(messages)
        oa_tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in (tools or [])
        ]
        opts = self.config.options or {}
        nudges_left = 2  # corrective retries for malformed tool calls

        turn_prompt = 0
        turn_completion = 0
        try:
            while True:
                kwargs: dict = {
                    "model": self.config.model,
                    "messages": convo,
                    "stream": True,
                    "stream_options": {"include_usage": True},  # report token usage per request
                }
                if oa_tools:
                    kwargs["tools"] = oa_tools
                    # Parallel tool calls are allowed (the model names new nodes via
                    # add_node's `id`, so it can add + connect in one response — far
                    # fewer round-trips than one call per request).
                # Default to temperature 0: weaker models are far more reliable at
                # tool selection deterministically. Overridable via options.
                temp = opts.get("temperature")
                kwargs["temperature"] = temp if isinstance(temp, (int, float)) else 0
                if isinstance(opts.get("max_tokens"), int):
                    kwargs["max_tokens"] = opts["max_tokens"]

                logger.debug(
                    "→ model request: model=%s tools=%d messages=%d temp=%s",
                    self.config.model, len(oa_tools), len(convo), kwargs.get("temperature"),
                )
                logger.debug("  request messages: %s", json.dumps(convo)[:4000])

                text_parts: list[str] = []
                calls: dict[int, dict] = {}  # index -> {id, name, args}
                usage = None
                try:
                    stream = await client.chat.completions.create(**kwargs)
                    async for chunk in stream:
                        if getattr(chunk, "usage", None):
                            usage = chunk.usage  # final usage-only chunk (empty choices)
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if getattr(delta, "content", None):
                            text_parts.append(delta.content)
                            await on_text(delta.content)
                        for tc in getattr(delta, "tool_calls", None) or []:
                            slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id:
                                slot["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    slot["name"] = tc.function.name
                                if tc.function.arguments:
                                    slot["args"] += tc.function.arguments
                    if usage:
                        turn_prompt += usage.prompt_tokens or 0
                        turn_completion += usage.completion_tokens or 0
                        logger.info(
                            "  📊 tokens +%s in / +%s out  (turn total: %d in / %d out)",
                            usage.prompt_tokens, usage.completion_tokens, turn_prompt, turn_completion,
                        )
                except OpenAIError as e:
                    # Some providers (e.g. Groq + Llama) reject a malformed tool call
                    # server-side. Nudge the model to re-issue a clean call and retry.
                    if _is_malformed_tool_error(str(e)) and nudges_left > 0:
                        nudges_left -= 1
                        logger.warning("⚠ malformed tool call; nudging + retry (%d left): %s", nudges_left, str(e)[:160])
                        logger.debug("   error.body=%r", getattr(e, "body", None))
                        fg = _failed_generation(e)
                        if fg:
                            logger.warning("   ↳ model actually emitted:\n%s", fg[:1500])
                        convo.append({
                            "role": "user",
                            "content": "(Your previous tool call was rejected as malformed. Make ONE tool call with the function name and its arguments as separate, valid JSON that exactly matches the tool's schema — correct field names and value types. Do not put arguments inside the function name.)",
                        })
                        continue
                    raise

                if not calls:
                    return  # final answer streamed

                if execute_tool is None:
                    raise ProviderError("Model requested a tool but tool execution is unavailable")

                # Echo the assistant's tool calls, then append each tool result.
                convo.append(
                    {
                        "role": "assistant",
                        "content": "".join(text_parts) or None,
                        "tool_calls": [
                            {"id": s["id"], "type": "function", "function": {"name": s["name"], "arguments": s["args"] or "{}"}}
                            for s in calls.values()
                        ],
                    }
                )
                for s in calls.values():
                    try:
                        args = json.loads(s["args"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await execute_tool(s["name"], args)
                    convo.append({"role": "tool", "tool_call_id": s["id"], "content": result})
        except OpenAIError as e:
            if _is_malformed_tool_error(str(e)):
                fg = _failed_generation(e)
                if fg:
                    logger.warning("✗ tool call still malformed after retries. Model emitted:\n%s", fg[:1500])
                raise ProviderError(
                    "The model produced an invalid tool call (some models do this on complex "
                    "requests). Try rephrasing or splitting the request into smaller steps, or "
                    "use a more capable model."
                ) from e
            raise ProviderError(str(e)) from e
        finally:
            await client.close()


register(
    "openai_compat",
    OpenAICompatProvider,
    label="OpenAI-compatible (OpenAI, OpenRouter, Ollama, LM Studio, vLLM…)",
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
