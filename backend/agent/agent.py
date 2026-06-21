"""The agent loop.

v1 is a single streamed completion (explain-only). It's structured as one
function per turn so a tool-use phase — backend-native tools plus graph-edit
tools bridged to the browser over the WebSocket — slots in later without
reshaping the call sites.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from agent.catalog import format_catalog
from agent.config import load_config
from agent.context import build_messages, build_system_prompt
from agent.providers import create
from agent.providers.base import TextDelta
from agent.session import get_session

OnText = Callable[[str], Awaitable[None]]


async def run_turn(
    *,
    session_id: str,
    message: str,
    graph: dict | None,
    catalog: list[dict] | None,
    on_text: OnText,
) -> str:
    """Run one explain-only turn, streaming text via `on_text`. Returns the full reply.

    Raises ProviderError if the configured provider is unset/misconfigured.
    """
    session = get_session(session_id)
    # The frontend sends the catalog once per session; cache its formatted form.
    if catalog:
        session.catalog_text = format_catalog(catalog)

    provider = create(load_config())
    system = build_system_prompt(session.catalog_text)
    messages = build_messages(session, message, graph)

    parts: list[str] = []
    async for event in provider.stream_chat(system=system, messages=messages):
        if isinstance(event, TextDelta) and event.text:
            parts.append(event.text)
            await on_text(event.text)

    answer = "".join(parts)
    # Store the raw question/answer (no graph blob) for the next turn.
    session.add_user(message)
    session.add_assistant(answer)
    return answer


async def validate_provider() -> None:
    """Ping the currently configured provider; raise ProviderError on failure."""
    provider = create(load_config())
    await provider.ping()
