"""The agent loop.

One function per turn. With an `execute_tool` bridge (provided by the /agent
WebSocket), the agent can edit the graph: the provider runs the tool-use loop,
each tool call is forwarded to the browser, applied via useGraph, and the result
fed back. Without a bridge it's explain-only (no tools).
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from agent.catalog import format_catalog
from agent.config import load_config
from agent.context import build_messages, build_system_prompt
from agent.providers import create
from agent.providers.base import ExecuteTool
from agent.session import get_session
from agent.tools import GRAPH_TOOLS

logger = logging.getLogger("nodetorch.agent")

OnText = Callable[[str], Awaitable[None]]


async def run_turn(
    *,
    session_id: str,
    message: str,
    graph: dict | None,
    catalog: list[dict] | None,
    on_text: OnText,
    execute_tool: ExecuteTool | None = None,
) -> str:
    """Run one turn, streaming text via `on_text`. Returns the full reply.

    If `execute_tool` is given, graph-edit tools are enabled and the provider
    runs the agentic loop. Raises ProviderError if the provider is misconfigured.
    """
    session = get_session(session_id)
    # The frontend sends the catalog once per session; cache its formatted form.
    if catalog:
        session.catalog_text = format_catalog(catalog)

    cfg = load_config()
    can_edit = execute_tool is not None
    logger.info(
        "▶ turn [%s] provider=%s model=%s edit=%s | %r",
        session_id, cfg.provider, cfg.model or "(default)", can_edit, message[:140],
    )
    provider = create(cfg)
    system = build_system_prompt(session.catalog_text, can_edit=can_edit)
    messages = build_messages(session, message, graph)

    parts: list[str] = []

    async def collect(text: str) -> None:
        parts.append(text)
        await on_text(text)

    await provider.run(
        system=system,
        messages=messages,
        on_text=collect,
        tools=GRAPH_TOOLS if can_edit else None,
        execute_tool=execute_tool,
    )

    answer = "".join(parts)
    logger.info("■ turn [%s] done — %d chars streamed", session_id, len(answer))
    # Store the raw question/answer (no graph blob) for the next turn.
    session.add_user(message)
    session.add_assistant(answer)
    return answer


async def validate_provider() -> None:
    """Ping the currently configured provider; raise ProviderError on failure."""
    provider = create(load_config())
    await provider.ping()
