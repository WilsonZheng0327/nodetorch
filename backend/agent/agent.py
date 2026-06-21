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
from agent.context import build_messages, build_system_prompt, format_blocks
from agent.providers import create
from agent.providers.base import ExecuteTool
from agent.session import get_session
from agent.tools import GRAPH_TOOLS

logger = logging.getLogger("nodetorch.agent")

OnText = Callable[[str], Awaitable[None]]

# Tools that mean the agent built or extended the graph this turn. After one of
# these, the turn is finished with a deterministic tidy + verify pass (below).
# Purely destructive turns (remove/clear) or property-only tweaks don't qualify —
# we don't reorganize a canvas the user carefully arranged, or re-validate a
# half-torn-down graph.
_BUILD_TOOLS = {"add_node", "connect", "add_block"}

# How many times the finish pass will hand validation errors back to the model to
# auto-repair before giving up and just reporting what's left. Bounded so a model
# that can't fix a problem doesn't loop (and burn tokens) forever.
_MAX_REPAIR_ROUNDS = 2


def _has_issues(report: str) -> bool:
    """True if a validate() report describes problems (anything but an 'ok …')."""
    return bool(report) and not report.lstrip().lower().startswith("ok")


def _repair_prompt(report: str) -> dict:
    """A user turn that feeds validation errors back to the model to fix."""
    return {
        "role": "user",
        "content": (
            "A forward-pass validation of the graph you just edited found problems:\n"
            f"{report}\n\n"
            "Fix them now using the edit tools. Inspect the current state with "
            "get_graph / get_node if you need to, and change only what's necessary — "
            "the nodes you added already exist, so don't re-create them."
        ),
    }


async def run_turn(
    *,
    session_id: str,
    message: str,
    graph: dict | None,
    catalog: list[dict] | None,
    on_text: OnText,
    blocks: list[dict] | None = None,
    execute_tool: ExecuteTool | None = None,
) -> str:
    """Run one turn, streaming text via `on_text`. Returns the full reply.

    If `execute_tool` is given, graph-edit tools are enabled and the provider
    runs the agentic loop. Raises ProviderError if the provider is misconfigured.
    """
    session = get_session(session_id)
    # The frontend sends the catalog + saved blocks once per session; cache them.
    if catalog:
        session.catalog_text = format_catalog(catalog)
    if blocks is not None:
        session.blocks_text = format_blocks(blocks)

    cfg = load_config()
    can_edit = execute_tool is not None
    logger.info(
        "▶ turn [%s] provider=%s model=%s edit=%s | %r",
        session_id, cfg.provider, cfg.model or "(default)", can_edit, message[:140],
    )
    provider = create(cfg)
    system = build_system_prompt(session.catalog_text, session.blocks_text, can_edit=can_edit)
    messages = build_messages(session, message, graph)

    parts: list[str] = []

    async def collect(text: str) -> None:
        parts.append(text)
        await on_text(text)

    # Track which tools the model actually invokes, to decide whether to run the
    # finish pass below. Wrap the bridge so every call is recorded.
    used_tools: set[str] = set()
    bridged = execute_tool
    if execute_tool is not None:
        _raw = execute_tool

        async def bridged(name: str, args: dict, _raw=_raw, _seen=used_tools) -> str:
            _seen.add(name)
            return await _raw(name, args)

    await provider.run(
        system=system,
        messages=messages,
        on_text=collect,
        tools=GRAPH_TOOLS if can_edit else None,
        execute_tool=bridged,
    )

    # If the agent built or extended the graph, finish it cleanly: auto-repair any
    # validation errors, then tidy the layout. Only constructive turns qualify.
    if can_edit and used_tools & _BUILD_TOOLS:
        await _finish_turn(
            provider=provider,
            system=system,
            base_messages=messages,
            collect=collect,
            bridged=bridged,
            execute_tool=execute_tool,
            session_id=session_id,
        )

    answer = "".join(parts)
    logger.info("■ turn [%s] done — %d chars streamed", session_id, len(answer))
    # Store the raw question/answer (no graph blob) for the next turn.
    session.add_user(message)
    session.add_assistant(answer)
    return answer


async def _finish_turn(
    *,
    provider,
    system: str,
    base_messages: list[dict],
    collect: OnText,
    bridged: ExecuteTool,
    execute_tool: ExecuteTool,
    session_id: str,
) -> None:
    """Verify a freshly built graph, auto-repair issues, then tidy the layout.

    Runs only on constructive turns. Flow:
      1. Validate the forward pass.
      2. While it reports problems (up to `_MAX_REPAIR_ROUNDS`), hand the errors
         back to the model as a new user turn and let it make corrective edits.
         Each round is a CONTINUATION, not a rebuild: the graph keeps the edits
         already applied (they live in the browser), and the model inspects the
         live state via get_graph/get_node — it isn't re-building from scratch.
      3. Tidy the layout once, after repairs settle (cosmetic; doesn't affect
         validity), and surface anything that survived the repair budget.

    Best-effort: any failure is logged and swallowed so it can't discard the
    answer already streamed. Cancellation (BaseException) still propagates.
    """
    try:
        report = await execute_tool("validate", {"mode": "forward"})

        rounds = 0
        while _has_issues(report) and rounds < _MAX_REPAIR_ROUNDS:
            rounds += 1
            logger.info("  ↻ repair round %d/%d [%s]", rounds, _MAX_REPAIR_ROUNDS, session_id)
            # Re-enter the model with the validation errors appended to the
            # original conversation. Providers copy `messages`, so base_messages is
            # untouched between rounds — each round starts from the same context
            # plus the latest error report.
            await provider.run(
                system=system,
                messages=base_messages + [_repair_prompt(report)],
                on_text=collect,
                tools=GRAPH_TOOLS,
                execute_tool=bridged,
            )
            report = await execute_tool("validate", {"mode": "forward"})

        await execute_tool("organize_layout", {})

        if _has_issues(report):
            await collect(f"\n\n⚠️ Validation still reports:\n{report}")
    except Exception as e:  # noqa: BLE001 — the finish pass must never break the turn
        logger.warning("finish pass failed [%s]: %s", session_id, e)


async def validate_provider() -> None:
    """Ping the currently configured provider; raise ProviderError on failure."""
    provider = create(load_config())
    await provider.ping()
