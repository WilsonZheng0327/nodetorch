"""Context assembly: system prompt, compact graph summary, and per-turn messages.

Caching-friendly layout: the system prompt (persona + node catalog) is stable
across turns; the volatile bit (the current graph) is embedded in the latest
user turn, not the system prompt. History stores raw question/answer text only
(no graph blobs) to stay token-light.
"""
from __future__ import annotations

from agent.session import AgentSession

# Soft cap so a huge architecture (e.g. ResNet-34) can't blow the context window.
_MAX_NODES_IN_SUMMARY = 120

_PERSONA = """You are NodeTorch Assistant, an AI tutor embedded in NodeTorch — a \
node-based visual tool where students build and inspect machine-learning models \
by wiring nodes on a canvas. You can see the user's current graph (below the \
question) and the full catalog of node types (below).

Guidelines:
- Ground everything in the user's ACTUAL graph — refer to nodes by their id and \
type (e.g. "your conv2d-1 layer").
- Be concise and concrete. Prefer short paragraphs and tight bullet lists. Use \
node types and property names from the catalog EXACTLY."""

_EDIT_INSTRUCTIONS = """
You can EDIT the graph by CALLING TOOLS:
- set_node_property(nodeId, key, value) — change a node's property.
- add_node(type, properties?) — add a node; it returns the new node's id, which \
you use to connect it.
- connect(sourceId, sourcePort, targetId, targetPort) — wire an output port to an \
input port (port ids are in the catalog, e.g. "out" -> "in").
- remove_node(nodeId) — delete a node and its edges.

IMPORTANT: when the user asks you to build, add, set, change, connect, wire, or \
remove anything, you MUST carry it out by CALLING these tools — do NOT just \
describe the steps in words. Make the edits with tool calls, then give a one-line \
summary of what you changed. Only reply in plain text for questions and \
explanations, never as a substitute for an action the user asked for.

Rules:
- Use exact node ids from the current graph, exact type strings and property keys \
from the catalog, and exact port ids. Add a node before connecting it (use the id \
the add_node call returns).
- If a tool returns an error, read it and fix the call.
- Don't make changes the user didn't ask for."""

_EXPLAIN_ONLY = """
You cannot modify the graph yourself — when a change is needed, describe exactly \
what the user should do on the canvas (which node/property/connection)."""


def build_system_prompt(catalog_text: str, can_edit: bool = False) -> str:
    return (
        _PERSONA
        + (_EDIT_INSTRUCTIONS if can_edit else _EXPLAIN_ONLY)
        + "\n\n# Available node types (the catalog)\n"
        + (catalog_text or "(node catalog unavailable)")
    )


def _node_graph(graph: dict) -> dict:
    """Accept either a full SerializedGraph or the inner graph object."""
    if isinstance(graph, dict) and "graph" in graph and isinstance(graph["graph"], dict):
        return graph["graph"]
    return graph or {}


def _format_props(props: dict) -> str:
    if not props:
        return ""
    items = ", ".join(f"{k}={v}" for k, v in props.items() if v not in (None, ""))
    return f"  props: {items}" if items else ""


def summarize_graph(graph: dict) -> str:
    """Compact node/edge listing — types, key props, and connections (no positions)."""
    g = _node_graph(graph)
    nodes = g.get("nodes", []) or []
    edges = g.get("edges", []) or []
    if not nodes:
        return "(the canvas is empty — no nodes yet)"

    lines: list[str] = []
    name = g.get("name")
    if name:
        lines.append(f"Graph: {name}")

    lines.append(f"Nodes ({len(nodes)}):")
    for node in nodes[:_MAX_NODES_IN_SUMMARY]:
        line = f"- {node.get('id')} ({node.get('type')})"
        sub = node.get("subgraph")
        if sub:
            inner = len(sub.get("nodes", []) or [])
            block_name = (node.get("properties") or {}).get("blockName", "")
            line += f"  [custom block{': ' + block_name if block_name else ''}, {inner} inner nodes]"
        props = _format_props(node.get("properties") or {})
        if props:
            line += props
        lines.append(line)
    if len(nodes) > _MAX_NODES_IN_SUMMARY:
        lines.append(f"  … and {len(nodes) - _MAX_NODES_IN_SUMMARY} more nodes")

    if edges:
        lines.append(f"Edges ({len(edges)}):")
        for e in edges[:_MAX_NODES_IN_SUMMARY]:
            s, t = e.get("source", {}), e.get("target", {})
            lines.append(
                f"- {s.get('nodeId')}.{s.get('portId')} -> {t.get('nodeId')}.{t.get('portId')}"
            )
        if len(edges) > _MAX_NODES_IN_SUMMARY:
            lines.append(f"  … and {len(edges) - _MAX_NODES_IN_SUMMARY} more edges")

    return "\n".join(lines)


def build_messages(session: AgentSession, user_message: str, graph: dict | None) -> list[dict]:
    """History + a final user turn that embeds the current graph snapshot."""
    messages = list(session.history)
    graph_block = ""
    if graph is not None:
        graph_block = "[Current graph]\n" + summarize_graph(graph) + "\n\n"
    messages.append({"role": "user", "content": graph_block + "[Question]\n" + user_message})
    return messages
