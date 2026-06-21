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
by wiring nodes on a canvas.

Your job right now is to EXPLAIN: answer the user's questions about their current \
graph and about ML concepts, in clear, student-friendly language. You can see the \
user's current graph (below the question) and the full catalog of node types they \
can use (below).

Guidelines:
- Ground answers in the user's ACTUAL graph — refer to nodes by their id and type \
(e.g. "your conv2d-1 layer").
- When something is wrong or won't train, explain why and describe the fix in \
words (which node/property/connection to change).
- Be concise and concrete. Prefer short paragraphs and tight bullet lists. Use the \
node types and property names from the catalog exactly.
- You can describe changes, but you cannot yet modify the graph yourself — tell the \
user what to do on the canvas."""


def build_system_prompt(catalog_text: str) -> str:
    return (
        _PERSONA
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
