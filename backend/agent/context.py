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
- add_node(type, id?, properties?) — add a node. Pass an `id` (e.g. "conv1") to \
name it so you can connect it in the SAME response.
- connect(sourceId, sourcePort, targetId, targetPort) — wire an output port to an \
input port (port ids are in the catalog, e.g. "out" -> "in").
- remove_node(nodeId) — delete a node and its edges.
- remove_edge(sourceId, sourcePort, targetId, targetPort) — disconnect two nodes.
- clear_graph() — remove EVERYTHING (only when the user asks to reset/start over).
- organize_layout() — tidy the node arrangement (cosmetic).
- add_block(filename) — drop a saved block / preset (see "Saved blocks" below).

You can also build and reuse CUSTOM BLOCKS (composite nodes with their own inner graph):
- enter_block(nodeId) — descend INTO a subgraph.block node; add_node / connect / \
set_node_property then edit its INNER graph.
- exit_block() — step back out to the parent canvas.
- save_block(nodeId, name?) — save a subgraph.block to the user's reusable-block library.
To author a reusable block, the flow is: add_node("subgraph.block", id="myblock") → \
enter_block("myblock") → build the inner network with add_node/connect → exit_block() → \
optionally save_block("myblock", name). Prefer add_block when a suitable saved block \
already exists rather than rebuilding it.

You can also READ the graph (no changes):
- get_graph() — re-read the current nodes & edges (the snapshot above can go stale \
after you've made edits).
- get_node(nodeId) — a node's properties, output shape, parameter count, and any error.
- validate(mode) — list exactly what's wrong: mode "forward" (can it run a forward \
pass) or "training" (everything needed to train). Use this to DIAGNOSE before you \
claim something is fixed or ready to train.

IMPORTANT: when the user asks you to build, add, set, change, connect, wire, or \
remove anything, you MUST carry it out by CALLING these tools — do NOT just \
describe the steps in words. Make the edits with tool calls, then give a one-line \
summary of what you changed. Only reply in plain text for questions and \
explanations, never as a substitute for an action the user asked for.

Be efficient — build in as FEW responses as possible:
- To build or extend a network, emit ALL the tool calls together in one response: \
give each add_node a short `id`, then connect them by those ids in the same \
response. You do NOT need to wait for one call before issuing the next. A whole \
small CNN should be one batch of add_node + connect calls.
- Use exact node ids (from the current graph or the ids you just assigned), exact \
type strings and property keys from the catalog, and exact port ids.
- To debug "why won't this work", call validate / get_node and act on what they say.
- If a tool returns an error, read it and fix just that call.
- Don't make changes the user didn't ask for.

When you finish building or extending the graph, NodeTorch automatically runs a \
forward-pass validation and tidies the layout. If validation fails, you'll be asked \
to fix the reported problems — so aim to get it right the first time: call validate \
yourself and FIX issues with set_node_property / connect / get_node before you wrap \
up, rather than relying on that follow-up."""

_EXPLAIN_ONLY = """
You cannot modify the graph yourself — when a change is needed, describe exactly \
what the user should do on the canvas (which node/property/connection)."""


def format_blocks(blocks: list[dict] | None) -> str:
    """Render the user's saved blocks/presets so add_block can reference them."""
    if not blocks:
        return ""
    lines = ["\n\n# Saved blocks (use the filename with add_block)"]
    for b in blocks:
        name = b.get("name") or b.get("filename")
        desc = b.get("description")
        lines.append(f"- {b.get('filename')} — {name}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


def build_system_prompt(catalog_text: str, blocks_text: str = "", can_edit: bool = False) -> str:
    return (
        _PERSONA
        + (_EDIT_INSTRUCTIONS if can_edit else _EXPLAIN_ONLY)
        + "\n\n# Available node types (the catalog)\n"
        + (catalog_text or "(node catalog unavailable)")
        + (blocks_text or "")
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
