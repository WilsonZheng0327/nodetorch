"""Graph-edit tools the agent can call.

These are the schemas the model sees; the backend does NOT execute them — each
call is bridged to the browser over the /agent WebSocket and applied via the
frontend's useGraph actions (the same code the inspector/palette use). The
catalog the agent already receives tells it the valid node types, property keys,
and port ids to fill these arguments.
"""
from __future__ import annotations

from agent.providers.base import ToolSpec

GRAPH_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="set_node_property",
        description=(
            "Change one property of an existing node on the canvas. Use the exact "
            "node id from the current graph and a property key from the catalog "
            "(e.g. outChannels, kernelSize)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "nodeId": {"type": "string", "description": "id of the node to edit"},
                "key": {"type": "string", "description": "property key, e.g. outChannels"},
                "value": {
                    "type": ["string", "number", "boolean"],
                    "description": "new value; must match the property's type",
                },
            },
            "required": ["nodeId", "key", "value"],
        },
    ),
    ToolSpec(
        name="add_node",
        description=(
            "Add a new node to the canvas. `type` must be a node type string from the "
            "catalog (e.g. ml.layers.conv2d). Pass `id` to name the node yourself so "
            "you can connect it in the SAME response (otherwise an id is generated and "
            "returned). Returns the node's actual id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "node type from the catalog"},
                "id": {
                    "type": "string",
                    "description": "optional unique id to give the node, e.g. 'conv1' — use it in connect() in the same response",
                },
                "properties": {
                    "type": "object",
                    "description": "optional initial properties; defaults are used otherwise",
                },
            },
            "required": ["type"],
        },
    ),
    ToolSpec(
        name="connect",
        description=(
            "Connect an output port of one node to an input port of another. Port "
            "ids come from the catalog (e.g. 'out' -> 'in')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sourceId": {"type": "string"},
                "sourcePort": {"type": "string", "description": "output port id, e.g. out"},
                "targetId": {"type": "string"},
                "targetPort": {"type": "string", "description": "input port id, e.g. in"},
            },
            "required": ["sourceId", "sourcePort", "targetId", "targetPort"],
        },
    ),
    ToolSpec(
        name="remove_node",
        description="Delete a node (and its connected edges) from the canvas.",
        parameters={
            "type": "object",
            "properties": {"nodeId": {"type": "string"}},
            "required": ["nodeId"],
        },
    ),
    ToolSpec(
        name="remove_edge",
        description="Disconnect a wire between two nodes (the edge from sourceId.sourcePort to targetId.targetPort).",
        parameters={
            "type": "object",
            "properties": {
                "sourceId": {"type": "string"},
                "sourcePort": {"type": "string"},
                "targetId": {"type": "string"},
                "targetPort": {"type": "string"},
            },
            "required": ["sourceId", "sourcePort", "targetId", "targetPort"],
        },
    ),
    ToolSpec(
        name="clear_graph",
        description="Remove ALL nodes and edges from the canvas (start from a blank graph). Destructive — only when the user asks to clear/reset/start over.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="organize_layout",
        description="Auto-arrange the nodes into a clean left-to-right layout. Purely cosmetic.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="add_block",
        description="Add a saved block / preset (a reusable sub-graph) to the canvas by its filename. Use a filename from the 'Saved blocks' list in the system prompt.",
        parameters={
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    ),
    ToolSpec(
        name="enter_block",
        description=(
            "Descend INTO a custom block (a subgraph.block node) so that subsequent "
            "add_node / connect / set_node_property calls edit its INNER graph instead "
            "of the main canvas. To build a reusable block: add_node('subgraph.block', "
            "id='myblock'), enter_block('myblock'), build the inner network, then "
            "exit_block(). The block's input/output ports are the inner io nodes."
        ),
        parameters={
            "type": "object",
            "properties": {"nodeId": {"type": "string", "description": "id of the subgraph.block node to enter"}},
            "required": ["nodeId"],
        },
    ),
    ToolSpec(
        name="exit_block",
        description="Step back out to the parent canvas after editing a custom block's inner graph (undoes one enter_block).",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="save_block",
        description=(
            "Save a custom block (a subgraph.block node) to the user's reusable-block "
            "library so it can be dropped in later with add_block. Optionally pass `name` "
            "to label it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "nodeId": {"type": "string", "description": "id of the subgraph.block node to save"},
                "name": {"type": "string", "description": "optional display name for the saved block"},
            },
            "required": ["nodeId"],
        },
    ),
    ToolSpec(
        name="get_graph",
        description="Read the CURRENT graph (nodes + edges) — use this to re-check state after edits, since the snapshot in the prompt can go stale mid-build.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_node",
        description="Inspect one node: its type, properties, computed output shape, parameter count, and any error. Use this to debug shape mismatches.",
        parameters={
            "type": "object",
            "properties": {"nodeId": {"type": "string"}},
            "required": ["nodeId"],
        },
    ),
    ToolSpec(
        name="validate",
        description="Run the pre-flight checks and return the exact problems (missing connections, shape mismatches, missing data/loss/optimizer, etc.). mode='forward' checks a forward pass; mode='training' checks everything needed to train.",
        parameters={
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["forward", "training"]}},
            "required": ["mode"],
        },
    ),
]
