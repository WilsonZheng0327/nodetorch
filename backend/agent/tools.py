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
            "Add a new node to the canvas. `type` must be a node type string from "
            "the catalog (e.g. ml.layers.conv2d). Returns the new node's id, which "
            "you can use to connect it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "node type from the catalog"},
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
]
