"""The serialized-graph wire format — the shape of every ``graph_data`` dict.

This is the backend mirror of the frontend's ``src/core/serialization.ts``. The
frontend serializes a ``Graph`` to this plain-JSON structure (version ``"1.0"``)
and sends it to the backend over the WebSocket (``/ws`` train) and the REST
endpoints (infer / test / step-through / export / …). Every backend function that
takes a ``graph_data`` parameter receives exactly this structure.

These are ``TypedDict``s, so they're **documentation you can hover**: annotate a
parameter ``graph_data: SerializedGraph`` and your editor will show this docstring
and resolve ``graph_data["graph"]["nodes"][0]["type"]`` down to ``str`` — instead
of the opaque ``dict`` that told you nothing. They are *not* enforced at runtime
(nothing validates against them here) and the backend isn't statically
type-checked in CI, so they never change behavior; they exist purely so the shape
is legible everywhere the dict travels.

Keep these in sync with ``src/core/serialization.ts`` — they are the same record
described on both sides of the wire (same convention as ``EpochMessage`` /
``EpochData``).

Shape at a glance::

    {
      "version": "1.0",
      "graph": {
        "id": "...", "name": "...",
        "nodes": [ {"id","type","position","properties", "subgraph"?}, ... ],
        "edges": [ {"id","source":{"nodeId","portId"},"target":{"nodeId","portId"}}, ... ],
      },
    }
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class Position(TypedDict):
    """A node's canvas coordinates. Carried for round-tripping the layout; the
    engine ignores it (execution depends only on nodes + edges).

    Fields:
    - x: horizontal position on the canvas, in pixels.
    - y: vertical position on the canvas, in pixels.
    """

    x: float
    y: float


class EdgeEndpoint(TypedDict):
    """One end of an edge — which node, and which of its ports.

    Fields:
    - nodeId: id of the node this end attaches to (matches a ``SerializedNode.id``).
    - portId: which port on that node — selects among multi-port nodes (a data
      node exposes ``"out"`` and ``"labels"``; an LSTM exposes ``"out"`` /
      ``"hidden"`` / ``"cell"``). ``gather_inputs`` keys a node's inputs by the
      *target* ``portId``.
    """

    nodeId: str
    portId: str


class SerializedEdge(TypedDict):
    """A directed connection ``source → target`` between two node ports.

    Fields:
    - id: unique edge id.
    - source: the upstream endpoint the tensor flows *from*.
    - target: the downstream endpoint the tensor flows *into*.
    """

    id: str
    source: EdgeEndpoint
    target: EdgeEndpoint


class SerializedNode(TypedDict):
    """One node as stored on the wire: identity, type, layout, and properties.

    Fields:
    - id: unique node id, referenced by edges.
    - type: registry key that decides everything about the node — which
      ``nn.Module`` is built, whether it's a data / loss / optimizer node, and how
      it executes (see the ``*_NODES`` sets and ``DATA_LOADERS``).
    - position: canvas coordinates (layout only; ignored by the engine).
    - properties: the node's free-form config bag; keys depend on ``type`` (an
      optimizer node's ``{"epochs": 3, "lr": 0.01}``, a conv node's
      ``{"outChannels": 8}``), so it stays ``dict[str, Any]``.
    - subgraph: present only for composite / custom-block nodes — a full nested
      inner graph (hence the recursion into ``SerializedGraphData``).
    """

    id: str
    type: str
    position: Position
    properties: dict[str, Any]
    subgraph: NotRequired["SerializedGraphData"]


class SerializedGraphData(TypedDict):
    """The inner graph payload — identity plus the node and edge lists. This is
    what the engine actually walks: ``nodes`` are topologically sorted and
    ``edges`` route tensors between them.

    Fields:
    - id: unique graph id.
    - name: human-readable graph name (shown in the UI).
    - nodes: every node in the graph; composite nodes recurse via ``subgraph``.
    - edges: every connection, routing outputs to inputs by port.
    """

    id: str
    name: str
    nodes: list[SerializedNode]
    edges: list[SerializedEdge]


class SerializedGraph(TypedDict):
    """The versioned top-level wrapper — the exact type of every ``graph_data``.

    Fields:
    - version: format version, gating compatibility (currently only ``"1.0"``).
    - graph: the ``SerializedGraphData`` the engine consumes.
    """

    version: Literal["1.0"]
    graph: SerializedGraphData
