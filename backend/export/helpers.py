"""Small helpers for generating identifiers, indentation, and node lookups."""

import re
from engine.graph_builder import (
    ALL_LOSS_NODES, OPTIMIZER_NODES,
)
from dataprep.data_loaders import DATA_LOADERS

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_id(node_id: str) -> str:
    """Convert node ID to a valid Python identifier (for variable / attribute names)."""
    s = re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
    if s[0:1].isdigit():
        s = '_' + s
    return s


def _indent(code: str, level: int = 2) -> str:
    """Indent every line of code by level * 4 spaces."""
    prefix = '    ' * level
    return '\n'.join(prefix + line if line.strip() else '' for line in code.split('\n'))


def _get_optimizer_props(nodes: dict) -> dict:
    """Find the first optimizer node and return its properties."""
    for n in nodes.values():
        if n["type"] in OPTIMIZER_NODES:
            return n.get("properties", {})
    return {"lr": 0.001, "epochs": 10}


def _get_optimizer_type(nodes: dict) -> str:
    """Return the optimizer node type string."""
    for n in nodes.values():
        if n["type"] in OPTIMIZER_NODES:
            return n["type"]
    return "ml.optimizers.adam"


def _get_data_node(nodes: dict) -> dict | None:
    """Find the data node."""
    for n in nodes.values():
        if n["type"] in DATA_LOADERS:
            return n
    return None


def _get_loss_node(nodes: dict) -> dict | None:
    """Find the loss node."""
    for n in nodes.values():
        if n["type"] in ALL_LOSS_NODES:
            return n
    return None


