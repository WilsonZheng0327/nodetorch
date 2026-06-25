"""Single forward pass over a graph.

``build_and_run_graph`` builds an ``nn.Module`` per node, runs one forward
pass, collects per-node display metadata, and caches the run in ``_last_run``
for later layer-detail queries. ``execute_graph`` wraps it in ``no_grad`` for
the /forward endpoint.
"""

import torch
import torch.nn as nn

from engine.graph_builder._state import _last_run
from engine.graph_builder.build import topological_sort
from engine.graph_builder.runners import RunContext, run_node


def build_and_run_graph(graph_data: dict) -> tuple[
    dict[str, nn.Module],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict],
    dict,
    list,
]:
    """
    Build modules and run a forward pass. Returns everything needed
    for both single forward and training.
    """
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    ctx = RunContext(edges=edges, results={}, modules={}, use_trained=False)
    node_results: dict[str, dict] = {}
    for node_id in order:
        node_results[node_id] = run_node(nodes[node_id], ctx)

    # Cache for layer detail queries
    _last_run.clear()
    _last_run["modules"] = ctx.modules
    _last_run["results"] = ctx.results
    _last_run["nodes"] = nodes
    _last_run["edges"] = edges

    return ctx.modules, ctx.results, node_results, nodes, edges


def execute_graph(graph_data: dict) -> dict:
    """Run a single forward pass. Returns per-node results."""
    with torch.no_grad():
        _, _, node_results, _, _ = build_and_run_graph(graph_data)
    return node_results
