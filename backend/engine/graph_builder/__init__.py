# graph_builder — converts a serialized NodeTorch graph into PyTorch modules and executes them.
#
# Mirrors the TypeScript engine: topological sort → gather inputs → run module → store output.
# Split into focused submodules; this package re-exports the full public API so existing
# `from engine.graph_builder import X` call sites keep working unchanged.
#
#   constants  — node-type sets and sentinel ids
#   _state     — device selection + in-memory trained-model store (shared mutable state)
#   stats      — tensor/parameter statistics for visualization metadata
#   build      — topological sort, input routing, module/subgraph construction
#   forward    — standalone inspection pass (build_and_run_graph / inspect_graph)
#   detail     — per-node inspector visualization (get_layer_detail)
#   inference  — infer_graph / evaluate_test_set + tracked-sample helpers

from engine.graph_builder.constants import (
    LOSS_NODES, OPTIMIZER_NODES, MULTI_INPUT_NODES, ALL_LOSS_NODES,
    GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE,
    SUBGRAPH_TYPE, SENTINEL_INPUT, SENTINEL_OUTPUT,
)
from engine.graph_builder._state import (
    get_device, set_device, get_device_name,
    has_trained_model, get_trained_modules,
    save_model, save_model_bytes, load_model_bytes, load_model,
    save_bundle_bytes, load_bundle_bytes, snapshot_trained_model, ensure_trained_model,
    _model_store, _last_run,
)
from engine.graph_builder.stats import (
    _safe_float, tensor_info, module_weight_info, batchnorm_info,
    gradient_info, activation_info,
)
from engine.graph_builder.build import (
    topological_sort, gather_inputs, SubGraphModule,
    build_subgraph_module, build_modules,
)
from engine.graph_builder.forward import build_and_run_graph, inspect_graph
from engine.graph_builder.detail import get_layer_detail
from engine.graph_builder.inference import (
    infer_graph, evaluate_test_set,
    _pick_tracked_samples, _probe_tracked_samples, _collect_misclassifications,
)


def train_graph(graph_data: dict, on_epoch=None, on_batch=None, cancel_event=None) -> dict:
    """Run a training loop. Delegates to the training plugin system.

    Auto-detects the training paradigm (standard, GAN, diffusion) from node
    types and runs the appropriate loop. See backend/training/ for details.
    """
    from training import run_training
    return run_training(graph_data, on_epoch, on_batch, cancel_event)
