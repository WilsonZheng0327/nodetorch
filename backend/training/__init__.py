"""Training loop plugin system.

Each training paradigm (standard, GAN, diffusion) is a separate function that
receives a TrainingContext and returns a TrainingResult. The registry maps
paradigm names to functions. detect_training_mode() auto-selects based on
which node types are present in the graph.

Adding a new training paradigm:
  1. Create a new file in this package (e.g., gan.py)
  2. Implement a function with signature (ctx: TrainingContext) -> TrainingResult
  3. Register it in TRAINING_LOOPS
  4. Add detection logic to detect_training_mode()
"""

from .base import TrainingContext, TrainingResult, build_training_context, save_training_results
from .standard import standard_train
from .gan import gan_train
from .diffusion import diffusion_train
from .autoregressive import autoregressive_train
from data_loaders import LM_DATASET_TYPES

TRAINING_LOOPS: dict[str, callable] = {
    "standard": standard_train,
    "gan": gan_train,
    "diffusion": diffusion_train,
    "autoregressive": autoregressive_train,
}


def detect_training_mode(nodes: dict) -> str:
    """Auto-detect which training loop to use based on node types in graph."""
    for n in nodes.values():
        ntype = n.get("type", "")
        if ntype in ("ml.gan.noise_input", "ml.loss.gan"):
            return "gan"
        if ntype == "ml.diffusion.noise_scheduler":
            return "diffusion"
        if ntype in LM_DATASET_TYPES:
            return "autoregressive"
    return "standard"


def run_training(graph_data: dict, on_epoch=None, on_batch=None, cancel_event=None) -> dict:
    """Drop-in replacement for graph_builder.train_graph().

    Builds the training context, detects the paradigm, runs the appropriate
    training loop, and stores results.
    """
    ctx = build_training_context(graph_data, on_epoch, on_batch, cancel_event)
    if isinstance(ctx, dict) and "error" in ctx:
        return ctx

    mode = detect_training_mode(ctx.nodes)
    loop_fn = TRAINING_LOOPS.get(mode, standard_train)
    result = loop_fn(ctx)

    if result.error:
        return {"error": result.error}

    return save_training_results(ctx, result)
