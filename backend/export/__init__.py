# export — generate a standalone, runnable Python training script from a NodeTorch graph.
#
# The generated file includes all imports, the model class (nn.Module) with __init__ and
# forward, dataset loading with correct normalization, a training loop (standard, VAE, GAN,
# or diffusion depending on the graph), and device selection (CUDA/MPS/CPU).
#
# Split into focused submodules; the public entry point is re-exported here:
#   from export import export_to_python
#
#   templates       — reusable injected code-string constants
#   helpers         — identifier/indentation/node-lookup helpers
#   layers          — per-node nn.Module + inline-op code generation
#   model           — forward body, subgraph classes, the model nn.Module
#   training_loops  — per-paradigm training loop + optimizer/scheduler code
#   datasets        — dataset-loading code generation
#   exporter        — export_to_python orchestration

from export.exporter import export_to_python

__all__ = ["export_to_python"]
