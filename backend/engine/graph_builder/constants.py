# Node type sets and sentinel identifiers shared across the engine.
#
# These describe how each node type is wired during execution:
#   - LOSS_NODES / ALL_LOSS_NODES: take named (predictions, labels) inputs
#   - OPTIMIZER_NODES: skipped during forward, drive the training loop
#   - MULTI_INPUT_NODES: take multiple named inputs, called with **kwargs
#   - SUBGRAPH_TYPE / SENTINEL_*: composite-block plumbing

# Node types that take multiple named inputs instead of a single "in" port
LOSS_NODES = {"ml.loss.cross_entropy", "ml.loss.mse"}
OPTIMIZER_NODES = {"ml.optimizers.sgd", "ml.optimizers.adam", "ml.optimizers.adamw"}
# Structural nodes with multiple named inputs (passed as **kwargs)
MULTI_INPUT_NODES = {"ml.structural.add", "ml.structural.concat", "ml.layers.multihead_attention", "ml.layers.attention", "ml.structural.reparameterize", "ml.loss.vae", "ml.loss.gan"}
# All node types recognized as loss functions (for training loop loss detection)
ALL_LOSS_NODES = LOSS_NODES | {"ml.loss.vae", "ml.loss.gan"}
# GAN-specific node types (noise input generates noise, not dataset)
GAN_NOISE_TYPE = "ml.gan.noise_input"
# Diffusion-specific node types
DIFFUSION_SCHEDULER_TYPE = "ml.diffusion.noise_scheduler"
DIFFUSION_EMBED_TYPE = "ml.diffusion.timestep_embed"
SUBGRAPH_TYPE = "subgraph.block"
SENTINEL_INPUT = "subgraph.input"
SENTINEL_OUTPUT = "subgraph.output"
