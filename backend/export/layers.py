"""Per-node code generation: nn.Module constructors and inline ops."""

from engine.graph_builder import (
    ALL_LOSS_NODES, DIFFUSION_EMBED_TYPE, DIFFUSION_SCHEDULER_TYPE, GAN_NOISE_TYPE, OPTIMIZER_NODES,
)
from dataprep.data_loaders import DATA_LOADERS

# ─── Module code generation ───────────────────────────────────────────────────

def _module_code(node_type: str, props: dict, module) -> str | None:
    """Generate nn.Module constructor code for a given node type.

    Uses the actual built module to resolve input dimensions (in_channels, in_features, etc.).
    Returns None for nodes that don't become modules (data, loss, optimizer, etc.).
    """
    if node_type == "ml.layers.conv2d":
        m = module
        return f"nn.Conv2d({m.in_channels}, {m.out_channels}, kernel_size={m.kernel_size[0]}, stride={m.stride[0]}, padding={m.padding[0]})"

    if node_type == "ml.layers.conv1d":
        m = module
        return f"nn.Conv1d({m.in_channels}, {m.out_channels}, kernel_size={m.kernel_size[0]}, stride={m.stride[0]}, padding={m.padding[0]})"

    if node_type == "ml.layers.conv_transpose2d":
        m = module
        op = m.output_padding[0] if hasattr(m, 'output_padding') else 0
        return f"nn.ConvTranspose2d({m.in_channels}, {m.out_channels}, kernel_size={m.kernel_size[0]}, stride={m.stride[0]}, padding={m.padding[0]}, output_padding={op})"

    if node_type == "ml.layers.linear":
        m = module
        return f"nn.Linear({m.in_features}, {m.out_features})"

    if node_type == "ml.layers.flatten":
        return "nn.Flatten()"

    if node_type == "ml.layers.maxpool2d":
        ks = props["kernelSize"]
        st = props["stride"]
        pa = props["padding"]
        return f"nn.MaxPool2d(kernel_size={ks}, stride={st}, padding={pa})"

    if node_type == "ml.layers.maxpool1d":
        ks = props["kernelSize"]
        st = props["stride"]
        pa = props["padding"]
        return f"nn.MaxPool1d(kernel_size={ks}, stride={st}, padding={pa})"

    if node_type == "ml.layers.avgpool2d":
        ks = props["kernelSize"]
        st = props["stride"]
        pa = props["padding"]
        return f"nn.AvgPool2d(kernel_size={ks}, stride={st}, padding={pa})"

    if node_type == "ml.layers.adaptive_avgpool2d":
        oh = props["outputH"]
        ow = props["outputW"]
        return f"nn.AdaptiveAvgPool2d(({oh}, {ow}))"

    if node_type == "ml.layers.batchnorm2d":
        m = module
        return f"nn.BatchNorm2d({m.num_features})"

    if node_type == "ml.layers.batchnorm1d":
        m = module
        return f"nn.BatchNorm1d({m.num_features})"

    if node_type == "ml.layers.groupnorm":
        m = module
        return f"nn.GroupNorm({m.num_groups}, {m.num_channels})"

    if node_type == "ml.layers.instancenorm2d":
        m = module
        return f"nn.InstanceNorm2d({m.num_features}, affine=True)"

    if node_type == "ml.layers.dropout":
        p = props["p"]
        return f"nn.Dropout(p={p})"

    if node_type == "ml.layers.dropout2d":
        p = props["p"]
        return f"nn.Dropout2d(p={p})"

    if node_type == "ml.layers.layernorm":
        m = module
        ns = list(m.normalized_shape)
        return f"nn.LayerNorm({ns})"

    if node_type == "ml.layers.embedding":
        ne = props["numEmbeddings"]
        ed = props["embeddingDim"]
        return f"nn.Embedding({ne}, {ed})"

    if node_type == "ml.layers.positional_encoding":
        max_len = props.get("maxLen", 512)
        encoding_type = props.get("encodingType", "learned")
        # Read embed_dim from the runtime module so it stays in sync with upstream shape
        embed_dim = getattr(module, "embed_dim", None) if module is not None else None
        if embed_dim is None:
            embed_dim = props.get("embeddingDim", 256)
        return f"__POSITIONAL_ENCODING__({max_len}, {embed_dim}, '{encoding_type}')"

    if node_type in (
        "ml.preprocessing.tokenizer_char",
        "ml.preprocessing.tokenizer_word",
        "ml.preprocessing.tokenizer_bpe",
    ):
        # Char tokenizer has no vocabSize prop — vocab is corpus-determined.
        vs = props.get("vocabSize", 1_000_000)
        ml = props.get("maxLen", 256)
        return f"__TOKENIZER__({vs}, {ml})"

    if node_type == "ml.layers.upsample":
        sf = props["scaleFactor"]
        mode = props.get("mode", "nearest")
        return f"nn.Upsample(scale_factor={sf}, mode='{mode}')"

    if node_type == "ml.layers.lstm":
        m = module.lstm  # LSTMWrapper
        return f"nn.LSTM(input_size={m.input_size}, hidden_size={m.hidden_size}, num_layers={m.num_layers}, batch_first=True, bidirectional={m.bidirectional})"

    if node_type == "ml.layers.gru":
        m = module.gru  # GRUWrapper
        return f"nn.GRU(input_size={m.input_size}, hidden_size={m.hidden_size}, num_layers={m.num_layers}, batch_first=True, bidirectional={m.bidirectional})"

    if node_type == "ml.layers.rnn":
        m = module.rnn  # RNNWrapper
        return f"nn.RNN(input_size={m.input_size}, hidden_size={m.hidden_size}, num_layers={m.num_layers}, batch_first=True, bidirectional={m.bidirectional})"

    if node_type == "ml.layers.multihead_attention":
        m = module.mha  # MHAWrapper
        return f"nn.MultiheadAttention(embed_dim={m.embed_dim}, num_heads={m.num_heads}, batch_first=True)"

    if node_type == "ml.layers.pretrained_resnet18":
        mode = props.get("mode", "features")
        freeze = props.get("freeze", True)
        # This gets special handling — not a simple nn.Module line
        return f"__PRETRAINED_RESNET18__(mode='{mode}', freeze={freeze})"

    # Activations
    if node_type == "ml.activations.relu":
        return "nn.ReLU()"
    if node_type == "ml.activations.sigmoid":
        return "nn.Sigmoid()"
    if node_type == "ml.activations.tanh":
        return "nn.Tanh()"
    if node_type == "ml.activations.gelu":
        return "nn.GELU()"
    if node_type == "ml.activations.leaky_relu":
        ns = props.get("negativeSlope", 0.01)
        return f"nn.LeakyReLU(negative_slope={ns})"
    if node_type == "ml.activations.softmax":
        dim = props.get("dim", -1)
        return f"nn.Softmax(dim={dim})"

    return None


# ─── Inline operation code (for nodes that don't become modules) ──────────────

def _inline_code(node_type: str, props: dict) -> str | None:
    """Return inline forward code for nodes that don't become self.<name> modules."""
    if node_type == "ml.structural.reshape":
        target_str = props.get("targetShape", "-1")
        parts = [s.strip() for s in target_str.split(",")]
        # Replace leading -1 with x.size(0) for batch dim
        if parts[0] == "-1":
            parts[0] = "x.size(0)"
        shape_str = ", ".join(parts)
        return f"x = x.reshape({shape_str})"

    if node_type == "ml.structural.permute":
        dims_str = props.get("dims", "0, 2, 1")
        return f"x = x.permute({dims_str})"

    if node_type == "ml.structural.sequence_pool":
        mode = props.get("mode", "last")
        if mode == "last":
            return "x = x[:, -1, :]"
        elif mode == "mean":
            return "x = x.mean(dim=1)"
        elif mode == "max":
            return "x = x.max(dim=1).values"
        return "x = x[:, -1, :]"

    return None


# ─── Node classification ──────────────────────────────────────────────────────

INLINE_NODES = {"ml.structural.reshape", "ml.structural.permute", "ml.structural.sequence_pool"}
SKIP_NODES = set(ALL_LOSS_NODES) | set(OPTIMIZER_NODES) | set(DATA_LOADERS.keys()) | {GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE}
MULTI_OUTPUT_NODES = {"ml.layers.lstm", "ml.layers.gru", "ml.layers.rnn"}


