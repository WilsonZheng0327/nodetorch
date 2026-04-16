# node_builders.py — Per-node-type builder functions.
#
# Each builder takes (properties, input_shapes) and returns a torch.nn.Module.
# input_shapes is a dict of port_id → shape list, derived from upstream tensors.
# This is how nodes that need input dimensions (Conv2d needs in_channels, Linear needs
# in_features) get them — the shapes are computed from actual tensor data during the
# initial forward pass in build_and_run_graph().
#
# Wrapper modules (LSTMWrapper, GRUWrapper, MHAWrapper, ConcatModule, etc.) exist
# because some PyTorch modules don't match our calling convention:
#   - LSTM/GRU return tuples → wrappers return dicts for multi-output routing
#   - MHA takes (Q,K,V) → wrapper accepts named kwargs
#   - Concat needs sorted inputs → wrapper sorts by key prefix
#   - F.scaled_dot_product_attention is functional → AttentionModule wraps it
#
# NODE_BUILDERS registry maps node type strings → builder functions.
# graph_builder.py looks up builders from this registry.

import torch
import torch.nn as nn


# --- Layers (no wrapper needed — single tensor in, single tensor out) ---

def build_conv2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_channels = in_shape[1] if in_shape else 1
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=props["outChannels"],
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_conv1d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_channels = in_shape[1] if in_shape else 1
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=props["outChannels"],
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_linear(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    in_features = in_shape[-1] if in_shape else 1
    return nn.Linear(
        in_features=in_features,
        out_features=props["outFeatures"],
    )


def build_flatten(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Flatten()


def build_maxpool2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MaxPool2d(
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_avgpool2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.AvgPool2d(
        kernel_size=props["kernelSize"],
        stride=props["stride"],
        padding=props["padding"],
    )


def build_adaptive_avgpool2d(props: dict, input_shapes: dict) -> nn.Module:
    return nn.AdaptiveAvgPool2d((props["outputH"], props["outputW"]))


def build_batchnorm2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_features = in_shape[1] if in_shape else 1
    return nn.BatchNorm2d(num_features=num_features)


def build_batchnorm1d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_features = in_shape[1] if in_shape else 1
    return nn.BatchNorm1d(num_features=num_features)


def build_dropout(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Dropout(p=props["p"])


def build_layernorm(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    n = props.get("numLastDims", 1)
    normalized_shape = in_shape[-n:] if in_shape else [1]
    return nn.LayerNorm(normalized_shape)


def build_embedding(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Embedding(
        num_embeddings=props["numEmbeddings"],
        embedding_dim=props["embeddingDim"],
    )


# --- Layers with wrappers (multi-output or non-standard calling convention) ---

class LSTMWrapper(nn.Module):
    """Why wrapper: nn.LSTM returns a tuple (output, (hidden, cell)). Our execution
    loop expects either a single tensor or a dict. This unpacks the tuple into a dict
    so each output can be routed to different downstream nodes via port IDs."""
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self.lstm = lstm

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, (hidden, cell) = self.lstm(x)
        return {"out": output, "hidden": hidden, "cell": cell}


def build_lstm(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    input_size = in_shape[-1] if in_shape else 1
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=props["hiddenSize"],
        num_layers=props.get("numLayers", 1),
        batch_first=True,
        bidirectional=props.get("bidirectional", False),
        dropout=props.get("dropout", 0) if props.get("numLayers", 1) > 1 else 0,
    )
    return LSTMWrapper(lstm)


class GRUWrapper(nn.Module):
    """Why wrapper: nn.GRU returns a tuple (output, hidden). Same reason as LSTMWrapper —
    unpacks the tuple into a dict for multi-output port routing."""
    def __init__(self, gru: nn.GRU):
        super().__init__()
        self.gru = gru

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, hidden = self.gru(x)
        return {"out": output, "hidden": hidden}


def build_gru(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    input_size = in_shape[-1] if in_shape else 1
    gru = nn.GRU(
        input_size=input_size,
        hidden_size=props["hiddenSize"],
        num_layers=props.get("numLayers", 1),
        batch_first=True,
        bidirectional=props.get("bidirectional", False),
        dropout=props.get("dropout", 0) if props.get("numLayers", 1) > 1 else 0,
    )
    return GRUWrapper(gru)


class MHAWrapper(nn.Module):
    """Why wrapper: nn.MultiheadAttention returns (attn_output, attn_weights) — we only
    want the output. Also, our multi-input path calls module(**kwargs) with named args
    (query, key, value), which MHA accepts but also returns the unwanted weights tuple."""
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.mha(query, key, value)
        return output


def build_multihead_attention(props: dict, input_shapes: dict) -> nn.Module:
    mha = nn.MultiheadAttention(
        embed_dim=props["embedDim"],
        num_heads=props["numHeads"],
        dropout=props.get("dropout", 0.0),
        batch_first=True,
    )
    return MHAWrapper(mha)


class AttentionModule(nn.Module):
    """Why wrapper: F.scaled_dot_product_attention is a function, not an nn.Module.
    Can't be registered in nn.ModuleDict, has no .parameters(). This wraps it
    so it behaves like any other module in the execution loop."""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0,
        )


def build_attention(props: dict, input_shapes: dict) -> nn.Module:
    return AttentionModule(dropout=props.get("dropout", 0.0))


# --- Activations (no wrapper needed — all are single tensor in/out) ---

def build_relu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.ReLU()


def build_sigmoid(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Sigmoid()


def build_softmax(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Softmax(dim=props.get("dim", -1))


def build_gelu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.GELU()


def build_tanh(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Tanh()


def build_leaky_relu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.LeakyReLU(negative_slope=props.get("negativeSlope", 0.01))


# --- Loss (no wrapper needed) ---

def build_cross_entropy_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.CrossEntropyLoss()


def build_mse_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MSELoss()


# --- Structural (all need wrappers — none are native nn.Modules) ---

class AddModule(nn.Module):
    """Why wrapper: element-wise addition is just `a + b` — not a PyTorch module.
    Needs to be a module so it can be stored in nn.ModuleDict and called via module(**kwargs)."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def build_add(props: dict, input_shapes: dict) -> nn.Module:
    return AddModule()


class ConcatModule(nn.Module):
    """Why wrapper: torch.cat is a function, not a module. Also needs to accept **kwargs
    (in_0, in_1, ...) from our multi-input calling convention and sort them by key."""
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        tensors = [v for k, v in sorted(inputs.items()) if k.startswith('in_')]
        return torch.cat(tensors, dim=self.dim)


def build_concat(props: dict, input_shapes: dict) -> nn.Module:
    return ConcatModule(dim=props.get("dim", 1))


class ReshapeModule(nn.Module):
    """Why wrapper: torch.reshape is a tensor method, not a module. Needs to store
    the target shape as state so it can be called as module(x) in the execution loop."""
    def __init__(self, target_shape: list[int]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.target_shape)


def build_reshape(props: dict, input_shapes: dict) -> nn.Module:
    target_str = props.get("targetShape", "-1")
    in_shape = input_shapes.get("in", [1])
    target = [int(s.strip()) for s in target_str.split(",")]
    # Resolve 0s (keep original dimension)
    resolved = [in_shape[i] if v == 0 and i < len(in_shape) else v for i, v in enumerate(target)]
    # Resolve -1 (infer dimension)
    total = 1
    for s in in_shape:
        total *= s
    neg_idx = -1
    for i, v in enumerate(resolved):
        if v == -1:
            neg_idx = i
    if neg_idx != -1:
        known = 1
        for i, v in enumerate(resolved):
            if i != neg_idx:
                known *= v
        resolved[neg_idx] = total // known
    return ReshapeModule(resolved)


class PermuteModule(nn.Module):
    """Why wrapper: torch.permute is a tensor method, not a module. Same as ReshapeModule —
    stores the dimension order as state."""
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


def build_permute(props: dict, input_shapes: dict) -> nn.Module:
    dims_str = props.get("dims", "0, 2, 1")
    dims = [int(s.strip()) for s in dims_str.split(",")]
    return PermuteModule(dims)


# --- Registry ---

NODE_BUILDERS: dict[str, callable] = {
    # Layers (no wrapper)
    "ml.layers.conv2d": build_conv2d,
    "ml.layers.conv1d": build_conv1d,
    "ml.layers.linear": build_linear,
    "ml.layers.flatten": build_flatten,
    "ml.layers.maxpool2d": build_maxpool2d,
    "ml.layers.avgpool2d": build_avgpool2d,
    "ml.layers.adaptive_avgpool2d": build_adaptive_avgpool2d,
    "ml.layers.batchnorm2d": build_batchnorm2d,
    "ml.layers.batchnorm1d": build_batchnorm1d,
    "ml.layers.dropout": build_dropout,
    "ml.layers.layernorm": build_layernorm,
    "ml.layers.embedding": build_embedding,
    # Layers (with wrapper)
    "ml.layers.lstm": build_lstm,
    "ml.layers.gru": build_gru,
    "ml.layers.multihead_attention": build_multihead_attention,
    "ml.layers.attention": build_attention,
    # Activations (no wrapper)
    "ml.activations.relu": build_relu,
    "ml.activations.sigmoid": build_sigmoid,
    "ml.activations.softmax": build_softmax,
    "ml.activations.gelu": build_gelu,
    "ml.activations.tanh": build_tanh,
    "ml.activations.leaky_relu": build_leaky_relu,
    # Loss (no wrapper)
    "ml.loss.cross_entropy": build_cross_entropy_loss,
    "ml.loss.mse": build_mse_loss,
    # Structural (all wrapped)
    "ml.structural.add": build_add,
    "ml.structural.concat": build_concat,
    "ml.structural.reshape": build_reshape,
    "ml.structural.permute": build_permute,
}
