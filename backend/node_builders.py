# Per-node-type builder functions.
# Each function takes (properties, input_shapes) and returns a torch.nn.Module.

import torch
import torch.nn as nn


# --- Layers ---

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


def build_batchnorm2d(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    num_features = in_shape[1] if in_shape else 1
    return nn.BatchNorm2d(num_features=num_features)


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


def build_multihead_attention(props: dict, input_shapes: dict) -> nn.Module:
    mha = nn.MultiheadAttention(
        embed_dim=props["embedDim"],
        num_heads=props["numHeads"],
        dropout=props.get("dropout", 0.0),
        batch_first=True,
    )
    return MHAWrapper(mha)


# --- Activations ---

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


# --- Loss ---

def build_cross_entropy_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.CrossEntropyLoss()


def build_mse_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MSELoss()


# --- Structural ---

class AddModule(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class ConcatModule(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        # Sort by key to ensure consistent ordering (in_0, in_1, in_2...)
        tensors = [v for k, v in sorted(inputs.items()) if k.startswith('in_')]
        return torch.cat(tensors, dim=self.dim)


class ReshapeModule(nn.Module):
    def __init__(self, target_shape: list[int]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.target_shape)


class PermuteModule(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class MHAWrapper(nn.Module):
    """Wraps nn.MultiheadAttention to accept named Q/K/V inputs."""
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.mha = mha

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.mha(query, key, value)
        return output


def build_add(props: dict, input_shapes: dict) -> nn.Module:
    return AddModule()


def build_concat(props: dict, input_shapes: dict) -> nn.Module:
    return ConcatModule(dim=props.get("dim", 1))


def build_reshape(props: dict, input_shapes: dict) -> nn.Module:
    target_str = props.get("targetShape", "-1")
    in_shape = input_shapes.get("in", [1])
    target = [int(s.strip()) for s in target_str.split(",")]
    # Resolve 0s
    resolved = [in_shape[i] if v == 0 and i < len(in_shape) else v for i, v in enumerate(target)]
    # Resolve -1
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


def build_permute(props: dict, input_shapes: dict) -> nn.Module:
    dims_str = props.get("dims", "0, 2, 1")
    dims = [int(s.strip()) for s in dims_str.split(",")]
    return PermuteModule(dims)


# --- Registry ---

NODE_BUILDERS: dict[str, callable] = {
    # Layers
    "ml.layers.conv2d": build_conv2d,
    "ml.layers.linear": build_linear,
    "ml.layers.flatten": build_flatten,
    "ml.layers.maxpool2d": build_maxpool2d,
    "ml.layers.batchnorm2d": build_batchnorm2d,
    "ml.layers.dropout": build_dropout,
    "ml.layers.layernorm": build_layernorm,
    "ml.layers.embedding": build_embedding,
    "ml.layers.multihead_attention": build_multihead_attention,
    # Activations
    "ml.activations.relu": build_relu,
    "ml.activations.sigmoid": build_sigmoid,
    "ml.activations.softmax": build_softmax,
    "ml.activations.gelu": build_gelu,
    "ml.activations.tanh": build_tanh,
    "ml.activations.leaky_relu": build_leaky_relu,
    # Loss
    "ml.loss.cross_entropy": build_cross_entropy_loss,
    "ml.loss.mse": build_mse_loss,
    # Structural
    "ml.structural.add": build_add,
    "ml.structural.concat": build_concat,
    "ml.structural.reshape": build_reshape,
    "ml.structural.permute": build_permute,
}
