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


# --- Activations ---

def build_relu(props: dict, input_shapes: dict) -> nn.Module:
    return nn.ReLU()


def build_sigmoid(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Sigmoid()


def build_softmax(props: dict, input_shapes: dict) -> nn.Module:
    return nn.Softmax(dim=props.get("dim", -1))


# --- Loss ---

def build_cross_entropy_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.CrossEntropyLoss()


def build_mse_loss(props: dict, input_shapes: dict) -> nn.Module:
    return nn.MSELoss()


# --- Structural ---

class AddModule(nn.Module):
    """Element-wise addition of two tensors."""
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def build_add(props: dict, input_shapes: dict) -> nn.Module:
    return AddModule()


# --- Registry ---

NODE_BUILDERS: dict[str, callable] = {
    # Layers
    "ml.layers.conv2d": build_conv2d,
    "ml.layers.linear": build_linear,
    "ml.layers.flatten": build_flatten,
    "ml.layers.maxpool2d": build_maxpool2d,
    "ml.layers.batchnorm2d": build_batchnorm2d,
    "ml.layers.dropout": build_dropout,
    # Activations
    "ml.activations.relu": build_relu,
    "ml.activations.sigmoid": build_sigmoid,
    "ml.activations.softmax": build_softmax,
    # Loss
    "ml.loss.cross_entropy": build_cross_entropy_loss,
    "ml.loss.mse": build_mse_loss,
    # Structural
    "ml.structural.add": build_add,
}
