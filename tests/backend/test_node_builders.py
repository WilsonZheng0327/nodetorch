"""Tests for backend/node_builders.py — NODE_BUILDERS registry and builder functions.

Each builder takes (props: dict, input_shapes: dict) and returns an nn.Module.
We verify each builder produces an nn.Module and that the module produces the
correct output shape when given a test tensor.
"""

import torch
import torch.nn as nn
import pytest

from node_builders import NODE_BUILDERS


class TestConv2d:
    def test_returns_module(self):
        props = {"outChannels": 16, "kernelSize": 3, "stride": 1, "padding": 1}
        input_shapes = {"in": [1, 3, 32, 32]}
        module = NODE_BUILDERS["ml.layers.conv2d"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"outChannels": 16, "kernelSize": 3, "stride": 1, "padding": 1}
        input_shapes = {"in": [1, 3, 32, 32]}
        module = NODE_BUILDERS["ml.layers.conv2d"](props, input_shapes)
        x = torch.randn(1, 3, 32, 32)
        out = module(x)
        assert list(out.shape) == [1, 16, 32, 32]


class TestLinear:
    def test_returns_module(self):
        props = {"outFeatures": 10}
        input_shapes = {"in": [1, 512]}
        module = NODE_BUILDERS["ml.layers.linear"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"outFeatures": 10}
        input_shapes = {"in": [1, 512]}
        module = NODE_BUILDERS["ml.layers.linear"](props, input_shapes)
        x = torch.randn(1, 512)
        out = module(x)
        assert list(out.shape) == [1, 10]


class TestFlatten:
    def test_returns_module(self):
        props = {}
        input_shapes = {"in": [1, 64, 7, 7]}
        module = NODE_BUILDERS["ml.layers.flatten"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {}
        input_shapes = {"in": [1, 64, 7, 7]}
        module = NODE_BUILDERS["ml.layers.flatten"](props, input_shapes)
        x = torch.randn(1, 64, 7, 7)
        out = module(x)
        assert list(out.shape) == [1, 3136]


class TestBatchNorm2d:
    def test_returns_module(self):
        input_shapes = {"in": [1, 16, 32, 32]}
        module = NODE_BUILDERS["ml.layers.batchnorm2d"]({}, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        input_shapes = {"in": [1, 16, 32, 32]}
        module = NODE_BUILDERS["ml.layers.batchnorm2d"]({}, input_shapes)
        x = torch.randn(1, 16, 32, 32)
        out = module(x)
        assert list(out.shape) == [1, 16, 32, 32]


class TestMaxPool2d:
    def test_returns_module(self):
        props = {"kernelSize": 2, "stride": 2, "padding": 0}
        input_shapes = {"in": [1, 16, 32, 32]}
        module = NODE_BUILDERS["ml.layers.maxpool2d"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"kernelSize": 2, "stride": 2, "padding": 0}
        input_shapes = {"in": [1, 16, 32, 32]}
        module = NODE_BUILDERS["ml.layers.maxpool2d"](props, input_shapes)
        x = torch.randn(1, 16, 32, 32)
        out = module(x)
        assert list(out.shape) == [1, 16, 16, 16]


class TestDropout:
    def test_returns_module(self):
        props = {"p": 0.5}
        input_shapes = {"in": [1, 512]}
        module = NODE_BUILDERS["ml.layers.dropout"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"p": 0.5}
        input_shapes = {"in": [1, 512]}
        module = NODE_BUILDERS["ml.layers.dropout"](props, input_shapes)
        x = torch.randn(1, 512)
        out = module(x)
        assert list(out.shape) == [1, 512]


class TestEmbedding:
    def test_returns_module(self):
        props = {"numEmbeddings": 10000, "embeddingDim": 64}
        input_shapes = {"in": [1, 128]}
        module = NODE_BUILDERS["ml.layers.embedding"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"numEmbeddings": 10000, "embeddingDim": 64}
        input_shapes = {"in": [1, 128]}
        module = NODE_BUILDERS["ml.layers.embedding"](props, input_shapes)
        x = torch.randint(0, 10000, (1, 128))
        out = module(x)
        assert list(out.shape) == [1, 128, 64]


class TestLSTM:
    def test_returns_module(self):
        props = {"hiddenSize": 128, "numLayers": 1, "bidirectional": False}
        input_shapes = {"in": [1, 128, 64]}
        module = NODE_BUILDERS["ml.layers.lstm"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"hiddenSize": 128, "numLayers": 1, "bidirectional": False}
        input_shapes = {"in": [1, 128, 64]}
        module = NODE_BUILDERS["ml.layers.lstm"](props, input_shapes)
        x = torch.randn(1, 128, 64)
        out = module(x)
        assert isinstance(out, dict)
        assert "out" in out
        assert list(out["out"].shape) == [1, 128, 128]


class TestReLU:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.activations.relu"]({}, {"in": [1, 64]})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.activations.relu"]({}, {"in": [1, 64]})
        x = torch.randn(1, 64)
        out = module(x)
        assert list(out.shape) == [1, 64]


class TestConvTranspose2d:
    def test_returns_module(self):
        props = {
            "outChannels": 16,
            "kernelSize": 4,
            "stride": 2,
            "padding": 1,
            "outputPadding": 0,
        }
        input_shapes = {"in": [1, 8, 7, 7]}
        module = NODE_BUILDERS["ml.layers.conv_transpose2d"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {
            "outChannels": 16,
            "kernelSize": 4,
            "stride": 2,
            "padding": 1,
            "outputPadding": 0,
        }
        input_shapes = {"in": [1, 8, 7, 7]}
        module = NODE_BUILDERS["ml.layers.conv_transpose2d"](props, input_shapes)
        x = torch.randn(1, 8, 7, 7)
        out = module(x)
        assert list(out.shape) == [1, 16, 14, 14]


class TestReshape:
    def test_returns_module(self):
        props = {"targetShape": "-1, 8, 7, 7"}
        input_shapes = {"in": [1, 392]}
        module = NODE_BUILDERS["ml.structural.reshape"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"targetShape": "-1, 8, 7, 7"}
        input_shapes = {"in": [1, 392]}
        module = NODE_BUILDERS["ml.structural.reshape"](props, input_shapes)
        x = torch.randn(1, 392)
        out = module(x)
        assert list(out.shape) == [1, 8, 7, 7]


class TestMSELoss:
    def test_returns_module(self):
        input_shapes = {"predictions": [1, 10], "labels": [1, 10]}
        module = NODE_BUILDERS["ml.loss.mse"]({}, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_is_scalar(self):
        input_shapes = {"predictions": [1, 10], "labels": [1, 10]}
        module = NODE_BUILDERS["ml.loss.mse"]({}, input_shapes)
        predictions = torch.randn(1, 10)
        labels = torch.randn(1, 10)
        out = module(predictions, labels)
        assert out.dim() == 0  # scalar


class TestCrossEntropyLoss:
    def test_returns_module(self):
        input_shapes = {"predictions": [1, 10], "labels": [1]}
        module = NODE_BUILDERS["ml.loss.cross_entropy"]({}, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_is_scalar(self):
        input_shapes = {"predictions": [1, 10], "labels": [1]}
        module = NODE_BUILDERS["ml.loss.cross_entropy"]({}, input_shapes)
        predictions = torch.randn(1, 10)
        labels = torch.randint(0, 10, (1,))
        out = module(predictions, labels)
        assert out.dim() == 0  # scalar
