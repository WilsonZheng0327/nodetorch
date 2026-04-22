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


# --- Additional layers ---


class TestConv1d:
    def test_returns_module(self):
        props = {"outChannels": 16, "kernelSize": 3, "stride": 1, "padding": 1}
        input_shapes = {"in": [2, 3, 32]}
        module = NODE_BUILDERS["ml.layers.conv1d"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"outChannels": 16, "kernelSize": 3, "stride": 1, "padding": 1}
        input_shapes = {"in": [2, 3, 32]}
        module = NODE_BUILDERS["ml.layers.conv1d"](props, input_shapes)
        x = torch.randn(2, 3, 32)
        out = module(x)
        assert list(out.shape) == [2, 16, 32]


class TestMaxPool1d:
    def test_returns_module(self):
        props = {"kernelSize": 2, "stride": 2, "padding": 0}
        module = NODE_BUILDERS["ml.layers.maxpool1d"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"kernelSize": 2, "stride": 2, "padding": 0}
        module = NODE_BUILDERS["ml.layers.maxpool1d"](props, {})
        x = torch.randn(2, 4, 32)
        out = module(x)
        assert list(out.shape) == [2, 4, 16]


class TestAvgPool2d:
    def test_returns_module(self):
        props = {"kernelSize": 2, "stride": 2, "padding": 0}
        module = NODE_BUILDERS["ml.layers.avgpool2d"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"kernelSize": 2, "stride": 2, "padding": 0}
        module = NODE_BUILDERS["ml.layers.avgpool2d"](props, {})
        x = torch.randn(2, 3, 64, 64)
        out = module(x)
        assert list(out.shape) == [2, 3, 32, 32]


class TestAdaptiveAvgPool2d:
    def test_returns_module(self):
        props = {"outputH": 1, "outputW": 1}
        module = NODE_BUILDERS["ml.layers.adaptive_avgpool2d"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"outputH": 1, "outputW": 1}
        module = NODE_BUILDERS["ml.layers.adaptive_avgpool2d"](props, {})
        x = torch.randn(2, 3, 128, 128)
        out = module(x)
        assert list(out.shape) == [2, 3, 1, 1]


class TestBatchNorm1d:
    def test_returns_module(self):
        input_shapes = {"in": [2, 16, 32]}
        module = NODE_BUILDERS["ml.layers.batchnorm1d"]({}, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        input_shapes = {"in": [2, 16, 32]}
        module = NODE_BUILDERS["ml.layers.batchnorm1d"]({}, input_shapes)
        x = torch.randn(2, 16, 32)
        out = module(x)
        assert list(out.shape) == [2, 16, 32]


class TestGroupNorm:
    def test_returns_module(self):
        props = {"numGroups": 8}
        input_shapes = {"in": [2, 32, 16, 16]}
        module = NODE_BUILDERS["ml.layers.groupnorm"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"numGroups": 8}
        input_shapes = {"in": [2, 32, 16, 16]}
        module = NODE_BUILDERS["ml.layers.groupnorm"](props, input_shapes)
        x = torch.randn(2, 32, 16, 16)
        out = module(x)
        assert list(out.shape) == [2, 32, 16, 16]


class TestInstanceNorm2d:
    def test_returns_module(self):
        input_shapes = {"in": [2, 3, 64, 64]}
        module = NODE_BUILDERS["ml.layers.instancenorm2d"]({}, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        input_shapes = {"in": [2, 3, 64, 64]}
        module = NODE_BUILDERS["ml.layers.instancenorm2d"]({}, input_shapes)
        x = torch.randn(2, 3, 64, 64)
        out = module(x)
        assert list(out.shape) == [2, 3, 64, 64]


class TestDropout2d:
    def test_returns_module(self):
        props = {"p": 0.5}
        module = NODE_BUILDERS["ml.layers.dropout2d"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"p": 0.5}
        module = NODE_BUILDERS["ml.layers.dropout2d"](props, {})
        x = torch.randn(2, 3, 32, 32)
        out = module(x)
        assert list(out.shape) == [2, 3, 32, 32]


class TestLayerNorm:
    def test_returns_module(self):
        props = {"numLastDims": 1}
        input_shapes = {"in": [2, 10, 256]}
        module = NODE_BUILDERS["ml.layers.layernorm"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"numLastDims": 1}
        input_shapes = {"in": [2, 10, 256]}
        module = NODE_BUILDERS["ml.layers.layernorm"](props, input_shapes)
        x = torch.randn(2, 10, 256)
        out = module(x)
        assert list(out.shape) == [2, 10, 256]


class TestPositionalEncoding:
    def test_returns_module(self):
        props = {"maxLen": 512, "encodingType": "learned"}
        input_shapes = {"in": [2, 20, 256]}
        module = NODE_BUILDERS["ml.layers.positional_encoding"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape_learned(self):
        props = {"maxLen": 512, "encodingType": "learned"}
        input_shapes = {"in": [2, 20, 256]}
        module = NODE_BUILDERS["ml.layers.positional_encoding"](props, input_shapes)
        x = torch.randn(2, 20, 256)
        out = module(x)
        assert list(out.shape) == [2, 20, 256]

    def test_output_shape_sinusoidal(self):
        props = {"maxLen": 512, "encodingType": "sinusoidal"}
        input_shapes = {"in": [2, 20, 256]}
        module = NODE_BUILDERS["ml.layers.positional_encoding"](props, input_shapes)
        x = torch.randn(2, 20, 256)
        out = module(x)
        assert list(out.shape) == [2, 20, 256]


class TestUpsample:
    def test_returns_module(self):
        props = {"scaleFactor": 2, "mode": "nearest"}
        module = NODE_BUILDERS["ml.layers.upsample"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"scaleFactor": 2, "mode": "nearest"}
        module = NODE_BUILDERS["ml.layers.upsample"](props, {})
        x = torch.randn(2, 3, 32, 32)
        out = module(x)
        assert list(out.shape) == [2, 3, 64, 64]


class TestGRU:
    def test_returns_module(self):
        props = {"hiddenSize": 64, "numLayers": 1, "bidirectional": False, "dropout": 0}
        input_shapes = {"in": [2, 10, 128]}
        module = NODE_BUILDERS["ml.layers.gru"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"hiddenSize": 64, "numLayers": 1, "bidirectional": False, "dropout": 0}
        input_shapes = {"in": [2, 10, 128]}
        module = NODE_BUILDERS["ml.layers.gru"](props, input_shapes)
        x = torch.randn(2, 10, 128)
        out = module(x)
        assert isinstance(out, dict)
        assert list(out["out"].shape) == [2, 10, 64]
        assert list(out["hidden"].shape) == [1, 2, 64]


class TestRNN:
    def test_returns_module(self):
        props = {"hiddenSize": 64, "numLayers": 1, "bidirectional": False, "nonlinearity": "tanh"}
        input_shapes = {"in": [2, 10, 128]}
        module = NODE_BUILDERS["ml.layers.rnn"](props, input_shapes)
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"hiddenSize": 64, "numLayers": 1, "bidirectional": False, "nonlinearity": "tanh"}
        input_shapes = {"in": [2, 10, 128]}
        module = NODE_BUILDERS["ml.layers.rnn"](props, input_shapes)
        x = torch.randn(2, 10, 128)
        out = module(x)
        assert isinstance(out, dict)
        assert list(out["out"].shape) == [2, 10, 64]
        assert list(out["hidden"].shape) == [1, 2, 64]


class TestMultiheadAttention:
    def test_returns_module(self):
        props = {"embedDim": 256, "numHeads": 8, "dropout": 0.0, "causalMask": False}
        module = NODE_BUILDERS["ml.layers.multihead_attention"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"embedDim": 256, "numHeads": 8, "dropout": 0.0, "causalMask": False}
        module = NODE_BUILDERS["ml.layers.multihead_attention"](props, {})
        q = k = v = torch.randn(2, 10, 256)
        out = module(query=q, key=k, value=v)
        assert list(out.shape) == [2, 10, 256]

    def test_causal_mask(self):
        props = {"embedDim": 64, "numHeads": 4, "dropout": 0.0, "causalMask": True}
        module = NODE_BUILDERS["ml.layers.multihead_attention"](props, {})
        q = k = v = torch.randn(2, 8, 64)
        out = module(query=q, key=k, value=v)
        assert list(out.shape) == [2, 8, 64]


class TestAttention:
    def test_returns_module(self):
        props = {"dropout": 0.0, "causalMask": False}
        module = NODE_BUILDERS["ml.layers.attention"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"dropout": 0.0, "causalMask": False}
        module = NODE_BUILDERS["ml.layers.attention"](props, {})
        q = k = v = torch.randn(2, 10, 64)
        out = module(query=q, key=k, value=v)
        assert list(out.shape) == [2, 10, 64]


# --- Additional activations ---


class TestSigmoid:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.activations.sigmoid"]({}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.activations.sigmoid"]({}, {})
        x = torch.randn(2, 10)
        out = module(x)
        assert list(out.shape) == [2, 10]


class TestSoftmax:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.activations.softmax"]({"dim": -1}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.activations.softmax"]({"dim": -1}, {})
        x = torch.randn(2, 10)
        out = module(x)
        assert list(out.shape) == [2, 10]


class TestGELU:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.activations.gelu"]({}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.activations.gelu"]({}, {})
        x = torch.randn(2, 10)
        out = module(x)
        assert list(out.shape) == [2, 10]


class TestTanh:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.activations.tanh"]({}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.activations.tanh"]({}, {})
        x = torch.randn(2, 10)
        out = module(x)
        assert list(out.shape) == [2, 10]


class TestLeakyReLU:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.activations.leaky_relu"]({"negativeSlope": 0.01}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.activations.leaky_relu"]({"negativeSlope": 0.01}, {})
        x = torch.randn(2, 10)
        out = module(x)
        assert list(out.shape) == [2, 10]


# --- Structural ---


class TestAdd:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.structural.add"]({}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.structural.add"]({}, {})
        a = torch.randn(2, 10)
        b = torch.randn(2, 10)
        out = module(a=a, b=b)
        assert list(out.shape) == [2, 10]


class TestConcat:
    def test_returns_module(self):
        props = {"dim": 1}
        module = NODE_BUILDERS["ml.structural.concat"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"dim": 1}
        module = NODE_BUILDERS["ml.structural.concat"](props, {})
        out = module(in_0=torch.randn(2, 10), in_1=torch.randn(2, 5))
        assert list(out.shape) == [2, 15]


class TestPermute:
    def test_returns_module(self):
        props = {"dims": "0, 2, 1"}
        module = NODE_BUILDERS["ml.structural.permute"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"dims": "0, 2, 1"}
        module = NODE_BUILDERS["ml.structural.permute"](props, {})
        x = torch.randn(2, 10, 5)
        out = module(x)
        assert list(out.shape) == [2, 5, 10]


class TestSequencePool:
    def test_returns_module(self):
        props = {"mode": "last"}
        module = NODE_BUILDERS["ml.structural.sequence_pool"](props, {})
        assert isinstance(module, nn.Module)

    def test_last_mode(self):
        props = {"mode": "last"}
        module = NODE_BUILDERS["ml.structural.sequence_pool"](props, {})
        x = torch.randn(2, 10, 256)
        out = module(x)
        assert list(out.shape) == [2, 256]

    def test_mean_mode(self):
        props = {"mode": "mean"}
        module = NODE_BUILDERS["ml.structural.sequence_pool"](props, {})
        x = torch.randn(2, 10, 256)
        out = module(x)
        assert list(out.shape) == [2, 256]


class TestReparameterize:
    def test_returns_module(self):
        module = NODE_BUILDERS["ml.structural.reparameterize"]({}, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        module = NODE_BUILDERS["ml.structural.reparameterize"]({}, {})
        mean = torch.randn(2, 128)
        logvar = torch.randn(2, 128)
        out = module(mean=mean, logvar=logvar)
        assert list(out.shape) == [2, 128]


# --- Loss ---


class TestVAELoss:
    def test_returns_module(self):
        props = {"beta": 1.0}
        module = NODE_BUILDERS["ml.loss.vae"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_is_scalar(self):
        props = {"beta": 1.0}
        module = NODE_BUILDERS["ml.loss.vae"](props, {})
        recon = torch.randn(2, 784)
        orig = torch.randn(2, 784)
        mean = torch.randn(2, 32)
        logvar = torch.randn(2, 32)
        out = module(reconstruction=recon, original=orig, mean=mean, logvar=logvar)
        assert out.dim() == 0


class TestGANLoss:
    def test_returns_module(self):
        props = {"labelSmoothing": 0.1}
        module = NODE_BUILDERS["ml.loss.gan"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_is_scalar(self):
        props = {"labelSmoothing": 0.1}
        module = NODE_BUILDERS["ml.loss.gan"](props, {})
        real = torch.randn(2, 1)
        fake = torch.randn(2, 1)
        out = module(real_scores=real, fake_scores=fake)
        assert out.dim() == 0


# --- GAN / Diffusion ---


class TestNoiseInput:
    def test_returns_module(self):
        props = {"latentDim": 100, "batchSize": 4}
        module = NODE_BUILDERS["ml.gan.noise_input"](props, {})
        assert isinstance(module, nn.Module)


class TestTimestepEmbed:
    def test_returns_module(self):
        props = {"embedDim": 128}
        module = NODE_BUILDERS["ml.diffusion.timestep_embed"](props, {})
        assert isinstance(module, nn.Module)

    def test_output_shape(self):
        props = {"embedDim": 128}
        module = NODE_BUILDERS["ml.diffusion.timestep_embed"](props, {})
        t = torch.tensor([5, 10])
        out = module(t)
        assert list(out.shape) == [2, 128]
