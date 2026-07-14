"""Tests for backend/node_viz.py — FORWARD_VIZ registry and transformation functions.

Each forward viz function signature:
    fn(node_type, module, input_tensor, output, inputs, out_dict) -> dict

Return dicts contain: transformation (dict with 'type' key), insight (optional str).
"""

import torch
import torch.nn as nn
import pytest

from visualize.node_viz import (
    FORWARD_VIZ,
    get_forward_viz,
    get_backward_viz,
    forward_viz_default,
    compact_stats_with_norm,
    param_grad_stats,
)
from engine.node_builders import MHAWrapper, AttentionModule, ConcatModule

VALID_RESULT_KEYS = {"transformation", "insight"}


# ============================================================================
# Registry completeness
# ============================================================================


class TestRegistryCompleteness:
    def test_forward_viz_values_are_callable(self):
        for key, fn in FORWARD_VIZ.items():
            assert callable(fn), f"FORWARD_VIZ['{key}'] is not callable"

    def test_forward_viz_is_not_empty(self):
        assert len(FORWARD_VIZ) > 0


# ============================================================================
# Forward viz returns valid structure
# ============================================================================


class TestForwardVizStructure:
    def test_conv2d_forward_viz(self):
        module = nn.Conv2d(3, 16, 3, padding=1)
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.conv2d"]
        result = fn("ml.layers.conv2d", module, x, output, {"in": x}, output)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_RESULT_KEYS
        assert "transformation" in result
        assert result["transformation"]["type"] == "conv2d"
        # Conv2d should have input, output, kernels
        t = result["transformation"]
        assert "input" in t
        assert "output" in t
        assert "kernels" in t
        assert t["kernels"] is not None
        assert t["kernels"]["totalFilters"] == 16

    def test_linear_forward_viz(self):
        module = nn.Linear(64, 10)
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.linear"]
        result = fn("ml.layers.linear", module, x, output, {"in": x}, output)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_RESULT_KEYS
        assert "transformation" in result
        assert result["transformation"]["type"] == "linear"
        t = result["transformation"]
        assert t["inputDim"] == 64
        assert t["outputDim"] == 10
        assert len(t["inputVector"]) > 0
        assert len(t["outputVector"]) > 0

    def test_relu_forward_viz(self):
        module = nn.ReLU()
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.relu"]
        result = fn("ml.activations.relu", module, x, output, {"in": x}, output)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_RESULT_KEYS
        assert "insight" in result
        assert result["transformation"]["type"] == "activation"
        assert result["transformation"]["fn"] == "relu"
        assert "deadFraction" in result["transformation"]

    def test_sigmoid_forward_viz(self):
        module = nn.Sigmoid()
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.sigmoid"]
        result = fn("ml.activations.sigmoid", module, x, output, {"in": x}, output)
        assert result["transformation"]["type"] == "activation"
        assert result["transformation"]["fn"] == "sigmoid"
        assert "saturatedFraction" in result["transformation"]

    def test_batchnorm_forward_viz(self):
        module = nn.BatchNorm2d(16)
        module.eval()
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.batchnorm2d"]
        result = fn("ml.layers.batchnorm2d", module, x, output, {"in": x}, output)
        assert result["transformation"]["type"] == "norm"
        assert "inputHist" in result["transformation"]
        assert "outputHist" in result["transformation"]

    def test_maxpool_forward_viz(self):
        module = nn.MaxPool2d(2)
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.maxpool2d"]
        result = fn("ml.layers.maxpool2d", module, x, output, {"in": x}, output)
        assert result["transformation"]["type"] == "pool"
        assert result["transformation"]["poolKind"] == "max"
        assert "input" in result["transformation"]
        assert "output" in result["transformation"]

    def test_flatten_forward_viz(self):
        module = nn.Flatten()
        x = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.flatten"]
        result = fn("ml.layers.flatten", module, x, output, {"in": x}, output)
        assert result["transformation"]["type"] == "flatten"
        assert result["transformation"]["outputLength"] == 256

    def test_conv2d_scatter_has_points(self):
        """Activation scatter should include sample points."""
        module = nn.ReLU()
        x = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.relu"]
        result = fn("ml.activations.relu", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert len(t["points"]) > 0
        assert "x" in t["points"][0]
        assert "y" in t["points"][0]


# ============================================================================
# Default fallback
# ============================================================================


class TestDefaultFallback:
    def test_get_forward_viz_unknown_type(self):
        """Unknown node type should use the default fallback and not crash."""
        output = torch.randn(1, 32)
        result = get_forward_viz(
            "ml.unknown.nonexistent", None, None, output, {}, output
        )
        assert isinstance(result, dict)
        assert "transformation" in result
        assert result["transformation"]["type"] == "default"

    def test_get_backward_viz_unknown_type(self):
        """Backward viz returns empty dict (placeholder)."""
        gradient = torch.randn(1, 32)
        result = get_backward_viz(
            "ml.unknown.nonexistent", None, None, gradient
        )
        assert isinstance(result, dict)

    def test_get_forward_viz_none_output(self):
        """Default fallback with None output should not crash."""
        result = get_forward_viz(
            "ml.unknown.nonexistent", None, None, None, {}, None
        )
        assert isinstance(result, dict)

    def test_get_backward_viz_none_gradient(self):
        """Default fallback with None gradient should not crash."""
        result = get_backward_viz(
            "ml.unknown.nonexistent", None, None, None
        )
        assert isinstance(result, dict)


# ============================================================================
# Activation variants (tanh, gelu, leaky_relu, softmax)
# ============================================================================


class TestActivationVariants:
    def test_tanh_forward_viz(self):
        module = nn.Tanh()
        x = torch.randn(1, 64) * 3  # spread so some saturate
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.tanh"]
        result = fn("ml.activations.tanh", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "activation"
        assert t["fn"] == "tanh"
        assert "saturatedFraction" in t
        assert 0.0 <= t["saturatedFraction"] <= 1.0
        assert len(t["points"]) > 0
        # shared histograms + range are attached when both inp and out present
        assert "inputHist" in t and "outputHist" in t
        assert t["sharedXMin"] <= t["sharedXMax"]
        assert "saturated region" in result["insight"]

    def test_gelu_forward_viz(self):
        module = nn.GELU()
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.gelu"]
        result = fn("ml.activations.gelu", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "activation"
        assert t["fn"] == "gelu"
        assert len(t["points"]) > 0
        assert "Smooth nonlinearity" in result["insight"]

    def test_leaky_relu_forward_viz(self):
        module = nn.LeakyReLU(negative_slope=0.2)
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.leaky_relu"]
        result = fn("ml.activations.leaky_relu", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "activation"
        assert t["fn"] == "leaky_relu"
        # negative_slope should be read off the module
        assert t["negativeSlope"] == pytest.approx(0.2)
        assert "0.2" in result["insight"]

    def test_softmax_forward_viz(self):
        module = nn.Softmax(dim=1)
        x = torch.randn(1, 10)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.softmax"]
        result = fn("ml.activations.softmax", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "softmax"
        assert len(t["logits"]) == 10
        assert len(t["probabilities"]) == 10
        # probabilities sum to ~1
        assert sum(t["probabilities"]) == pytest.approx(1.0, abs=1e-4)
        # topK is sorted descending and length capped at 5
        assert len(t["topK"]) == 5
        vals = [e["value"] for e in t["topK"]]
        assert vals == sorted(vals, reverse=True)

    def test_softmax_none_output(self):
        """Softmax with no output tensor degrades to empty lists, no crash."""
        fn = FORWARD_VIZ["ml.activations.softmax"]
        result = fn("ml.activations.softmax", None, None, None, {}, None)
        t = result["transformation"]
        assert t["logits"] == []
        assert t["probabilities"] == []
        assert t["topK"] == []


# ============================================================================
# Normalization variants
# ============================================================================


class TestNormVariants:
    def _run(self, node_type, module, x):
        module.eval()
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ[node_type]
        return fn(node_type, module, x, output, {"in": x}, output)

    def test_batchnorm1d_forward_viz(self):
        result = self._run("ml.layers.batchnorm1d", nn.BatchNorm1d(16), torch.randn(4, 16))
        t = result["transformation"]
        assert t["normKind"] == "BatchNorm1d"
        assert "inputHist" in t and "outputHist" in t
        # affine params exposed as gamma/beta
        assert "gamma" in t and "beta" in t

    def test_layernorm_forward_viz(self):
        result = self._run("ml.layers.layernorm", nn.LayerNorm(32), torch.randn(2, 32))
        t = result["transformation"]
        assert t["normKind"] == "LayerNorm"
        assert t["outputHist"]["counts"]  # non-empty histogram
        assert "features per sample" in result["insight"]

    def test_groupnorm_forward_viz(self):
        result = self._run("ml.layers.groupnorm", nn.GroupNorm(4, 16), torch.randn(2, 16, 8, 8))
        t = result["transformation"]
        assert t["normKind"] == "GroupNorm"
        assert "groups" in result["insight"]

    def test_instancenorm2d_forward_viz(self):
        # affine=True so gamma/beta appear
        module = nn.InstanceNorm2d(16, affine=True)
        result = self._run("ml.layers.instancenorm2d", module, torch.randn(2, 16, 8, 8))
        t = result["transformation"]
        assert t["normKind"] == "InstanceNorm2d"
        assert "gamma" in t and "beta" in t

    def test_norm_none_input(self):
        """Norm with no tensors returns empty histogram placeholders."""
        fn = FORWARD_VIZ["ml.layers.layernorm"]
        result = fn("ml.layers.layernorm", None, None, None, {}, None)
        t = result["transformation"]
        assert t["inputHist"] == {"bins": [], "counts": [], "mean": 0, "std": 0}
        assert t["outputHist"] == {"bins": [], "counts": [], "mean": 0, "std": 0}


# ============================================================================
# Pooling variants + upsample
# ============================================================================


class TestPoolingVariants:
    def _run(self, node_type, module, x):
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ[node_type]
        return fn(node_type, module, x, output, {"in": x}, output), output

    def test_avgpool2d_forward_viz(self):
        result, _ = self._run("ml.layers.avgpool2d", nn.AvgPool2d(2), torch.randn(1, 8, 8, 8))
        t = result["transformation"]
        assert t["type"] == "pool"
        assert t["poolKind"] == "avg"
        # raw single-channel grids populated for 4D in/out
        assert t["rawInputH"] == 8 and t["rawInputW"] == 8
        assert len(t["rawOutput"]) == 4  # 8 -> 4 after pool-2
        assert t["poolSize"] == [2, 2]
        assert "Average pooling" in result["insight"]

    def test_adaptive_avgpool2d_channel_values(self):
        module = nn.AdaptiveAvgPool2d((1, 1))
        result, output = self._run("ml.layers.adaptive_avgpool2d", module, torch.randn(1, 12, 8, 8))
        t = result["transformation"]
        assert t["poolKind"] == "adaptive_avg"
        assert output.shape[-2:] == (1, 1)
        # 1x1 output => per-channel scalar values populated (capped at 16)
        assert "channelValues" in t
        assert len(t["channelValues"]) == 12
        assert "Adaptive pool" in result["insight"]

    def test_maxpool1d_forward_viz(self):
        # 1D pooling produces 3D tensors -> no 4D feature maps, EMPTY placeholders
        module = nn.MaxPool1d(2)
        result, _ = self._run("ml.layers.maxpool1d", module, torch.randn(1, 8, 16))
        t = result["transformation"]
        assert t["type"] == "pool"
        assert t["poolKind"] == "max"
        assert t["input"] == {"maps": [], "channels": 0, "showing": 0, "height": 0, "width": 0}

    def test_upsample_forward_viz(self):
        module = nn.Upsample(scale_factor=2, mode="nearest")
        result, output = self._run("ml.layers.upsample", module, torch.randn(1, 4, 4, 4))
        t = result["transformation"]
        assert t["type"] == "upsample"
        assert t["input"]["channels"] == 4
        assert output.shape[-1] == 8
        assert "Upsampled from 4x4 to 8x8" in result["insight"]


# ============================================================================
# Dropout
# ============================================================================


class TestDropout:
    def test_dropout_forward_viz(self):
        module = nn.Dropout(p=0.5)
        module.train()  # active so some values are zeroed
        torch.manual_seed(0)
        x = torch.randn(1, 128) + 5.0  # nonzero input
        output = module(x)
        fn = FORWARD_VIZ["ml.layers.dropout"]
        result = fn("ml.layers.dropout", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "dropout"
        assert t["totalElements"] == 128
        assert t["inputNonzero"] == 128
        # training-mode dropout zeroes some, so fewer nonzeros out
        assert t["outputNonzero"] < t["inputNonzero"]
        assert "inputHist" in t and "outputHist" in t
        assert "Randomly zeroes" in result["insight"]

    def test_dropout2d_feature_maps(self):
        module = nn.Dropout2d(p=0.5)
        module.eval()
        x = torch.randn(1, 8, 6, 6)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.dropout2d"]
        result = fn("ml.layers.dropout2d", module, x, output, {"in": x}, output)
        t = result["transformation"]
        # 4D input/output => feature maps attached
        assert "inputMaps" in t and "outputMaps" in t
        assert t["inputMaps"]["channels"] == 8


# ============================================================================
# Flatten / reshape / concat / add
# ============================================================================


class TestStructural:
    def test_reshape_forward_viz(self):
        x = torch.randn(1, 3, 4, 4)
        output = x.reshape(1, 48)
        fn = FORWARD_VIZ["ml.structural.reshape"]
        result = fn("ml.structural.reshape", None, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "reshape"
        assert t["inputShape"] == [1, 3, 4, 4]
        assert t["outputShape"] == [1, 48]
        assert "inputFmaps" in t  # 4D input
        assert "outputVector" in t  # 2D output
        assert "same data, different layout" in result["insight"]

    def test_add_forward_viz(self):
        a = torch.randn(1, 16)
        b = torch.randn(1, 16)
        output = a + b
        inputs = {"in_0": a, "in_1": b}
        fn = FORWARD_VIZ["ml.structural.add"]
        result = fn("ml.structural.add", None, None, output, inputs, output)
        t = result["transformation"]
        assert t["type"] == "add"
        assert len(t["inputs"]) == 2
        assert all("vector" in e for e in t["inputs"])
        assert t["outputVector"] is not None
        assert "Element-wise sum of 2 inputs" in result["insight"]

    def test_add_feature_maps(self):
        """4D inputs → residual add shows feature maps."""
        a = torch.randn(1, 4, 6, 6)
        b = torch.randn(1, 4, 6, 6)
        output = a + b
        inputs = {"in_0": a, "in_1": b}
        fn = FORWARD_VIZ["ml.structural.add"]
        result = fn("ml.structural.add", None, None, output, inputs, output)
        t = result["transformation"]
        assert all("featureMaps" in e for e in t["inputs"])
        assert t["output"] is not None and t["output"]["channels"] == 4

    def test_concat_forward_viz(self):
        a = torch.randn(1, 8, 6, 6)
        b = torch.randn(1, 8, 6, 6)
        module = ConcatModule(dim=1)
        with torch.no_grad():
            output = module(in_0=a, in_1=b)
        inputs = {"in_0": a, "in_1": b}
        fn = FORWARD_VIZ["ml.structural.concat"]
        result = fn("ml.structural.concat", module, None, output, inputs, output)
        t = result["transformation"]
        assert t["type"] == "concat"
        assert t["dim"] == 1
        assert len(t["inputs"]) == 2
        assert t["outputShape"] == [1, 16, 6, 6]
        assert "outputFmaps" in t
        assert "Concatenated 2 inputs along dimension 1" in result["insight"]

    def test_concat_constant_input(self):
        """A constant channel (e.g. timestep) is flagged, not rendered as data."""
        const = torch.full((1, 1, 6, 6), 0.7)
        data = torch.randn(1, 8, 6, 6)
        module = ConcatModule(dim=1)
        inputs = {"in_0": data, "in_1": const}
        with torch.no_grad():
            output = module(in_0=data, in_1=const)
        fn = FORWARD_VIZ["ml.structural.concat"]
        result = fn("ml.structural.concat", module, None, output, inputs, output)
        entries = result["transformation"]["inputs"]
        const_entry = next(e for e in entries if e.get("isConstant"))
        assert const_entry["constantValue"] == pytest.approx(0.7)

    def test_reparameterize_forward_viz(self):
        mean = torch.randn(1, 8)
        logvar = torch.randn(1, 8)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        inputs = {"mean": mean, "logvar": logvar}
        fn = FORWARD_VIZ["ml.structural.reparameterize"]
        result = fn("ml.structural.reparameterize", None, None, z, inputs, z)
        t = result["transformation"]
        assert t["type"] == "reparameterize"
        assert len(t["meanValues"]) == 8
        assert len(t["logvarValues"]) == 8
        assert t["latentDim"] == 8
        assert "meanHist" in t and "logvarHist" in t

    def test_permute_uses_default(self):
        """permute has no dedicated transformation; uses default + its insight."""
        x = torch.randn(1, 4, 8)
        output = x.permute(0, 2, 1)
        fn = FORWARD_VIZ["ml.structural.permute"]
        result = fn("ml.structural.permute", None, x, output, {"in": x}, output)
        assert result["transformation"]["type"] == "default"
        assert result["insight"] == "Reordered tensor dimensions"


# ============================================================================
# Embedding / recurrent
# ============================================================================


class TestSequenceLayers:
    def test_embedding_forward_viz(self):
        module = nn.Embedding(50, 16)
        tokens = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            output = module(tokens)  # [1, 10, 16]
        fn = FORWARD_VIZ["ml.layers.embedding"]
        result = fn("ml.layers.embedding", module, tokens, output, {"in": tokens}, output)
        t = result["transformation"]
        assert t["type"] == "default"
        assert "Looked up embeddings of dim 16" in result["insight"]

    def test_lstm_forward_viz(self):
        module = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        x = torch.randn(1, 10, 16)
        with torch.no_grad():
            output, _ = module(x)  # [1, 10, 32]
        fn = FORWARD_VIZ["ml.layers.lstm"]
        result = fn("ml.layers.lstm", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "default"
        assert "vector" in t
        assert result["insight"] == "Processed sequence through recurrent cells"

    def test_gru_forward_viz(self):
        module = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
        x = torch.randn(1, 5, 8)
        with torch.no_grad():
            output, _ = module(x)
        fn = FORWARD_VIZ["ml.layers.gru"]
        result = fn("ml.layers.gru", module, x, output, {"in": x}, output)
        assert result["transformation"]["type"] == "default"
        assert "recurrent" in result["insight"]


# ============================================================================
# Attention (MHA + SDPA)
# ============================================================================


class TestAttention:
    def test_mha_forward_viz(self):
        mha = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        module = MHAWrapper(mha, is_causal=False)
        module.eval()
        q = torch.randn(1, 6, 16)
        k = torch.randn(1, 6, 16)
        v = torch.randn(1, 6, 16)
        with torch.no_grad():
            output = module(q, k, v)
        inputs = {"query": q, "key": k, "value": v}
        fn = FORWARD_VIZ["ml.layers.multihead_attention"]
        result = fn("ml.layers.multihead_attention", module, q, output, inputs, {})
        t = result["transformation"]
        assert t["type"] == "attention"
        assert t["numHeads"] == 4
        assert t["seqLen"] == 6
        assert t["causalMask"] is False
        # per-head weights: [H][S][S]
        assert len(t["perHeadWeights"]) == 4
        assert len(t["perHeadWeights"][0]) == 6
        assert len(t["avgWeights"]) == 6
        assert len(t["headEntropy"]) == 4
        assert "focusRow" in t and t["focusRow"]["queryIndex"] == 5
        assert "Multi-head attention" in result["insight"]

    def test_mha_causal_mask(self):
        mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        module = MHAWrapper(mha, is_causal=True)
        module.eval()
        q = torch.randn(1, 4, 8)
        inputs = {"query": q, "key": q, "value": q}
        with torch.no_grad():
            output = module(q, q, q)
        fn = FORWARD_VIZ["ml.layers.multihead_attention"]
        result = fn("ml.layers.multihead_attention", module, q, output, inputs, {})
        t = result["transformation"]
        assert t["causalMask"] is True
        # causal: query position 0 attends only to itself (upper-triangle masked to 0)
        avg = t["avgWeights"]
        assert avg[0][1] == pytest.approx(0.0, abs=1e-5)
        assert "causal mask" in result["insight"]

    def test_mha_missing_inputs_falls_back(self):
        """No q/k/v tensors → default transformation, no crash."""
        mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        module = MHAWrapper(mha)
        output = torch.randn(1, 4, 8)
        fn = FORWARD_VIZ["ml.layers.multihead_attention"]
        result = fn("ml.layers.multihead_attention", module, None, output, {}, {})
        assert result["transformation"]["type"] == "default"

    def test_sdpa_forward_viz(self):
        module = AttentionModule(is_causal=False)
        q = torch.randn(1, 5, 16)
        k = torch.randn(1, 5, 16)
        v = torch.randn(1, 5, 16)
        inputs = {"query": q, "key": k, "value": v}
        with torch.no_grad():
            output = module(q, k, v)
        fn = FORWARD_VIZ["ml.layers.attention"]
        result = fn("ml.layers.attention", module, q, output, inputs, {})
        t = result["transformation"]
        assert t["type"] == "attention"
        assert t["numHeads"] == 1  # 3D q/k/v -> single head
        assert t["seqLen"] == 5
        # each attention row is a probability distribution summing to ~1
        row_sum = sum(t["avgWeights"][0])
        assert row_sum == pytest.approx(1.0, abs=1e-4)
        assert "Scaled dot-product attention" in result["insight"]

    def test_sdpa_causal(self):
        module = AttentionModule(is_causal=True)
        q = torch.randn(1, 4, 8)
        inputs = {"query": q, "key": q, "value": q}
        with torch.no_grad():
            output = module(q, q, q)
        fn = FORWARD_VIZ["ml.layers.attention"]
        result = fn("ml.layers.attention", module, q, output, inputs, {})
        t = result["transformation"]
        assert t["causalMask"] is True
        assert t["avgWeights"][0][1] == pytest.approx(0.0, abs=1e-5)


# ============================================================================
# Convolution variants + upsample-style conv
# ============================================================================


class TestConvVariants:
    def test_conv_transpose2d_forward_viz(self):
        module = nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1)
        x = torch.randn(1, 4, 4, 4)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.conv_transpose2d"]
        result = fn("ml.layers.conv_transpose2d", module, x, output, {"in": x}, output)
        t = result["transformation"]
        assert t["type"] == "conv2d"
        assert t["kernels"] is not None
        # ConvTranspose2d weight is [in, out, kH, kW]; extract_conv_kernels reports
        # shape[0] (= in_channels here) as totalFilters.
        assert t["kernels"]["totalFilters"] == 4
        assert t["kernels"]["kernelH"] == 3 and t["kernels"]["kernelW"] == 3


# ============================================================================
# Loss nodes
# ============================================================================


class TestLossViz:
    def test_cross_entropy_forward_viz(self):
        preds = torch.randn(1, 10)
        labels = torch.tensor([3])
        inputs = {"predictions": preds, "labels": labels}
        output = torch.tensor(1.5)
        fn = FORWARD_VIZ["ml.loss.cross_entropy"]
        result = fn("ml.loss.cross_entropy", None, None, output, inputs, output)
        t = result["transformation"]
        assert t["type"] == "cross_entropy"
        assert t["trueLabel"] == 3
        assert len(t["probabilities"]) == 10
        assert sum(t["probabilities"]) == pytest.approx(1.0, abs=1e-4)
        assert t["loss"] > 0
        assert "Loss = -log" in result["insight"]

    def test_mse_loss_forward_viz(self):
        preds = torch.randn(1, 1, 8, 8)
        targets = torch.randn(1, 1, 8, 8)
        inputs = {"predictions": preds, "labels": targets}
        output = torch.nn.functional.mse_loss(preds, targets)
        fn = FORWARD_VIZ["ml.loss.mse"]
        result = fn("ml.loss.mse", None, None, output, inputs, output)
        t = result["transformation"]
        assert t["type"] == "mse_loss"
        assert t["numElements"] == 64
        assert t["meanSquared"] >= 0
        # 4D preds/targets => spatial error map
        assert "errorMap" in t
        assert t["errorH"] == 8 and t["errorW"] == 8

    def test_gan_loss_forward_viz(self):
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        inputs = {"real_scores": real, "fake_scores": fake}
        output = torch.tensor(1.2)
        fn = FORWARD_VIZ["ml.loss.gan"]
        result = fn("ml.loss.gan", None, None, output, inputs, output)
        t = result["transformation"]
        assert t["type"] == "gan_loss"
        assert 0.0 <= t["realProb"] <= 1.0
        assert 0.0 <= t["fakeProb"] <= 1.0
        assert "dLossReal" in t and "dLossFake" in t

    def test_vae_loss_forward_viz(self):
        recon = torch.rand(1, 1, 8, 8)
        original = torch.rand(1, 1, 8, 8)
        mean = torch.randn(1, 8)
        logvar = torch.randn(1, 8)
        inputs = {"reconstruction": recon, "original": original,
                  "mean": mean, "logvar": logvar}
        output = torch.tensor(0.9)
        fn = FORWARD_VIZ["ml.loss.vae"]
        result = fn("ml.loss.vae", None, None, output, inputs, output)
        t = result["transformation"]
        assert t["type"] == "vae_loss"
        assert t["reconLoss"] >= 0
        assert "klLoss" in t
        assert "originalFmaps" in t and "reconFmaps" in t

    def test_loss_scalar_fallback(self):
        """Unknown loss inputs fall back to plain scalar."""
        output = torch.tensor(2.5)
        fn = FORWARD_VIZ["ml.loss.mse"]
        result = fn("ml.loss.mse", None, None, output, {}, output)
        # No preds/targets -> mse branch still returns type mse_loss with just loss
        t = result["transformation"]
        assert t["type"] == "mse_loss"
        assert t["loss"] == pytest.approx(2.5)


# ============================================================================
# Data nodes
# ============================================================================


class TestDataViz:
    def test_mnist_data_forward_viz(self):
        output = torch.randn(1, 1, 28, 28)
        fn = FORWARD_VIZ["data.mnist"]
        result = fn("data.mnist", None, None, output, {}, output)
        t = result["transformation"]
        assert t["type"] == "data"
        assert "featureMaps" in t
        # mnist has a denormalizer -> histograms present
        assert "rawHist" in t and "normHist" in t
        assert "normalized" in result["insight"]

    def test_text_data_forward_viz(self):
        """Non-image data (e.g. token ids) shows a vector, not feature maps."""
        output = torch.randn(1, 32)
        fn = FORWARD_VIZ["data.tiny_shakespeare"]
        result = fn("data.tiny_shakespeare", None, None, output, {}, output)
        t = result["transformation"]
        assert t["type"] == "data"
        assert "vector" in t


# ============================================================================
# Backward path (placeholder registry + legacy stat helpers)
# ============================================================================


class TestBackwardViz:
    def test_backward_viz_known_type_is_placeholder(self):
        """get_backward_viz is a placeholder: returns {} even for known layers."""
        module = nn.Conv2d(3, 8, 3)
        grad = torch.randn(1, 8, 6, 6)
        result = get_backward_viz("ml.layers.conv2d", module, None, grad)
        assert result == {}

    def test_backward_viz_linear_placeholder(self):
        module = nn.Linear(16, 4)
        grad = torch.randn(1, 4)
        assert get_backward_viz("ml.layers.linear", module, None, grad) == {}

    def test_backward_viz_activation_placeholder(self):
        grad = torch.randn(1, 32)
        assert get_backward_viz("ml.activations.relu", nn.ReLU(), None, grad) == {}

    def test_compact_stats_with_norm_conv_gradient(self):
        """Gradient tensor → mean/std/min/max/norm stats used by backprop viz."""
        module = nn.Conv2d(3, 8, 3)
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        out = module(x)
        out.sum().backward()
        stats = compact_stats_with_norm(x.grad)
        assert set(stats.keys()) == {"mean", "std", "min", "max", "norm"}
        assert stats["min"] <= stats["max"]
        assert stats["norm"] >= 0

    def test_compact_stats_empty_tensor(self):
        assert compact_stats_with_norm(torch.empty(0)) == {}

    def test_param_grad_stats_linear(self):
        """After backward, param_grad_stats reports grad/weight norms + health."""
        module = nn.Linear(16, 4)
        x = torch.randn(2, 16)
        module(x).sum().backward()
        stats = param_grad_stats(module)
        assert set(stats.keys()) == {"gradNorm", "weightNorm", "ratio", "health"}
        assert stats["gradNorm"] >= 0
        assert stats["health"] in {"vanishing", "exploding", "healthy"}

    def test_param_grad_stats_no_grad_returns_none(self):
        """A module whose params have no .grad yet returns None."""
        module = nn.Linear(8, 2)
        assert param_grad_stats(module) is None

    def test_param_grad_stats_none_module(self):
        assert param_grad_stats(None) is None


# ============================================================================
# Additional graceful-degradation
# ============================================================================


class TestGracefulDegradation:
    def test_flatten_none_output(self):
        fn = FORWARD_VIZ["ml.layers.flatten"]
        result = fn("ml.layers.flatten", None, None, None, {}, None)
        t = result["transformation"]
        assert t["outputLength"] == 0
        assert t["flatPixels"] == []

    def test_pooling_none_tensors(self):
        fn = FORWARD_VIZ["ml.layers.maxpool2d"]
        result = fn("ml.layers.maxpool2d", None, None, None, {}, None)
        t = result["transformation"]
        assert t["input"] == {"maps": [], "channels": 0, "showing": 0, "height": 0, "width": 0}
        assert result["insight"] is None

    def test_reshape_none_tensors(self):
        fn = FORWARD_VIZ["ml.structural.reshape"]
        result = fn("ml.structural.reshape", None, None, None, {}, None)
        t = result["transformation"]
        assert t["inputShape"] == []
        assert t["outputShape"] == []

    def test_dropout_none_output(self):
        """Dropout with input but no output skips the nonzero counters, no crash."""
        x = torch.randn(1, 16)
        fn = FORWARD_VIZ["ml.layers.dropout"]
        result = fn("ml.layers.dropout", nn.Dropout(), x, None, {"in": x}, None)
        assert result["transformation"]["type"] == "dropout"
        assert "totalElements" not in result["transformation"]

    def test_embedding_none_output(self):
        fn = FORWARD_VIZ["ml.layers.embedding"]
        result = fn("ml.layers.embedding", nn.Embedding(10, 4), None, None, {}, None)
        assert result["transformation"]["type"] == "default"
        assert result["insight"] is None
