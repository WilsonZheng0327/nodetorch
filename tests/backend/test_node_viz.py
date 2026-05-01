"""Tests for backend/node_viz.py — FORWARD_VIZ registry and transformation functions.

Each forward viz function signature:
    fn(node_type, module, input_tensor, output, inputs, out_dict) -> dict

Return dicts contain: transformation (dict with 'type' key), insight (optional str).
"""

import torch
import torch.nn as nn
import pytest

from node_viz import (
    FORWARD_VIZ,
    get_forward_viz,
    get_backward_viz,
    forward_viz_default,
)

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
