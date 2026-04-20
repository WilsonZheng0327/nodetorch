"""Tests for backend/node_viz.py — FORWARD_VIZ and BACKWARD_VIZ registries.

Each forward viz function signature:
    fn(node_type, module, input_tensor, output, inputs, out_dict) -> dict

Each backward viz function signature:
    fn(module, activation, gradient) -> dict

Return dicts may contain keys: viz, extras, insight.
"""

import torch
import torch.nn as nn
import pytest

from node_viz import (
    FORWARD_VIZ,
    BACKWARD_VIZ,
    get_forward_viz,
    get_backward_viz,
    forward_viz_default,
    backward_viz_default,
)

VALID_VIZ_KEYS = {"viz", "extras", "insight"}


# ============================================================================
# Registry completeness
# ============================================================================


class TestRegistryCompleteness:
    def test_forward_viz_values_are_callable(self):
        for key, fn in FORWARD_VIZ.items():
            assert callable(fn), f"FORWARD_VIZ['{key}'] is not callable"

    def test_backward_viz_values_are_callable(self):
        for key, fn in BACKWARD_VIZ.items():
            assert callable(fn), f"BACKWARD_VIZ['{key}'] is not callable"

    def test_forward_viz_is_not_empty(self):
        assert len(FORWARD_VIZ) > 0

    def test_backward_viz_is_not_empty(self):
        assert len(BACKWARD_VIZ) > 0


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
        assert result.keys() <= VALID_VIZ_KEYS
        assert "viz" in result

    def test_linear_forward_viz(self):
        module = nn.Linear(64, 10)
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.linear"]
        result = fn("ml.layers.linear", module, x, output, {"in": x}, output)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_VIZ_KEYS
        assert "viz" in result

    def test_relu_forward_viz(self):
        module = nn.ReLU()
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.activations.relu"]
        result = fn("ml.activations.relu", module, x, output, {"in": x}, output)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_VIZ_KEYS
        assert "insight" in result

    def test_forward_viz_conv2d_has_extras(self):
        """Conv2d viz should include kernel visualization as extras."""
        module = nn.Conv2d(3, 16, 3, padding=1)
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.conv2d"]
        result = fn("ml.layers.conv2d", module, x, output, {"in": x}, output)
        assert "extras" in result
        assert isinstance(result["extras"], list)
        assert len(result["extras"]) > 0

    def test_forward_viz_linear_has_extras(self):
        """Linear viz should include weight matrix as extras."""
        module = nn.Linear(64, 10)
        x = torch.randn(1, 64)
        with torch.no_grad():
            output = module(x)
        fn = FORWARD_VIZ["ml.layers.linear"]
        result = fn("ml.layers.linear", module, x, output, {"in": x}, output)
        assert "extras" in result
        assert isinstance(result["extras"], list)


# ============================================================================
# Backward viz returns valid structure
# ============================================================================


class TestBackwardVizStructure:
    def test_conv2d_backward_viz(self):
        module = nn.Conv2d(3, 16, 3, padding=1)
        activation = torch.randn(1, 3, 8, 8)
        # Simulate a forward + backward to get real gradients
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        out = module(x)
        out.sum().backward()
        gradient = x.grad
        fn = BACKWARD_VIZ["ml.layers.conv2d"]
        result = fn(module, activation, gradient)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_VIZ_KEYS

    def test_linear_backward_viz(self):
        module = nn.Linear(64, 10)
        activation = torch.randn(1, 64)
        x = torch.randn(1, 64, requires_grad=True)
        out = module(x)
        out.sum().backward()
        gradient = x.grad
        fn = BACKWARD_VIZ["ml.layers.linear"]
        result = fn(module, activation, gradient)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_VIZ_KEYS
        assert "viz" in result

    def test_relu_backward_viz(self):
        module = nn.ReLU()
        activation = torch.randn(1, 64)
        gradient = torch.randn(1, 64)
        fn = BACKWARD_VIZ["ml.activations.relu"]
        result = fn(module, activation, gradient)
        assert isinstance(result, dict)
        assert result.keys() <= VALID_VIZ_KEYS
        assert "insight" in result


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
        # Default should still produce viz for a valid tensor
        assert "viz" in result

    def test_get_backward_viz_unknown_type(self):
        """Unknown node type should use the default fallback and not crash."""
        gradient = torch.randn(1, 32)
        result = get_backward_viz(
            "ml.unknown.nonexistent", None, None, gradient
        )
        assert isinstance(result, dict)
        assert "viz" in result

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
