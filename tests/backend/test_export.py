"""Tests for backend/export_python.py — Python code generation from NodeTorch graphs.

Verifies that export_to_python generates valid, complete Python files for
various graph configurations: classification, autoregressive, VAE, and GAN.
"""

import json
import glob
import os
import pytest

from export_python import export_to_python


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(nodes, edges, name="Test"):
    """Wrap nodes/edges into a full graph JSON structure."""
    return {
        "version": "1.0",
        "graph": {
            "id": "test",
            "name": name,
            "nodes": nodes,
            "edges": edges,
        },
    }


def _n(id, type, props=None, x=0, y=0):
    """Shorthand to create a node dict."""
    return {"id": id, "type": type, "position": {"x": x, "y": y}, "properties": props or {}}


def _e(id, src_node, src_port, tgt_node, tgt_port):
    """Shorthand to create an edge dict."""
    return {
        "id": id,
        "source": {"nodeId": src_node, "portId": src_port},
        "target": {"nodeId": tgt_node, "portId": tgt_port},
    }


# ---------------------------------------------------------------------------
# Basic export structure
# ---------------------------------------------------------------------------

class TestBasicExport:
    """Minimal classification graph: MNIST → Flatten → Linear → Loss → Optimizer."""

    @pytest.fixture
    def graph(self):
        return _make_graph(
            nodes=[
                _n("data", "data.mnist", {"batchSize": 32}),
                _n("flat", "ml.layers.flatten", {}),
                _n("fc", "ml.layers.linear", {"outFeatures": 10}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 5}),
            ],
            edges=[
                _e("e1", "data", "out", "flat", "in"),
                _e("e2", "flat", "out", "fc", "in"),
                _e("e3", "fc", "out", "loss", "predictions"),
                _e("e4", "data", "labels", "loss", "labels"),
                _e("e5", "loss", "out", "opt", "loss"),
            ],
        )

    def test_returns_string(self, graph):
        code = export_to_python(graph)
        assert isinstance(code, str)

    def test_contains_model_class(self, graph):
        code = export_to_python(graph)
        assert "class Model(nn.Module):" in code

    def test_contains_imports(self, graph):
        code = export_to_python(graph)
        assert "import torch" in code
        assert "import torch.nn as nn" in code

    def test_contains_training_loop(self, graph):
        code = export_to_python(graph)
        assert "optimizer" in code.lower()
        assert "loss" in code.lower()

    def test_contains_main_block(self, graph):
        code = export_to_python(graph)
        assert 'if __name__ == "__main__":' in code

    def test_contains_device_selection(self, graph):
        code = export_to_python(graph)
        assert "cuda" in code or "device" in code


# ---------------------------------------------------------------------------
# Helper class injection
# ---------------------------------------------------------------------------

class TestHelperClassInjection:
    def test_tokenizer_class_injected(self):
        graph = _make_graph(
            nodes=[
                _n("data", "data.imdb", {"batchSize": 32, "maxLen": 256, "vocabSize": 10000}),
                _n("tok", "ml.preprocessing.tokenizer", {"mode": "word", "vocabSize": 10000, "maxLen": 128}),
                _n("embed", "ml.layers.embedding", {"numEmbeddings": 10000, "embeddingDim": 64}),
                _n("pool", "ml.structural.sequence_pool", {"mode": "mean"}),
                _n("fc", "ml.layers.linear", {"outFeatures": 2}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 1}),
            ],
            edges=[
                _e("e1", "data", "out", "tok", "in"),
                _e("e2", "tok", "out", "embed", "in"),
                _e("e3", "embed", "out", "pool", "in"),
                _e("e4", "pool", "out", "fc", "in"),
                _e("e5", "fc", "out", "loss", "predictions"),
                _e("e6", "data", "labels", "loss", "labels"),
                _e("e7", "loss", "out", "opt", "loss"),
            ],
        )
        code = export_to_python(graph)
        assert "class Tokenizer(nn.Module):" in code
        assert "self.tok = Tokenizer(10000, 128)" in code

    def test_positional_encoding_class_injected(self):
        graph = _make_graph(
            nodes=[
                _n("data", "data.tiny_shakespeare", {"batchSize": 32, "seqLen": 64}),
                _n("embed", "ml.layers.embedding", {"numEmbeddings": 65, "embeddingDim": 64}),
                _n("pos", "ml.layers.positional_encoding", {"maxLen": 64, "encodingType": "learned"}),
                _n("pool", "ml.structural.sequence_pool", {"mode": "last"}),
                _n("fc", "ml.layers.linear", {"outFeatures": 65}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 1}),
            ],
            edges=[
                _e("e1", "data", "out", "embed", "in"),
                _e("e2", "embed", "out", "pos", "in"),
                _e("e3", "pos", "out", "pool", "in"),
                _e("e4", "pool", "out", "fc", "in"),
                _e("e5", "fc", "out", "loss", "predictions"),
                _e("e6", "data", "labels", "loss", "labels"),
                _e("e7", "loss", "out", "opt", "loss"),
            ],
        )
        code = export_to_python(graph)
        assert "class PositionalEncoding(nn.Module):" in code


# ---------------------------------------------------------------------------
# All presets export without error
# ---------------------------------------------------------------------------

class TestPresetExport:
    """Every preset JSON should export to valid Python code."""

    @pytest.fixture(params=sorted(glob.glob("model-presets/*.json")))
    def preset(self, request):
        path = request.param
        with open(path) as f:
            return json.load(f), os.path.basename(path)

    def test_exports_without_error(self, preset):
        graph, name = preset
        code = export_to_python(graph)
        assert isinstance(code, str), f"Preset {name} did not return a string"

    def test_has_model_or_generator_class(self, preset):
        graph, name = preset
        code = export_to_python(graph)
        has_model = "class Model" in code or "class Generator" in code or "class Encoder" in code
        assert has_model, f"Preset {name} missing model class"

    def test_has_main_block(self, preset):
        graph, name = preset
        code = export_to_python(graph)
        assert '__name__' in code, f"Preset {name} missing __main__ block"
