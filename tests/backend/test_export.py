"""Tests for backend/export_python.py — Python code generation from NodeTorch graphs.

Verifies that export_to_python generates valid, complete Python files for
various graph configurations: classification, autoregressive, VAE, and GAN.
"""

import json
import glob
import os
import pytest

import torch
import torch.nn as nn

from export.datasets import _generate_dataset_code

from export import export_to_python


# ---------------------------------------------------------------------------
# Execution harness — actually RUN the generated model, don't just compile it.
# ---------------------------------------------------------------------------

def _exec_model_classes(code):
    """Exec only the top-level class definitions from a generated script.

    The generated file loads a real dataset and trains at module/`__main__`
    scope — neither is CI-safe. But every helper/subgraph/`Model` class is
    defined at column 0 with dimensions baked in as literals (no dependency on
    the dataset globals), so we can lift just those blocks out and exec them to
    get a live, constructible model. This turns "compiles" into "actually runs"
    without downloading anything.

    Returns the exec namespace (classes keyed by name).
    """
    ns = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    try:  # some transfer/pretrained models reference torchvision at construction
        import torchvision
        ns["torchvision"] = torchvision
    except ImportError:
        pass

    lines = code.split("\n")
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].startswith("class "):
            j = i + 1
            # a top-level class body is every following blank or indented line
            while j < len(lines) and (lines[j].strip() == "" or lines[j][:1] in (" ", "\t")):
                j += 1
            blocks.append("\n".join(lines[i:j]))
            i = j
        else:
            i += 1
    assert blocks, "generated script defined no top-level classes"
    exec("\n\n".join(blocks), ns)
    return ns


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
                _n("tok", "ml.preprocessing.tokenizer_word", {"vocabSize": 10000, "maxLen": 128}),
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


# ---------------------------------------------------------------------------
# Custom dataset (data.custom) — previously an unimplemented TODO stub
# ---------------------------------------------------------------------------

def _custom_node(**props):
    return _n("data", "data.custom", props)


class TestCustomDatasetExport:
    """`data.custom` used to export a `# TODO ... train_loader = None` stub. The
    generator now emits a self-contained CustomDataset mirroring the backend's
    interpreter across the (input_kind, task) grid."""

    def test_full_graph_export_has_no_stub_and_compiles(self):
        """A custom image-classification graph exports a complete, compilable script."""
        graph = _make_graph(
            nodes=[
                _custom_node(inputKind="image", task="classification", hfId="mnist",
                             inputColumn="image", labelColumn="label", channels=1,
                             imageSize=28, batchSize=32),
                _n("flat", "ml.layers.flatten", {}),
                _n("fc", "ml.layers.linear", {"outFeatures": 10}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 1}),
            ],
            edges=[
                _e("e1", "data", "out", "flat", "in"),
                _e("e2", "flat", "out", "fc", "in"),
                _e("e3", "fc", "out", "loss", "predictions"),
                _e("e4", "data", "labels", "loss", "labels"),
                _e("e5", "loss", "out", "opt", "loss"),
            ],
        )
        code = export_to_python(graph)
        assert "# TODO: Add dataset loading" not in code
        assert "train_loader = None" not in code
        assert "CustomDataset" in code
        assert "load_dataset" in code
        compile(code, "<custom-export>", "exec")  # whole script is valid Python

    # One representative node per supported (input_kind, task) × tokenizer cell.
    CELLS = [
        _custom_node(inputKind="image", task="classification", hfId="mnist",
                     inputColumn="image", channels=1, imageSize=28,
                     normalizeMean="0.5", normalizeStd="0.5"),
        _custom_node(inputKind="image", task="autoencoder", hfId="mnist",
                     inputColumn="image", channels=1, imageSize=28),
        _custom_node(inputKind="text", task="classification", hfId="ag_news",
                     tokenizer="word", inputColumn="text", labelColumn="label"),
        _custom_node(inputKind="text", task="classification", hfId="imdb",
                     tokenizer="char", inputColumn="text"),
        _custom_node(inputKind="text", task="classification", hfId="imdb",
                     tokenizer="bpe", inputColumn="text", vocabSize=500),
        _custom_node(inputKind="text", task="language_model", hfId="u/c",
                     tokenizer="char", inputColumn="text", seqLen=64),
        _custom_node(inputKind="text", task="language_model", hfId="u/c",
                     tokenizer="bpe", inputColumn="text", vocabSize=500, seqLen=64),
        _custom_node(inputKind="text", task="language_model", hfId="u/c",
                     tokenizer="word", inputColumn="text", seqLen=32),
        _custom_node(inputKind="vector", task="regression", hfId="u/tab",
                     inputColumn="features", labelColumn="target"),
        _custom_node(inputKind="vector", task="classification", hfId="u/tab",
                     inputColumn="features", labelColumn="y"),
    ]

    @pytest.mark.parametrize("node", CELLS, ids=lambda n: f"{n['properties']['inputKind']}-{n['properties']['task']}-{n['properties'].get('tokenizer','')}")
    def test_generated_dataset_code_is_valid(self, node):
        code = _generate_dataset_code(node)
        assert "# TODO" not in code
        assert "train_loader" in code
        # Compiles as Python (torch is imported earlier in the full script).
        compile("import torch\nimport torch.utils.data\n" + code, "<cell>", "exec")

    def test_bpe_tokenizer_class_is_emitted_for_bpe_custom_text(self):
        code = _generate_dataset_code(
            _custom_node(inputKind="text", task="classification", hfId="imdb",
                         tokenizer="bpe", inputColumn="text", vocabSize=500))
        assert "class BPETokenizer" in code
        assert "def encode" in code

    def test_language_model_path_emits_next_token_windows(self):
        code = _generate_dataset_code(
            _custom_node(inputKind="text", task="language_model", hfId="u/c",
                         tokenizer="char", inputColumn="text", seqLen=64))
        assert "chunk[:-1], chunk[1:]" in code  # input / shifted-target windows
        assert "VOCAB_SIZE" in code


# ---------------------------------------------------------------------------
# Executable export — the generated model runs, not just compiles
# ---------------------------------------------------------------------------

class TestExportedModelRuns:
    """`compile()` catches syntax errors; it does not catch a `Model` that
    references an undefined name, wires incompatible layer dims, or produces the
    wrong forward shape. These tests instantiate the generated model and push a
    synthetic batch through forward+backward — genuine execution, no dataset."""

    def test_classification_model_forward_and_backward(self):
        """A multi-layer MLP: flatten → linear → relu → linear. Exercises real
        depth (two weight matrices + an activation) and verifies gradients reach
        every parameter after a backward pass."""
        graph = _make_graph(
            nodes=[
                _n("data", "data.mnist", {"batchSize": 32}),
                _n("flat", "ml.layers.flatten", {}),
                _n("fc1", "ml.layers.linear", {"outFeatures": 32}),
                _n("relu", "ml.activations.relu", {}),
                _n("fc2", "ml.layers.linear", {"outFeatures": 10}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 1}),
            ],
            edges=[
                _e("e1", "data", "out", "flat", "in"),
                _e("e2", "flat", "out", "fc1", "in"),
                _e("e3", "fc1", "out", "relu", "in"),
                _e("e4", "relu", "out", "fc2", "in"),
                _e("e5", "fc2", "out", "loss", "predictions"),
                _e("e6", "data", "labels", "loss", "labels"),
                _e("e7", "loss", "out", "opt", "loss"),
            ],
        )
        model = _exec_model_classes(export_to_python(graph))["Model"]()
        out = model(torch.randn(4, 1, 28, 28))
        assert out.shape == (4, 10)
        out.sum().backward()  # gradients flow through the whole generated graph
        params = list(model.parameters())
        assert len(params) == 4  # 2 linears × (weight + bias)
        assert all(p.grad is not None for p in params)

    def test_custom_image_classification_model_runs(self):
        """The §4 custom-dataset export path produces a runnable model too."""
        graph = _make_graph(
            nodes=[
                _custom_node(inputKind="image", task="classification", hfId="mnist",
                             inputColumn="image", channels=1, imageSize=28, batchSize=32),
                _n("flat", "ml.layers.flatten", {}),
                _n("fc", "ml.layers.linear", {"outFeatures": 10}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 1}),
            ],
            edges=[
                _e("e1", "data", "out", "flat", "in"),
                _e("e2", "flat", "out", "fc", "in"),
                _e("e3", "fc", "out", "loss", "predictions"),
                _e("e4", "data", "labels", "loss", "labels"),
                _e("e5", "loss", "out", "opt", "loss"),
            ],
        )
        model = _exec_model_classes(export_to_python(graph))["Model"]()
        assert model(torch.randn(4, 1, 28, 28)).shape == (4, 10)

    def test_text_classification_model_runs_on_token_ids(self):
        """Embedding → sequence pool → linear: forward on integer token ids."""
        graph = _make_graph(
            nodes=[
                _n("data", "data.imdb", {"batchSize": 32, "maxLen": 64, "vocabSize": 1000}),
                _n("embed", "ml.layers.embedding", {"numEmbeddings": 1000, "embeddingDim": 32}),
                _n("pool", "ml.structural.sequence_pool", {"mode": "mean"}),
                _n("fc", "ml.layers.linear", {"outFeatures": 2}),
                _n("loss", "ml.loss.cross_entropy", {}),
                _n("opt", "ml.optimizers.adam", {"lr": 0.001, "epochs": 1}),
            ],
            edges=[
                _e("e1", "data", "out", "embed", "in"),
                _e("e2", "embed", "out", "pool", "in"),
                _e("e3", "pool", "out", "fc", "in"),
                _e("e4", "fc", "out", "loss", "predictions"),
                _e("e5", "data", "labels", "loss", "labels"),
                _e("e6", "loss", "out", "opt", "loss"),
            ],
        )
        model = _exec_model_classes(export_to_python(graph))["Model"]()
        out = model(torch.randint(0, 1000, (4, 64)))
        assert out.shape == (4, 2)

    @pytest.fixture(params=sorted(glob.glob("model-presets/*.json")))
    def preset(self, request):
        with open(request.param) as f:
            return json.load(f), os.path.basename(request.param)

    def test_every_preset_model_constructs(self, preset):
        """Every shipped preset must generate a model class that actually
        instantiates in PyTorch — catches bad layer dims / broken wiring that a
        compile check sails right past. (Forward needs per-preset input shapes;
        construction alone already exercises every layer's __init__.)"""
        graph, name = preset
        ns = _exec_model_classes(export_to_python(graph))
        cls = next((ns[c] for c in ("Model", "Generator", "Encoder") if c in ns), None)
        assert cls is not None, f"Preset {name} generated no model/generator/encoder class"
        cls()  # raises if any layer dim is invalid
