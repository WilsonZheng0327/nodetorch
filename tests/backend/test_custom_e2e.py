"""End-to-end tests for the custom dataset node: a real (tiny) training run and
the detail endpoint, proving the whole Phase-2/3 wiring holds together.

Hermetic — a fake HuggingFace dataset is injected into the cache, so a genuine
build_training_context → run_training pass executes without any download.
"""

import dataprep.custom_dataset as C
from dataprep.custom_dataset import DatasetSpec
from training import run_training, detect_training_mode
from api.data import _detail_response


class FakeClassLabel:
    def __init__(self, names): self.names = names


class FakeHF:
    def __init__(self, rows, features=None):
        self.rows = rows
        self.features = features or {}

    def __len__(self): return len(self.rows)
    def __iter__(self): return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        if isinstance(key, slice):
            return {k: [r[k] for r in self.rows[key]] for k in self.rows[0]}
        return self.rows[key]


def _inject_text_hf(hf_id, n=40):
    """A small 2-class text dataset (alternating simple sentences)."""
    animals = "the cat sat on the warm mat and slept"
    finance = "buy the stock market shares rise today fast"
    rows = [{"text": animals if i % 2 == 0 else finance, "label": i % 2} for i in range(n)]
    spec = DatasetSpec.from_props({"hfId": hf_id})
    C._HF_CACHE[(hf_id, None, "train")] = FakeHF(rows, {"label": FakeClassLabel(["animals", "finance"])})


def _text_cls_graph(hf_id, vocab=50):
    return {"version": "1.0", "graph": {
        "nodes": [
            {"id": "data", "type": "data.custom", "properties": {
                "inputKind": "text", "task": "classification", "hfId": hf_id,
                "inputColumn": "text", "labelColumn": "label", "tokenizer": "word",
                "vocabSize": vocab, "maxLen": 12, "batchSize": 8}},
            {"id": "embed", "type": "ml.layers.embedding", "properties": {"numEmbeddings": vocab, "embeddingDim": 8}},
            {"id": "pool", "type": "ml.structural.sequence_pool", "properties": {"mode": "mean"}},
            {"id": "linear", "type": "ml.layers.linear", "properties": {"outFeatures": 2}},
            {"id": "loss", "type": "ml.loss.cross_entropy", "properties": {}},
            {"id": "adam", "type": "ml.optimizers.adam", "properties": {"lr": 0.01, "epochs": 1, "valSplit": 0.0}},
        ],
        "edges": [
            {"source": {"nodeId": "data", "portId": "out"}, "target": {"nodeId": "embed", "portId": "in"}},
            {"source": {"nodeId": "embed", "portId": "out"}, "target": {"nodeId": "pool", "portId": "in"}},
            {"source": {"nodeId": "pool", "portId": "out"}, "target": {"nodeId": "linear", "portId": "in"}},
            {"source": {"nodeId": "linear", "portId": "out"}, "target": {"nodeId": "loss", "portId": "predictions"}},
            {"source": {"nodeId": "data", "portId": "labels"}, "target": {"nodeId": "loss", "portId": "labels"}},
            {"source": {"nodeId": "loss", "portId": "out"}, "target": {"nodeId": "adam", "portId": "loss"}},
        ],
    }}


def test_custom_text_classification_trains_end_to_end():
    _inject_text_hf("e2e_txt")
    captured = {}
    result = run_training(_text_cls_graph("e2e_txt"), on_epoch=lambda m: captured.update(msg=m))
    assert not (isinstance(result, dict) and result.get("error")), result
    m = captured["msg"]
    # The whole wired pipeline ran: class names resolved, tracked samples probed.
    assert m["classNames"] == ["animals", "finance"]
    assert len(m["trackedSamples"]) == 12
    assert m["loss"] is not None and m["accuracy"] is not None


def test_detect_training_mode_custom():
    lm = {"d": {"type": "data.custom", "id": "d", "properties": {"task": "language_model", "inputKind": "text"}}}
    cls = {"d": {"type": "data.custom", "id": "d", "properties": {"task": "classification", "inputKind": "image"}}}
    assert detect_training_mode(lm) == "autoregressive"
    assert detect_training_mode(cls) == "standard"


def test_dataset_detail_endpoint_custom():
    _inject_text_hf("e2e_detail")
    node = {"type": "data.custom", "id": "d", "properties": {
        "inputKind": "text", "task": "classification", "hfId": "e2e_detail", "labelColumn": "label"}}
    resp = _detail_response(node)
    assert resp["status"] == "ok"
    assert resp["detail"]["labels"] == ["animals", "finance"] and resp["detail"]["isText"]
