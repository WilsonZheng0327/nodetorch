"""Tests for the self-contained model bundle (graph + weights) and the
crash-recovery snapshot in engine/graph_builder/_state.py.

build_modules() runs a real forward pass (it needs a data node to bootstrap
shapes), so we monkeypatch it to a stub that returns fresh modules — the bundle
logic under test is the serialize/embed-graph/match-weights/lazy-load behavior,
not module construction (covered by test_node_builders).
"""

import torch
import torch.nn as nn

import engine.graph_builder.build as build_mod
from engine.graph_builder import _state


GRAPH = {
    "version": "1.0",
    "graph": {
        "id": "main", "name": "Main",
        "nodes": [{"id": "lin", "type": "ml.layers.linear", "position": {"x": 0, "y": 0},
                   "properties": {"outFeatures": 2}}],
        "edges": [],
    },
}


def _seed_trained_model() -> nn.Linear:
    """Put a known trained module in the store and return it."""
    torch.manual_seed(0)
    trained = nn.Linear(4, 2)
    _state._model_store["current"] = {"lin": trained}
    return trained


def _stub_build_modules(monkeypatch):
    """Make build_modules() return a FRESH (differently-initialized) module, so a
    successful load is observable as the weights changing to the saved ones."""
    def fake(graph_data):
        torch.manual_seed(999)
        return {"lin": nn.Linear(4, 2)}
    monkeypatch.setattr(build_mod, "build_modules", fake)


def test_bundle_round_trip_restores_graph_and_weights(monkeypatch):
    trained = _seed_trained_model()
    saved_w = trained.weight.detach().clone()

    data = _state.save_bundle_bytes(GRAPH)
    assert data is not None

    _state._model_store.clear()          # simulate a fresh process
    _stub_build_modules(monkeypatch)

    result = _state.load_bundle_bytes(data)
    assert result["status"] == "ok"
    assert result["graph"] == GRAPH      # embedded graph comes back for the canvas
    # The fresh module now carries the SAVED weights, not its random init.
    assert torch.equal(_state._model_store["current"]["lin"].weight, saved_w)


def test_save_bundle_returns_none_without_a_trained_model():
    _state._model_store.clear()
    assert _state.save_bundle_bytes(GRAPH) is None


def test_load_bundle_rejects_garbage_and_non_bundles():
    assert "error" in _state.load_bundle_bytes(b"not a torch file")

    import io
    buf = io.BytesIO()
    torch.save({"just": "a dict", "no": "weights"}, buf)   # valid torch, wrong shape
    assert "error" in _state.load_bundle_bytes(buf.getvalue())


def test_snapshot_and_lazy_load_round_trip(monkeypatch, tmp_path):
    trained = _seed_trained_model()
    saved_w = trained.weight.detach().clone()
    snap = str(tmp_path / "snap.ntmodel")

    assert _state.snapshot_trained_model(GRAPH, snap)["status"] == "ok"

    _state._model_store.clear()
    _stub_build_modules(monkeypatch)
    # ensure_trained_model lazy-loads the snapshot when the store is empty.
    assert _state.ensure_trained_model(snap) is True
    assert torch.equal(_state._model_store["current"]["lin"].weight, saved_w)


def test_ensure_trained_model_false_when_no_snapshot(tmp_path):
    _state._model_store.clear()
    assert _state.ensure_trained_model(str(tmp_path / "nope.ntmodel")) is False


def test_ensure_trained_model_uses_in_memory_model_first(tmp_path):
    _seed_trained_model()
    # A model is already loaded → returns True without touching disk.
    assert _state.ensure_trained_model(str(tmp_path / "nope.ntmodel")) is True
