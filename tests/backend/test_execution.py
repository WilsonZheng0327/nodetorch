"""Integration tests for the graph-execution walks.

These drive the per-node-type dispatch end-to-end on real shipped presets:
  build_modules / inspect_graph (forward)         — engine/graph_builder/forward.py
  infer_graph, evaluate_test_set                 — engine/graph_builder/inference.py
  run_final_forward (via train_graph)            — training/base.py

Those walks currently have ZERO coverage, yet they're the duplicated
forward-dispatch we want to unify. This suite is the safety net for that
refactor: it pins current behavior so a behavior-preserving change stays green.

Only MNIST / FashionMNIST presets are used (small, already-cached datasets), but
they cover nearly every irregular dispatch branch: conv, linear, dropout,
reparameterize + VAE loss, GAN noise_input + subgraph.block, diffusion
noise_scheduler + concat.
"""
import glob
import json
import math
import os

import pytest
import torch.nn as nn

from engine.graph_builder import (
    build_modules,
    inspect_graph,
    infer_graph,
    evaluate_test_set,
    train_graph,
    _model_store,
)

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model-presets")


def load_preset(name: str) -> dict:
    with open(os.path.join(PRESETS_DIR, f"{name}.json")) as f:
        return json.load(f)


def node_ids(graph_data: dict) -> set[str]:
    return {n["id"] for n in graph_data["graph"]["nodes"]}


# MNIST/FashionMNIST presets — fast (cached) data, covering the irregular branches.
FAST_PRESETS = [
    "mlp-mnist",          # baseline: linear / flatten / dropout / relu
    "lenet5-mnist",       # conv2d / pooling
    "cnn-fashion-mnist",  # conv on a second dataset
    "autoencoder-mnist",  # reconstruction (no classifier head)
    "vae-mnist",          # reparameterize + VAE loss (multi-input)
    "gan-mnist",          # GAN noise_input + subgraph.block
    "diffusion-mnist",    # diffusion noise_scheduler + concat
]


# --- Forward walk (no training needed) ---

@pytest.mark.parametrize("name", FAST_PRESETS)
def test_forward_walk_handles_every_node(name):
    """build_and_run_graph must produce a result for EVERY node, whatever its
    type. This is the core regression check: if the dispatch ever drops or fails
    to handle a node type (the drift we're guarding against), a node goes missing
    here."""
    g = load_preset(name)
    results = inspect_graph(g)

    assert isinstance(results, dict) and results, f"{name}: no results"
    missing = node_ids(g) - set(results)
    assert not missing, f"{name}: nodes not handled by the forward walk: {sorted(missing)}"
    for nid, r in results.items():
        assert "outputs" in r and "metadata" in r, f"{name}: malformed result for {nid}"


# The exact metadata keys the forward walk emits per node type. In forward mode
# these are uniform across presets (no position-dependent fields), so we pin them
# by type — this is the structural snapshot that guards the describe/runner
# refactor: if a metadata field is dropped or added, the matching node fails here.
FORWARD_META_KEYS = {
    "data.mnist": ["outputShape"],
    "data.fashion_mnist": ["outputShape"],
    "ml.layers.flatten": ["activations", "outputShape", "paramCount"],
    "ml.layers.linear": ["activations", "outputShape", "paramCount", "weights"],
    "ml.layers.conv2d": ["activations", "outputShape", "paramCount", "weights"],
    "ml.layers.conv_transpose2d": ["activations", "outputShape", "paramCount", "weights"],
    "ml.layers.batchnorm2d": ["activations", "batchnorm", "outputShape", "paramCount", "weights"],
    "ml.activations.relu": ["activations", "outputShape", "paramCount"],
    "ml.activations.sigmoid": ["activations", "outputShape", "paramCount"],
    "ml.layers.avgpool2d": ["activations", "outputShape", "paramCount"],
    "ml.layers.maxpool2d": ["activations", "outputShape", "paramCount"],
    "ml.layers.dropout": ["activations", "outputShape", "paramCount"],
    "ml.structural.reshape": ["activations", "outputShape", "paramCount"],
    "ml.structural.concat": ["outputShape"],
    "ml.structural.add": ["outputShape"],
    "ml.structural.reparameterize": ["outputShape"],
    "ml.loss.cross_entropy": ["lossValue", "outputShape"],
    "ml.loss.mse": ["lossValue", "outputShape"],
    "ml.loss.gan": ["outputShape"],
    "ml.loss.vae": ["outputShape"],
    "ml.gan.noise_input": ["outputShape"],
    "subgraph.block": ["activations", "innerSnapshots", "outputShape", "paramCount"],
    "ml.diffusion.noise_scheduler": ["outputShape", "shapes"],
    "ml.optimizers.adam": [],
    "ml.optimizers.sgd": [],
    "ml.optimizers.adamw": [],
}


@pytest.mark.parametrize("name", FAST_PRESETS)
def test_forward_metadata_structure_is_stable(name):
    """Pin the exact per-node-type metadata keys, so the describe/runner refactor
    stays behavior-preserving (the #8 'every node handled' check doesn't see fields)."""
    g = load_preset(name)
    results = inspect_graph(g)
    types = {n["id"]: n["type"] for n in g["graph"]["nodes"]}
    for nid, r in results.items():
        t = types[nid]
        if t in FORWARD_META_KEYS:
            assert sorted(r["metadata"].keys()) == FORWARD_META_KEYS[t], f"{name}/{t}"


def test_mlp_forward_produces_correct_shapes():
    """Pin exact forward behavior on the baseline classifier."""
    g = load_preset("mlp-mnist")
    results = inspect_graph(g)

    errors = {nid: r["metadata"]["error"] for nid, r in results.items() if r["metadata"].get("error")}
    assert not errors, f"unexpected forward errors: {errors}"

    # Data node emits an output shape; the final linear emits 10 logits.
    assert results["mnist"]["metadata"].get("outputShape")
    assert results["linear3"]["metadata"]["outputShape"][-1] == 10


def test_build_modules_creates_trainable_layers():
    g = load_preset("mlp-mnist")
    modules = build_modules(g)
    assert isinstance(modules.get("linear1"), nn.Module)
    assert isinstance(modules.get("linear3"), nn.Module)


def test_gan_preset_runs_its_subgraph_block():
    """gan-mnist contains a subgraph.block — exercise the recursive branch."""
    g = load_preset("gan-mnist")
    block_ids = [n["id"] for n in g["graph"]["nodes"] if n["type"] == "subgraph.block"]
    assert block_ids, "expected gan-mnist to contain a subgraph.block"
    results = inspect_graph(g)
    for bid in block_ids:
        assert bid in results, f"subgraph block {bid} not handled"


# --- Train → infer → test (one shared 1-epoch run) ---

@pytest.fixture(scope="module")
def trained_mlp():
    """Train mlp-mnist for a single fast epoch and leave it in the model store, so
    the infer/test walks have a trained model to read."""
    _model_store.clear()
    g = load_preset("mlp-mnist")
    for n in g["graph"]["nodes"]:
        if n["type"].startswith("ml.optimizers"):
            n["properties"]["epochs"] = 1      # one epoch
        if n["type"].startswith("data."):
            n["properties"]["batchSize"] = 512  # fewer iterations → faster
    epochs: list[dict] = []
    result = train_graph(g, on_epoch=epochs.append)
    return g, result, epochs


def test_train_graph_one_epoch_smoke(trained_mlp):
    g, result, epochs = trained_mlp
    assert "error" not in result, result.get("error")
    assert len(epochs) >= 1, "no epoch was reported"
    assert math.isfinite(epochs[-1]["loss"]), "final loss is not finite"
    assert "current" in _model_store, "training did not populate the model store"


def test_infer_graph_predicts_after_training(trained_mlp):
    g, _, _ = trained_mlp
    out = infer_graph(g)
    assert "error" not in out, out.get("error")
    assert out["prediction"] is not None
    assert "nodeResults" in out

    # describe_inference output is now produced by the shared dispatch — pin its
    # shape so a metadata regression in the inference presenter is caught.
    pred = out["prediction"]
    assert set(pred) == {"predictedClass", "confidence", "probabilities"}
    assert 0 <= pred["predictedClass"] < 10 and len(pred["probabilities"]) == 10

    data_meta = out["nodeResults"]["mnist"]["metadata"]
    assert "outputShape" in data_meta and "actualLabel" in data_meta
    assert "imagePixels" in data_meta  # MNIST is a 4D image → single-sample preview

    # The classifier head carries the same prediction in its per-node metadata.
    assert out["nodeResults"]["linear3"]["metadata"]["prediction"] == pred


def test_evaluate_test_set_after_training(trained_mlp):
    g, _, _ = trained_mlp
    out = evaluate_test_set(g)
    assert "error" not in out, out.get("error")
    assert 0.0 <= out["testAccuracy"] <= 1.0
    assert out["testSamples"] > 0


def test_tracked_samples_probed_during_training(trained_mlp):
    """probe_tracked_samples runs the fixed tracked set through the live model
    each epoch — pin its per-sample output (the dispatch we're unifying)."""
    _, _, epochs = trained_mlp
    probes = epochs[-1]["trackedSamples"]
    assert probes, "no tracked samples were probed"
    p = probes[0]
    assert {"idx", "label", "probabilities", "predictedClass", "confidence", "loss"} <= set(p)
    assert len(p["probabilities"]) == 10
    assert 0 <= p["predictedClass"] < 10


# --- GAN / diffusion final-forward walk (run_final_forward) ---
#
# These paradigms exercise run_final_forward branches the standard path never
# hits: GAN noise_input + subgraph blocks, diffusion noise_scheduler + concat/add.
# The walk produces *lite* metadata — deliberately lighter than the inspection
# walk's FORWARD_META_KEYS above: no weights, no batchnorm, no subgraph inner
# snapshots. This map pins that, so migrating run_final_forward onto the shared
# execute() dispatch stays behavior-preserving.
LITE_META_KEYS = {
    "data.mnist": ["outputShape"],
    "ml.gan.noise_input": ["outputShape"],
    "subgraph.block": ["outputShape"],
    "ml.loss.gan": ["outputShape"],
    "ml.diffusion.noise_scheduler": ["outputShape"],
    "ml.structural.concat": ["outputShape"],
    "ml.structural.add": ["outputShape"],
    "ml.layers.conv2d": ["activations", "outputShape", "paramCount"],
    "ml.layers.batchnorm2d": ["activations", "outputShape", "paramCount"],
    "ml.activations.relu": ["activations", "outputShape", "paramCount"],
    "ml.loss.mse": ["lossValue", "outputShape"],
    # optimizers don't run; save_training_results overwrites them with final stats.
    "ml.optimizers.adam": ["finalAccuracy", "finalLoss"],
}


@pytest.fixture(scope="module")
def _mnist_subset():
    """Shrink the MNIST training set to a single batch so the heavier GAN /
    diffusion loops train in seconds. Restores the real loader afterward."""
    import torch.utils.data as tud
    import dataprep.data_loaders as dl

    original = dl.TRAIN_DATASETS["data.mnist"]
    full = original()
    dl.TRAIN_DATASETS["data.mnist"] = lambda: tud.Subset(full, range(512))
    yield
    dl.TRAIN_DATASETS["data.mnist"] = original


def _train_one_epoch(name: str):
    """Train a preset for a single batched epoch, leaving its result in hand."""
    g = load_preset(name)
    for n in g["graph"]["nodes"]:
        if n["type"].startswith("ml.optimizers"):
            n["properties"]["epochs"] = 1
        if n["type"].startswith("data."):
            n["properties"]["batchSize"] = 512
    _model_store.clear()
    return g, train_graph(g)


def _assert_lite_walk(name: str, g: dict, result: dict):
    assert "error" not in result, result.get("error")
    nr = result["nodeResults"]
    types = {n["id"]: n["type"] for n in g["graph"]["nodes"]}
    # run_final_forward must emit a result for EVERY node — the regression check
    # for a dropped/unhandled node type, same spirit as the forward walk above.
    assert set(nr) == set(types), f"{name}: missing {sorted(set(types) - set(nr))}"
    for nid, r in nr.items():
        t = types[nid]
        if t in LITE_META_KEYS:
            assert sorted(r["metadata"].keys()) == LITE_META_KEYS[t], f"{name}/{t}"


@pytest.fixture(scope="module")
def trained_gan(_mnist_subset):
    return _train_one_epoch("gan-mnist")


@pytest.fixture(scope="module")
def trained_diffusion(_mnist_subset):
    return _train_one_epoch("diffusion-mnist")


def test_gan_train_one_epoch_final_forward(trained_gan):
    """GAN final forward: noise_input + two subgraph blocks + gan loss."""
    g, result = trained_gan
    _assert_lite_walk("gan-mnist", g, result)


def test_diffusion_train_one_epoch_final_forward(trained_diffusion):
    """Diffusion final forward: noise_scheduler + concat/add + conv stack."""
    g, result = trained_diffusion
    _assert_lite_walk("diffusion-mnist", g, result)


# --- Coverage note ---

def test_fast_presets_all_exist():
    """Guard against a preset being renamed out from under the suite."""
    available = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(PRESETS_DIR, "*.json"))}
    assert set(FAST_PRESETS) <= available, f"missing presets: {set(FAST_PRESETS) - available}"
