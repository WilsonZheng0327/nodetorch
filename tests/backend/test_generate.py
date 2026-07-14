"""Tests for the generation package (backend/generate/).

Covers the two on-demand generation entry points, which had ZERO coverage:
  generate_text        — generate/text_generate.py (autoregressive char/BPE decode)
  generate_gan_images  — generate/gan_generate.py  (noise → generator → images)

Both require a trained model in the shared _model_store; without one they must
degrade to an {"error": ...} dict rather than crash. The e2e tests train a
single fast epoch on a shrunk corpus/dataset (same trick as test_execution) so
the whole file stays well under a few seconds.
"""
import json
import os

import pytest
import torch.utils.data as tud

from engine.graph_builder import _model_store, train_graph
from generate.text_generate import generate_text, _get_tokenizer, _get_dataset_type
from generate.gan_generate import generate_gan_images

PRESETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model-presets")


def load_preset(name: str) -> dict:
    with open(os.path.join(PRESETS_DIR, f"{name}.json")) as f:
        return json.load(f)


def _nodes(graph_data: dict) -> dict:
    """Return the {id: node} dict the generate helpers operate on."""
    return {n["id"]: n for n in graph_data["graph"]["nodes"]}


def _train_one_epoch(name: str):
    """Train a preset for a single batched epoch, leaving it in _model_store.

    Mirrors test_execution._train_one_epoch: caps epochs=1 and shrinks the
    batch so a single epoch is a handful of steps. Returns (graph, result).
    """
    g = load_preset(name)
    for n in g["graph"]["nodes"]:
        if n["type"].startswith("ml.optimizers"):
            n["properties"]["epochs"] = 1
        if n["type"].startswith("data."):
            n["properties"]["batchSize"] = 128
    _model_store.clear()
    result = train_graph(g, on_epoch=lambda e: None)
    return g, result


# --- Corpus/dataset shrinking fixtures (copied from test_execution) ---


@pytest.fixture
def _shakespeare_subset():
    """Shrink TinyShakespeare to a single small batch of short sequences so the
    autoregressive loop trains in seconds. Restores the real loader afterward.

    The vocab is built from the FULL corpus, so subsetting *sequences* keeps
    every character id in range — safe for both training and generation decode.
    """
    import dataprep.data_loaders as dl

    original = dl.TRAIN_DATASETS["data.tiny_shakespeare"]
    sub = tud.Subset(dl.train_dataset_tiny_shakespeare(seqLen=64), range(128))
    dl.TRAIN_DATASETS["data.tiny_shakespeare"] = lambda: sub
    yield
    dl.TRAIN_DATASETS["data.tiny_shakespeare"] = original


@pytest.fixture
def _mnist_subset():
    """Shrink MNIST to a single batch so the GAN loop trains in seconds."""
    import dataprep.data_loaders as dl

    original = dl.TRAIN_DATASETS["data.mnist"]
    full = original()
    dl.TRAIN_DATASETS["data.mnist"] = lambda: tud.Subset(full, range(256))
    yield
    dl.TRAIN_DATASETS["data.mnist"] = original


# --- Pure helpers: tokenizer / dataset detection (fast, no training) ---


class TestGraphHelpers:
    """_get_tokenizer / _get_dataset_type parse the graph with no model needed."""

    def test_dataset_type_detected_on_charlm(self):
        """char-lm-shakespeare's data node is the tiny_shakespeare LM dataset."""
        nodes = _nodes(load_preset("char-lm-shakespeare"))
        assert _get_dataset_type(nodes) == "data.tiny_shakespeare"

    def test_no_tokenizer_node_defaults_to_none(self):
        """char-lm-shakespeare has no explicit tokenizer node — helper returns
        (None, None), which the decoder treats as character-level."""
        nodes = _nodes(load_preset("char-lm-shakespeare"))
        assert _get_tokenizer(nodes) == (None, None)

    def test_char_tokenizer_node_detected(self):
        """mini-gpt-shakespeare ships an explicit character tokenizer node."""
        nodes = _nodes(load_preset("mini-gpt-shakespeare"))
        tok_type, tok_props = _get_tokenizer(nodes)
        assert tok_type == "ml.preprocessing.tokenizer_char"
        assert isinstance(tok_props, dict)


# --- Error paths (fast, no training) ---


class TestErrorPaths:
    """With no trained model (or a malformed graph) both entry points must
    return an {"error": ...} dict rather than raising."""

    def test_generate_text_without_trained_model(self):
        _model_store.clear()
        out = generate_text(load_preset("char-lm-shakespeare"), prompt="a", max_tokens=5)
        assert "error" in out

    def test_generate_gan_without_trained_model(self):
        _model_store.clear()
        out = generate_gan_images(load_preset("gan-mnist"), num_samples=2)
        assert "error" in out

    def test_generate_text_without_lm_dataset_node(self, _mnist_subset):
        """A trained non-LM graph (no tiny_shakespeare node) has nothing to
        decode against — generate_text reports the missing dataset node."""
        _train_one_epoch("gan-mnist")  # populates the store with a real model
        # gan-mnist has no LM dataset node.
        out = generate_text(load_preset("gan-mnist"), prompt="x", max_tokens=3)
        assert "error" in out
        assert "dataset" in out["error"].lower()

    def test_generate_gan_without_noise_node(self, _mnist_subset):
        """A trained graph with no noise_input node is not a GAN — reported."""
        _train_one_epoch("gan-mnist")  # any trained model in the store
        g = load_preset("gan-mnist")
        # Strip the noise node AND any edges touching it (a dangling edge would
        # crash the topological sort before the noise check runs).
        noise_ids = {n["id"] for n in g["graph"]["nodes"]
                     if n["type"] == "ml.gan.noise_input"}
        g["graph"]["nodes"] = [n for n in g["graph"]["nodes"]
                               if n["id"] not in noise_ids]
        g["graph"]["edges"] = [
            e for e in g["graph"]["edges"]
            if e["source"]["nodeId"] not in noise_ids
            and e["target"]["nodeId"] not in noise_ids
        ]
        out = generate_gan_images(g, num_samples=2)
        assert "error" in out
        assert "noise" in out["error"].lower()


# --- Text generation end-to-end ---


class TestTextGenerate:
    """Train char-lm-shakespeare one epoch, then decode a short sample."""

    def test_generate_after_training(self, _shakespeare_subset):
        g, result = _train_one_epoch("char-lm-shakespeare")
        assert "error" not in result, result.get("error")

        out = generate_text(g, prompt="The ", max_tokens=15, temperature=0.8)
        assert "error" not in out, out.get("error")

        # generated text is a non-empty string, roughly bounded by max_tokens
        assert isinstance(out["generated"], str) and out["generated"]
        assert len(out["generated"]) <= 15  # char-level: 1 char per token

        # fullText echoes the user's prompt followed by the generation
        assert out["prompt"] == "The "
        assert out["fullText"].startswith("The ")
        assert out["fullText"] == out["prompt"] + out["generated"]

        # per-token records carry a decoded token + a valid probability
        assert isinstance(out["tokens"], list) and out["tokens"]
        assert len(out["tokens"]) == len(out["generated"])
        for t in out["tokens"]:
            assert 0.0 <= t["prob"] <= 1.0

    def test_greedy_low_temperature_is_stable(self, _shakespeare_subset):
        """top_k=1 with near-zero temperature is greedy decode: given the same
        trained weights and prompt it should reproduce the same text."""
        g, result = _train_one_epoch("char-lm-shakespeare")
        assert "error" not in result, result.get("error")

        kw = dict(prompt="A", max_tokens=12, temperature=1e-6, top_k=1)
        a = generate_text(g, **kw)
        b = generate_text(g, **kw)
        assert "error" not in a and "error" not in b
        assert a["generated"] == b["generated"]

    def test_empty_prompt_still_generates(self, _shakespeare_subset):
        """No prompt → an internal seed token is used, but it is NOT echoed back
        as user text (prompt stays empty)."""
        g, result = _train_one_epoch("char-lm-shakespeare")
        assert "error" not in result, result.get("error")

        out = generate_text(g, prompt="", max_tokens=8)
        assert "error" not in out, out.get("error")
        assert out["prompt"] == ""
        assert out["generated"]
        assert out["fullText"] == out["generated"]


# --- GAN generation end-to-end ---


class TestGanGenerate:
    """Train gan-mnist one epoch, then sample a few images from the generator."""

    def test_generate_images_after_training(self, _mnist_subset):
        g, result = _train_one_epoch("gan-mnist")
        assert "error" not in result, result.get("error")

        out = generate_gan_images(g, num_samples=3)
        assert "error" not in out, out.get("error")

        assert out["numSamples"] == 3
        assert len(out["images"]) == 3

        H, W, C = out["imageH"], out["imageW"], out["channels"]
        assert H > 0 and W > 0 and C >= 1

        # For a single-channel image each sample is a 2D [H][W] pixel grid.
        img = out["images"][0]
        assert len(img) == H
        assert len(img[0]) == W
        if C == 1:
            assert all(isinstance(px, (int, float)) for row in img for px in row)
        else:
            # multi-channel: innermost is a per-pixel RGB list of length C
            assert len(img[0][0]) == C
