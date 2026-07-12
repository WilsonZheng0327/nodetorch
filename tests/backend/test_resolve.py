"""Tests for the node-aware dataset resolvers (dataprep.resolve).

Two guarantees: built-in datasets resolve to the exact same registry values as
before (invariance), and custom nodes resolve from their DatasetSpec. Hermetic —
a fake HF dataset is injected into the cache for the custom cases.
"""

import torch

import dataprep.custom_dataset as C
import dataprep.data_loaders as dl
from dataprep import resolve as R


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


def _custom(props, node_id="n"):
    node = {"type": "data.custom", "id": node_id, "properties": props}
    spec = R._spec(node)
    C._HF_CACHE[(spec.hf_id, spec.hf_config, spec.train_split)] = FakeHF(
        [{"text": "the cat sat", "label": 0}, {"text": "buy the stock", "label": 1}],
        {"label": FakeClassLabel(["animals", "finance"])},
    )
    return node


# --- built-in invariance: resolvers return the same registry values ---

def test_builtin_is_lm_matches_registry():
    assert R.resolve_is_lm({"type": "data.tiny_shakespeare"}) is True
    assert R.resolve_is_lm({"type": "data.mnist"}) is False


def test_builtin_denormalizer_matches_registry():
    assert R.resolve_denormalizer({"type": "data.mnist"}) is dl.DENORMALIZERS.get("data.mnist")
    assert R.resolve_denormalizer({"type": "data.ag_news"}) is None  # text: no denorm


def test_builtin_class_names_matches_registry():
    assert R.resolve_class_names({"type": "data.mnist"}) == dl.CLASS_NAMES.get("data.mnist", [])


# --- custom branches ---

def test_custom_is_lm_from_task():
    assert R.resolve_is_lm(_custom({"task": "language_model", "inputKind": "text"})) is True
    assert R.resolve_is_lm(_custom({"task": "classification", "inputKind": "text"})) is False


def test_custom_class_names_from_hf():
    node = _custom({"inputKind": "text", "task": "classification", "labelColumn": "label"})
    assert R.resolve_class_names(node) == ["animals", "finance"]


def test_custom_train_dataset_builds():
    node = _custom({"inputKind": "text", "task": "classification", "maxLen": 6, "tokenizer": "word"})
    ds = R.resolve_train_dataset(node)
    assert ds is not None and len(ds) == 2
    x, y = ds[0]
    assert tuple(x.shape) == (6,) and y.dtype == torch.long


def test_custom_test_dataset_none_without_split():
    node = _custom({"inputKind": "text", "task": "classification", "testSplit": ""})
    assert R.resolve_test_dataset(node) is None


def test_custom_denormalizer_image():
    # with normalize → reverses; without → identity; non-image → None
    dn = R.resolve_denormalizer({"type": "data.custom", "id": "d", "properties":
                                 {"inputKind": "image", "normalizeMean": "0.5", "normalizeStd": "0.5"}})
    assert abs(dn(torch.zeros(1, 2, 2)).flatten()[0].item() - 0.5) < 1e-6
    ident = R.resolve_denormalizer({"type": "data.custom", "id": "d", "properties": {"inputKind": "image"}})
    t = torch.zeros(1, 2, 2)
    assert ident(t) is t
    assert R.resolve_denormalizer({"type": "data.custom", "id": "d", "properties": {"inputKind": "text"}}) is None


def test_custom_detail():
    node = _custom({"inputKind": "text", "task": "classification", "labelColumn": "label"})
    d = R.resolve_detail(node)
    assert d["isText"] and d["labels"] == ["animals", "finance"]
