"""Tests for the spec-driven custom dataset interpreter (dataprep.custom_dataset).

Hermetic: a fake in-memory HuggingFace dataset is injected into the module cache,
so these run without any network/download.
"""

import torch
import pytest
from PIL import Image

import dataprep.custom_dataset as C
from dataprep.custom_dataset import (
    DatasetSpec, CustomDataset, infer_class_names, build_detail,
    get_raw_text, get_lm_vocab, _parse_norm,
)


class FakeClassLabel:
    def __init__(self, names): self.names = names


class FakeHF:
    """Minimal stand-in for a datasets.Dataset: row dicts, column access, len,
    iteration, and a .features map (for ClassLabel name inference)."""

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


def _inject(spec, hf):
    C._HF_CACHE[(spec.hf_id, spec.hf_config, spec.train_split)] = hf


# --- DatasetSpec.from_props ---

def test_from_props_defaults():
    s = DatasetSpec.from_props({}, "n1")
    assert s.input_kind == "image" and s.task == "classification"
    assert s.label_column == "label" and s.cache_key == "custom_n1_"


def test_parse_norm():
    assert _parse_norm("") is None and _parse_norm(None) is None
    assert _parse_norm("0.5, 0.5") == (0.5, 0.5)
    assert _parse_norm([0.1, 0.2]) == (0.1, 0.2)


def test_unsupported_cell_raises():
    with pytest.raises(NotImplementedError):
        # image + language_model is not a supported cell
        CustomDataset(DatasetSpec.from_props({"inputKind": "image", "task": "language_model", "hfId": "x"}))


# --- image + classification / regression / autoencoder ---

def _img_hf(n=6, size=20):
    rows = [{"image": Image.new("L", (size, size), 40 * i), "label": i % 2} for i in range(n)]
    return FakeHF(rows, features={"label": FakeClassLabel(["cat", "dog"])})


def test_image_classification():
    spec = DatasetSpec.from_props({"inputKind": "image", "task": "classification",
                                   "hfId": "img", "inputColumn": "image", "channels": 1, "imageSize": 8})
    _inject(spec, _img_hf())
    ds = CustomDataset(spec)
    x, y = ds[0]
    assert tuple(x.shape) == (1, 8, 8) and x.dtype == torch.float32
    assert y.item() == 0 and y.dtype == torch.long


def test_image_channels_and_normalize():
    spec = DatasetSpec.from_props({"inputKind": "image", "task": "classification", "hfId": "img2",
                                   "inputColumn": "image", "channels": 3, "imageSize": 16,
                                   "normalizeMean": "0.5,0.5,0.5", "normalizeStd": "0.5,0.5,0.5"})
    _inject(spec, _img_hf())
    x, _ = CustomDataset(spec)[0]
    assert tuple(x.shape) == (3, 16, 16)


def test_image_autoencoder_target_is_input():
    spec = DatasetSpec.from_props({"inputKind": "image", "task": "autoencoder", "hfId": "img3",
                                   "inputColumn": "image", "channels": 1, "imageSize": 8})
    _inject(spec, _img_hf())
    x, y = CustomDataset(spec)[0]
    assert torch.equal(x, y)


# --- text + classification (word / char / bpe) ---

def _text_hf():
    return FakeHF(
        [{"text": "the cat sat on the mat", "label": 0},
         {"text": "buy stocks now market up", "label": 1}],
        features={"label": FakeClassLabel(["animals", "finance"])},
    )


@pytest.mark.parametrize("tok", ["word", "char", "bpe"])
def test_text_classification(tok):
    spec = DatasetSpec.from_props({"inputKind": "text", "task": "classification", "hfId": f"txt_{tok}",
                                   "tokenizer": tok, "maxLen": 12, "vocabSize": 100})
    _inject(spec, _text_hf())
    x, y = CustomDataset(spec)[0]
    assert tuple(x.shape) == (12,) and x.dtype == torch.long
    assert y.item() == 0 and y.dtype == torch.long


# --- text + language_model (corpus chunking) ---

@pytest.mark.parametrize("tok", ["char", "word", "bpe"])
def test_language_model_windows(tok):
    spec = DatasetSpec.from_props({"inputKind": "text", "task": "language_model", "hfId": f"lm_{tok}",
                                   "tokenizer": tok, "seqLen": 4, "vocabSize": 100})
    _inject(spec, FakeHF([{"text": "the cat sat on the mat"}, {"text": "the dog ran fast"}]))
    ds = CustomDataset(spec)
    assert len(ds) == max(0, (len(ds._lm_data) - 1) // 4)
    x, y = ds[0]
    assert tuple(x.shape) == (4,) and tuple(y.shape) == (4,)
    # target is the input shifted one token ahead
    assert ds._lm_data[1:5].tolist() == y.tolist()


# --- vector + classification / regression ---

def test_vector_classification_and_regression():
    spec = DatasetSpec.from_props({"inputKind": "vector", "task": "classification",
                                   "inputColumn": "feat", "labelColumn": "y"})
    d = object.__new__(CustomDataset)
    d.spec = spec
    x = d._preprocess_input({"feat": [1.0, 2.0, 3.0]})
    assert x.dtype == torch.float32 and x.tolist() == [1.0, 2.0, 3.0]
    assert d._make_target({"y": 1}, x).dtype == torch.long
    spec_r = DatasetSpec.from_props({"inputKind": "vector", "task": "regression", "labelColumn": "y"})
    d.spec = spec_r
    assert d._make_target({"y": 2.5}, x).dtype == torch.float32


# --- metadata helpers ---

def test_infer_class_names():
    # override (comma string and list)
    assert infer_class_names(DatasetSpec.from_props({"task": "classification", "classNames": "a, b, c"})) == ["a", "b", "c"]
    assert infer_class_names(DatasetSpec.from_props({"task": "classification", "classNames": ["x", "y"]})) == ["x", "y"]
    # HF ClassLabel names
    spec = DatasetSpec.from_props({"task": "classification", "hfId": "cl", "labelColumn": "label"})
    _inject(spec, _text_hf())
    assert infer_class_names(spec) == ["animals", "finance"]
    # non-classification
    assert infer_class_names(DatasetSpec.from_props({"task": "language_model", "hfId": "cl"})) == []


def test_build_detail_text_and_lm():
    spec = DatasetSpec.from_props({"inputKind": "text", "task": "classification", "hfId": "d_txt", "labelColumn": "label"})
    _inject(spec, _text_hf())
    d = build_detail(spec)
    assert d["isText"] and d["labels"] == ["animals", "finance"] and "sampleTexts" in d

    lm = DatasetSpec.from_props({"inputKind": "text", "task": "language_model", "hfId": "d_lm"})
    _inject(lm, FakeHF([{"text": "hello world " * 50}]))
    dl = build_detail(lm)
    assert dl["isLanguageModel"] and dl["labels"] == [] and "vocabSize" in dl


def test_build_detail_image():
    spec = DatasetSpec.from_props({"inputKind": "image", "task": "classification", "hfId": "d_img",
                                   "inputColumn": "image", "channels": 1, "imageSize": 8, "classNames": "cat,dog"})
    _inject(spec, _img_hf())
    d = build_detail(spec)
    assert d["channels"] == 1 and d["imageSize"] == [8, 8]
    assert set(d["sampleImages"].keys()) == {"cat", "dog"}


def test_get_lm_vocab_roundtrip():
    for tok in ("char", "word", "bpe"):
        spec = DatasetSpec.from_props({"inputKind": "text", "task": "language_model",
                                       "hfId": f"v_{tok}", "tokenizer": tok, "vocabSize": 100})
        _inject(spec, FakeHF([{"text": "the cat sat"}, {"text": "the dog ran"}]))
        assert get_raw_text(spec) == "the cat sat\nthe dog ran"
        t2i, i2t = get_lm_vocab(spec)
        assert len(t2i) > 0 and all(i2t[i] == t for t, i in list(t2i.items())[:5])
