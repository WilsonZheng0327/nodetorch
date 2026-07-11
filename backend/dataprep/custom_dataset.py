# custom_dataset.py — Generic, spec-driven dataset for the `data.custom` node.
#
# A custom dataset is described declaratively by a DatasetSpec (parsed from the
# node's properties) along two orthogonal axes — input_kind (image | text |
# vector) × task (classification | language_model | regression | autoencoder) —
# plus its HuggingFace source, column mapping, and per-kind transform params.
# One interpreter (CustomDataset) reads the spec and returns the same
# (input, label) tensors the built-in datasets produce, so it flows through
# training / inference / preview unchanged. This module is pure interpreter — the
# node-aware resolution layer (resolve.py) wires it into the rest of the backend.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.utils.data


def _parse_norm(v) -> tuple[float, ...] | None:
    """Parse a normalization vector from props: a "0.5,0.5,0.5" string or a list.
    Empty / missing → None (meaning: no normalization)."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return tuple(float(x) for x in v.split(",") if x.strip())
    return tuple(float(x) for x in v)


@dataclass
class DatasetSpec:
    """Everything needed to build a custom dataset, derived from node properties.

    The two axes fully classify the dataset: ``input_kind`` (how each sample is
    encoded) and ``task`` (how the target is formed). The remaining fields are the
    HuggingFace source, the column mapping, and the per-kind transform params.
    """

    # --- axes ---
    input_kind: str          # "image" | "text" | "vector"
    task: str                # "classification" | "language_model" | "regression" | "autoencoder"

    # --- source (HuggingFace Hub) ---
    hf_id: str               # e.g. "mnist", "ag_news", "user/dataset"
    hf_config: str | None = None      # HF config / "name" argument
    train_split: str = "train"
    test_split: str | None = "test"
    trust_remote_code: bool = False

    # --- column mapping ---
    input_column: str = "text"        # image / text / feature column
    label_column: str | None = "label"

    # --- image params ---
    image_size: int = 32              # square resize target (H == W)
    channels: int = 3                 # 1 or 3
    normalize_mean: tuple[float, ...] | None = None   # None → no normalization
    normalize_std: tuple[float, ...] | None = None

    # --- text params ---
    tokenizer: str = "word"           # "char" | "word" | "bpe"
    vocab_size: int = 10000
    max_len: int = 128                # classification / regression sequence length
    seq_len: int = 128                # language_model context length

    # --- overrides / misc ---
    class_names_override: list[str] | None = None
    batch_size: int = 32
    cache_key: str = ""               # stable per-node key for vocab / BPE caches

    @classmethod
    def from_props(cls, props: dict, node_id: str = "") -> "DatasetSpec":
        """Build a spec from a data node's ``properties`` dict.

        ``node_id`` seeds ``cache_key`` so per-node tokenizer vocab / BPE merges
        are cached and stable across calls (mirrors ``_build_vocab``'s cache key).
        """
        return cls(
            input_kind=props.get("inputKind", "image"),
            task=props.get("task", "classification"),
            hf_id=str(props.get("hfId", "")).strip(),
            hf_config=(props.get("hfConfig") or None),
            train_split=props.get("trainSplit", "train"),
            test_split=(props.get("testSplit") or None),
            trust_remote_code=bool(props.get("trustRemoteCode", False)),
            input_column=props.get("inputColumn", "text"),
            label_column=(props.get("labelColumn") or "label"),
            image_size=int(props.get("imageSize", 32)),
            channels=int(props.get("channels", 3)),
            normalize_mean=_parse_norm(props.get("normalizeMean")),
            normalize_std=_parse_norm(props.get("normalizeStd")),
            tokenizer=props.get("tokenizer", "word"),
            vocab_size=int(props.get("vocabSize", 10000)),
            max_len=int(props.get("maxLen", 128)),
            seq_len=int(props.get("seqLen", 128)),
            class_names_override=(props.get("classNames") or None),
            batch_size=int(props.get("batchSize", 32)),
            cache_key=f"custom_{node_id}_{props.get('hfId', '')}",
        )


# --- Which (input_kind, task) cells the interpreter supports in v1 ---
# The grid is 3 input kinds × 4 tasks; these are the cells that map cleanly onto
# the existing (input, label) contract. Unsupported cells raise a clear error at
# construction time rather than producing garbage tensors.
_SUPPORTED: set[tuple[str, str]] = {
    ("image", "classification"), ("image", "regression"), ("image", "autoencoder"),
    ("text", "classification"), ("text", "language_model"), ("text", "regression"),
    ("vector", "classification"), ("vector", "regression"), ("vector", "autoencoder"),
}


# Cache HuggingFace splits so repeated loads (batch preview, train, detail) hit
# memory, not disk/network. Keyed by (hf_id, hf_config, split). Mirrors the
# per-dataset caches in data_loaders.py (e.g. _imdb_cache).
_HF_CACHE: dict[tuple[str, str | None, str], object] = {}


def _load_hf(spec: DatasetSpec, split: str):
    """Load one split of the spec's HuggingFace dataset (cached)."""
    key = (spec.hf_id, spec.hf_config, split)
    if key not in _HF_CACHE:
        from datasets import load_dataset
        _HF_CACHE[key] = load_dataset(
            spec.hf_id, spec.hf_config, split=split,
            trust_remote_code=spec.trust_remote_code,
        )
    return _HF_CACHE[key]


def _to_pil(value):
    """A HF image cell → PIL.Image. Usually already a PIL image; also handle an
    undecoded {bytes|path} dict or a raw numpy array."""
    from PIL import Image
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        import io
        return Image.open(io.BytesIO(value["bytes"])) if value.get("bytes") else Image.open(value["path"])
    return Image.fromarray(value)


def _build_image_transform(spec: DatasetSpec):
    """Resize → ToTensor (→ Normalize) for image inputs. Channel conversion (to L
    or RGB) happens before this, in _preprocess_image, so shapes are consistent."""
    import torchvision.transforms as transforms
    steps: list = [
        transforms.Resize((spec.image_size, spec.image_size)),
        transforms.ToTensor(),
    ]
    if spec.normalize_mean is not None and spec.normalize_std is not None:
        steps.append(transforms.Normalize(spec.normalize_mean, spec.normalize_std))
    return transforms.Compose(steps)


class CustomDataset(torch.utils.data.Dataset):
    """Generic dataset that interprets a DatasetSpec into (input, label) tensors.

    Two sample-access shapes, chosen by task:
      - *row* path (classification / regression / autoencoder): one HF row per
        sample — build the input tensor from the input column, form the target
        from the label column.
      - *corpus* path (language_model): the whole tokenized corpus concatenated
        and chunked into fixed-length next-token windows.

    The per-modality encoding is filled in over subsequent steps; this skeleton
    establishes construction, the supported-cell guard, and the dispatch shape.
    """

    def __init__(self, spec: DatasetSpec, split: str | None = None):
        cell = (spec.input_kind, spec.task)
        if cell not in _SUPPORTED:
            raise NotImplementedError(
                f"Custom dataset does not support {spec.input_kind} + {spec.task}. "
                f"Supported: {sorted(_SUPPORTED)}"
            )
        self.spec = spec
        self.split = split or spec.train_split
        self.is_corpus = spec.task == "language_model"
        self._img_tf = _build_image_transform(spec) if spec.input_kind == "image" else None
        self.hf = _load_hf(spec, self.split)
        self._vocab = None       # word tokenizer: word → id
        self._bpe = None         # bpe tokenizer: BPETokenizer
        self._char_vocab = None  # char tokenizer: char → id
        self._lm_data = None     # language_model: flat token-id stream (whole corpus)
        self._lm_vocab = None    # language_model: token → id map (for decoding)
        if spec.input_kind == "text":
            if spec.task == "language_model":
                self._setup_lm()
            else:
                self._setup_text()

    def __len__(self) -> int:
        if self.is_corpus:
            return max(0, (len(self._lm_data) - 1) // self.spec.seq_len)
        return len(self.hf)

    def __getitem__(self, idx: int):
        if self.is_corpus:
            return self._getitem_corpus(idx)
        return self._getitem_row(idx)

    # --- row path (classification / regression / autoencoder) ---
    def _getitem_row(self, idx: int):
        row = self.hf[idx]
        x = self._preprocess_input(row)
        y = self._make_target(row, x)
        return x, y

    def _preprocess_input(self, row) -> torch.Tensor:
        kind = self.spec.input_kind
        if kind == "image":
            return self._preprocess_image(row[self.spec.input_column])
        if kind == "text":
            return self._preprocess_text(row[self.spec.input_column])
        if kind == "vector":
            return torch.tensor(row[self.spec.input_column], dtype=torch.float32)
        raise NotImplementedError(f"{kind} input preprocessing")

    def _corpus_texts(self, n: int = 5000) -> list:
        """A sample of the input column, used to fit the tokenizer once. The
        frequency ranking / merge set is stable over a sample, so we don't scan
        the whole (possibly huge) corpus."""
        m = min(n, len(self.hf))
        return self.hf[:m][self.spec.input_column]

    def _setup_text(self) -> None:
        """Fit the tokenizer the spec asks for, once, at construction time."""
        tk = self.spec.tokenizer
        if tk == "word":
            from dataprep.data_loaders import _build_vocab
            self._vocab = _build_vocab(self._corpus_texts(), self.spec.vocab_size, self.spec.cache_key)
        elif tk == "bpe":
            from dataprep.bpe import get_bpe_tokenizer
            corpus = "\n".join(self._corpus_texts())
            self._bpe = get_bpe_tokenizer(corpus, self.spec.vocab_size, self.spec.cache_key)
        elif tk == "char":
            self._char_vocab = self._build_char_vocab()
        else:
            raise NotImplementedError(f"text tokenizer '{tk}'")

    def _build_char_vocab(self) -> dict:
        """char→id vocab from the corpus sample (0=pad, 1=unk, then sorted chars)."""
        chars: set = set()
        for t in self._corpus_texts():
            chars.update(t)
        vocab = {"<pad>": 0, "<unk>": 1}
        for ch in sorted(chars):
            vocab[ch] = len(vocab)
        return vocab

    def _preprocess_text(self, text) -> torch.Tensor:
        """One text string → padded [max_len] token-id tensor, per tokenizer."""
        max_len = self.spec.max_len
        tk = self.spec.tokenizer
        if tk == "word":
            from dataprep.data_loaders import _encode_texts
            return _encode_texts([text], self._vocab, max_len)[0]
        if tk == "bpe":
            return torch.tensor(self._bpe.encode(text, max_len), dtype=torch.long)
        if tk == "char":
            ids = [self._char_vocab.get(c, 1) for c in text[:max_len]]
            ids += [0] * (max_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)
        raise NotImplementedError(f"text tokenizer '{tk}'")

    def _preprocess_image(self, value) -> torch.Tensor:
        """HF image cell → [C, H, W] tensor (resize/normalize per spec)."""
        img = _to_pil(value).convert("L" if self.spec.channels == 1 else "RGB")
        return self._img_tf(img)

    def _make_target(self, row, x: torch.Tensor):
        task = self.spec.task
        if task == "classification":
            return torch.tensor(int(row[self.spec.label_column]), dtype=torch.long)
        if task == "regression":
            return torch.tensor(row[self.spec.label_column], dtype=torch.float32)
        if task == "autoencoder":
            return x  # reconstruct the input: target IS the input
        raise NotImplementedError(f"{task} target")

    # --- corpus path (language_model) ---
    def _setup_lm(self) -> None:
        """Tokenize the ENTIRE corpus into one flat id stream (self._lm_data) for
        next-token training, plus a token→id map for decoding. Mirrors the
        built-in TinyShakespeare (char) / BPELMDataset (bpe) datasets."""
        corpus = "\n".join(self.hf[self.spec.input_column])
        tk = self.spec.tokenizer
        if tk == "char":
            chars = sorted(set(corpus))
            self._lm_vocab = {ch: i for i, ch in enumerate(chars)}
            ids = [self._lm_vocab[ch] for ch in corpus]
        elif tk == "bpe":
            from dataprep.bpe import get_bpe_tokenizer
            self._bpe = get_bpe_tokenizer(corpus, self.spec.vocab_size, self.spec.cache_key)
            self._lm_vocab = self._bpe.vocab
            ids = self._bpe.encode(corpus)  # whole corpus, no truncation
        elif tk == "word":
            from dataprep.data_loaders import _build_vocab, _tokenize
            self._vocab = _build_vocab([corpus], self.spec.vocab_size, self.spec.cache_key)
            self._lm_vocab = self._vocab
            unk = self._vocab.get("<unk>", 1)
            ids = [self._vocab.get(t, unk) for t in _tokenize(corpus)]
        else:
            raise NotImplementedError(f"language_model tokenizer '{tk}'")
        self._lm_data = torch.tensor(ids, dtype=torch.long)

    def _getitem_corpus(self, idx: int):
        """The idx-th consecutive window: (input, target) each [seq_len], target
        shifted one token ahead. Tail window is zero-padded."""
        seq_len = self.spec.seq_len
        start = idx * seq_len
        chunk = self._lm_data[start : start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            pad = torch.zeros(seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk[:-1], chunk[1:]


def infer_class_names(spec: DatasetSpec) -> list[str]:
    """The class labels for a classification dataset, indexed by class id.

    Priority: explicit spec override → HuggingFace ``ClassLabel.names`` → sorted
    unique label values as strings → ``[]`` (non-classification, or unknowable).
    Loads only the cached HF split; does not build tokenizers.
    """
    if spec.class_names_override:
        ov = spec.class_names_override
        if isinstance(ov, str):
            return [s.strip() for s in ov.split(",") if s.strip()]
        return [str(s) for s in ov]
    if spec.task != "classification" or not spec.label_column:
        return []
    hf = _load_hf(spec, spec.train_split)
    # A HF ClassLabel feature carries the human-readable names in class-id order.
    try:
        feat = hf.features.get(spec.label_column)
        names = getattr(feat, "names", None)
        if names:
            return [str(n) for n in names]
    except Exception:
        pass
    # Fallback: the distinct label values, sorted (assumes 0..N-1 int labels).
    try:
        return [str(v) for v in sorted(set(hf[spec.label_column]))]
    except Exception:
        return []


def build_detail(spec: DatasetSpec) -> dict:
    """UI metadata for the dataset-detail modal — same shape the built-in
    detail_* functions produce (labels + sample images/texts + stats), chosen by
    modality: image → sampleImages per class; language_model → one corpus sample;
    text classification → sampleTexts per class."""
    hf = _load_hf(spec, spec.train_split)
    detail: dict = {
        "name": spec.hf_id or "Custom dataset",
        "description": f"Custom HuggingFace dataset: {spec.hf_id}",
        "trainSamples": len(hf),
        "diskSize": "—",
    }
    if spec.input_kind == "image":
        labels = infer_class_names(spec)
        detail.update({
            "labels": labels,
            "channels": spec.channels,
            "imageSize": [spec.image_size, spec.image_size],
            "sampleImages": _detail_sample_images(spec, hf, labels),
        })
    elif spec.task == "language_model":
        sample = "\n".join(hf[spec.input_column][:5])[:500]
        detail.update({
            "isText": True,
            "isLanguageModel": True,
            "labels": [],
            "vocabSize": spec.vocab_size,
            "sampleTexts": {"Sample": [sample]},
        })
    else:  # text (or vector) classification / regression
        labels = infer_class_names(spec)
        detail.update({
            "isText": spec.input_kind == "text",
            "labels": labels,
            "sampleTexts": _detail_sample_texts(spec, hf, labels),
        })
    return detail


def _detail_sample_images(spec, hf, labels, per_class: int = 4, max_classes: int = 20) -> dict:
    """A few display-ready pixel arrays per class. Uses a display-only transform
    (resize + ToTensor, NO normalization) so pixels are already in 0..1 — no
    denormalizer needed for the preview."""
    import torchvision.transforms as transforms
    from dataprep.data_loaders import _tensor_to_pixels
    disp_tf = transforms.Compose([
        transforms.Resize((spec.image_size, spec.image_size)),
        transforms.ToTensor(),
    ])
    n_classes = min(len(labels), max_classes) if labels else 1
    buckets: dict[int, list] = {i: [] for i in range(n_classes)}
    for row in hf:
        try:
            lbl = int(row[spec.label_column]) if spec.label_column else 0
        except Exception:
            lbl = 0
        if lbl in buckets and len(buckets[lbl]) < per_class:
            pil = _to_pil(row[spec.input_column]).convert("L" if spec.channels == 1 else "RGB")
            buckets[lbl].append(_tensor_to_pixels(disp_tf(pil), None))
        if all(len(v) >= per_class for v in buckets.values()):
            break
    name = lambda i: labels[i] if i < len(labels) else str(i)
    return {name(k): v for k, v in buckets.items()}


def _detail_sample_texts(spec, hf, labels, per_class: int = 3) -> dict:
    """A few short text snippets per class (falls back to ungrouped samples if the
    labels aren't clean 0..N-1 ints)."""
    n_classes = len(labels) if labels else 0
    if n_classes:
        buckets: dict[int, list] = {i: [] for i in range(n_classes)}
        for row in hf:
            try:
                lbl = int(row[spec.label_column])
            except Exception:
                lbl = None
            if lbl in buckets and len(buckets[lbl]) < per_class:
                buckets[lbl].append(_snippet(row[spec.input_column]))
            if all(len(v) >= per_class for v in buckets.values()):
                break
        if any(buckets.values()):
            name = lambda i: labels[i] if i < len(labels) else str(i)
            return {name(k): v for k, v in buckets.items()}
    # fallback: first few, ungrouped
    return {"Sample": [_snippet(hf[spec.input_column][i]) for i in range(min(3, len(hf)))]}


def _snippet(text, limit: int = 200) -> str:
    text = str(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def get_raw_text(spec: DatasetSpec) -> str:
    """The whole corpus as one string — for BPE training and step-through decode
    (the resolver's resolve_raw_text calls this for custom text datasets)."""
    hf = _load_hf(spec, spec.train_split)
    return "\n".join(hf[spec.input_column])


def get_lm_vocab(spec: DatasetSpec) -> tuple[dict, dict]:
    """(token→id, id→token) maps for decoding language-model output, mirroring the
    built-in get_shakespeare_vocab. Which tokenizer decides the token unit."""
    tk = spec.tokenizer
    if tk == "char":
        token2idx = {ch: i for i, ch in enumerate(sorted(set(get_raw_text(spec))))}
    elif tk == "bpe":
        from dataprep.bpe import get_bpe_tokenizer
        token2idx = get_bpe_tokenizer(get_raw_text(spec), spec.vocab_size, spec.cache_key).vocab
    elif tk == "word":
        from dataprep.data_loaders import _build_vocab
        token2idx = _build_vocab([get_raw_text(spec)], spec.vocab_size, spec.cache_key)
    else:
        raise NotImplementedError(f"language_model tokenizer '{tk}'")
    idx2token = {i: t for t, i in token2idx.items()}
    return token2idx, idx2token
