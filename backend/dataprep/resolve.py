# resolve.py — node-aware dataset resolution.
#
# WHAT A "RESOLVER" IS
# ────────────────────
# Each resolver answers one question about a data node: "given this node, what is
# its <X>?" — where <X> is one component the rest of the backend needs from a
# dataset. There is one resolver per component:
#
#     resolve_train_dataset  → the training torch Dataset
#     resolve_test_dataset   → the held-out test torch Dataset
#     resolve_detail         → the inspector detail card (labels/samples/stats)
#     resolve_class_names    → the class-label list
#     resolve_denormalizer   → the image preview de-normalizer
#     resolve_is_lm          → whether it's a language-model dataset
#     resolve_raw_text       → the raw corpus string
#
# You hand a resolver the node dict; it hands back that one component, ready to
# use. So yes — a resolver returns "the respective component of the dataset,"
# and it figures out how to produce it based on the node's type.
#
# WHY THEY EXIST (the custom-vs-built-in split)
# ─────────────────────────────────────────────
# Built-in datasets are keyed by a static node type across several parallel
# registries in data_loaders.py: TRAIN_DATASETS[type], CLASS_NAMES[type],
# DENORMALIZERS[type], and so on. A built-in's behavior is fixed by its type.
#
# The custom dataset (`data.custom`) has no fixed behavior — it's derived from
# the node's *properties* (parsed into a DatasetSpec, interpreted by
# CustomDataset in custom_dataset.py). There's nothing to look up in a registry.
#
# A resolver hides that difference behind a single call. Internally every one
# does the same two-way branch:
#
#     data.custom  → build the component from the node's DatasetSpec
#     any built-in → the exact same registry lookup the old code did
#
# So a call site writes `resolve_class_names(node)` instead of
# `CLASS_NAMES.get(node["type"], [])`, and it now works for BOTH kinds without
# the caller knowing which it holds. Built-in behavior is unchanged — the else
# branch calls the identical registry function — so only `data.custom` takes the
# new spec-driven path.
#
# (Note: `data.custom` is ALSO registered in DATA_LOADERS, so the many
# `node["type"] in DATA_LOADERS` "is this the data node?" checks scattered across
# the backend recognize it with no change.)

import inspect

import torch

from dataprep import data_loaders as dl
from dataprep.custom_dataset import (
    DatasetSpec, CustomDataset,
    infer_class_names, build_detail, get_raw_text,
)

CUSTOM_TYPE = "data.custom"


def is_custom(node: dict) -> bool:
    """True if this data node is the generic custom dataset (vs. a built-in)."""
    return node.get("type") == CUSTOM_TYPE


def _spec(node: dict) -> DatasetSpec:
    """Parse the node's properties into a DatasetSpec (the custom-branch input)."""
    return DatasetSpec.from_props(node.get("properties", {}), node.get("id", ""))


def _call_factory(builder, props: dict):
    """Call a built-in dataset factory, forwarding whichever props match its
    signature — the same introspection base.load_dataset / evaluate_test_set use."""
    params = inspect.signature(builder).parameters
    if params:
        return builder(**{k: props[k] for k in params if k in props})
    return builder()


def resolve_train_dataset(node: dict):
    """The training Dataset for this data node.

    Custom → a CustomDataset over the spec's train split. Built-in → the
    registered TRAIN_DATASETS factory, called with matching node properties.

    Returns:
        torch.utils.data.Dataset — the (un-batched) training dataset; or None if a
        built-in type has no registered training factory. Val-splitting is the
        caller's job (see base.load_dataset)."""
    if is_custom(node):
        spec = _spec(node)
        return CustomDataset(spec, spec.train_split)
    builder = dl.TRAIN_DATASETS.get(node["type"])
    return _call_factory(builder, node.get("properties", {})) if builder else None


def resolve_test_dataset(node: dict):
    """The held-out test Dataset for this data node.

    Returns:
        torch.utils.data.Dataset — the test dataset; or None when there is none:
        a built-in without a registered test factory, or a custom dataset whose
        spec has no test_split."""
    if is_custom(node):
        spec = _spec(node)
        return CustomDataset(spec, spec.test_split) if spec.test_split else None
    builder = dl.TEST_DATASETS.get(node["type"])
    return _call_factory(builder, node.get("properties", {})) if builder else None


def resolve_detail(node: dict) -> dict | None:
    """The dataset-detail card shown in the inspector modal (name, description,
    class labels, a few sample images or texts, and dataset stats).

    Returns:
        dict — shaped like the built-in detail_* functions (image datasets carry
        `sampleImages`, text carry `sampleTexts`, LMs carry `isLanguageModel`);
        or None if the type is unknown."""
    if is_custom(node):
        return build_detail(_spec(node))
    fn = dl.DATASET_DETAILS.get(node["type"])
    return fn() if fn else None


def resolve_class_names(node: dict) -> list[str]:
    """The human-readable class labels, indexed by class id.

    Returns:
        list[str] — e.g. ["World", "Sports", "Business", "Sci/Tech"]; an empty
        list for non-classification datasets or when labels can't be determined."""
    if is_custom(node):
        return infer_class_names(_spec(node))
    return dl.CLASS_NAMES.get(node["type"], [])


def resolve_denormalizer(node: dict):
    """The preview de-normalizer that undoes a dataset's input normalization, so
    a stored (normalized) image tensor can be shown as real pixels.

    Returns:
        A callable img[C,H,W] → img[C,H,W], or None when not applicable.
        • Built-in → the per-type denormalizer (None for non-image datasets like
          text).
        • Custom image → reverses the spec's Normalize (img*std + mean), or the
          identity function when the custom dataset didn't normalize (ToTensor
          output is already 0..1).
        • Custom non-image → None."""
    if is_custom(node):
        spec = _spec(node)
        if spec.input_kind != "image":
            return None
        if spec.normalize_mean is not None and spec.normalize_std is not None:
            mean = torch.tensor(spec.normalize_mean).view(-1, 1, 1)
            std = torch.tensor(spec.normalize_std).view(-1, 1, 1)
            return lambda img: img * std + mean
        return lambda img: img
    return dl.DENORMALIZERS.get(node["type"])


def resolve_is_lm(node: dict) -> bool:
    """Whether this data node drives autoregressive / language-model training
    (next-token prediction). Selects the autoregressive training paradigm and the
    text-generation / token-decoding paths.

    Returns:
        bool — custom: True when the spec's task is "language_model"; built-in:
        True when the type is in LM_DATASET_TYPES."""
    if is_custom(node):
        return _spec(node).task == "language_model"
    return node["type"] in dl.LM_DATASET_TYPES


def resolve_raw_text(node: dict) -> str:
    """The dataset's raw corpus as one string — used to learn BPE merges and to
    decode token ids back to text in the step-through views.

    Returns:
        str — the concatenated corpus (custom), or the built-in dataset's raw
        text via get_raw_texts (empty string if it has none)."""
    if is_custom(node):
        return get_raw_text(_spec(node))
    return dl.get_raw_texts(node["type"])
