# Plan: Modality-agnostic custom dataset node (`data.custom`)

> Status: **implemented** — [issue #7](https://github.com/WilsonZheng0327/nodetorch/issues/7).
> This is the original design; see [`../features/custom-datasets.md`](../features/custom-datasets.md)
> for how it actually shipped.

## Context

Every dataset in NodeTorch is a hardcoded node type (`data.mnist`, `data.ag_news`, …)
whose behavior is spread across parallel registries in
`backend/dataprep/data_loaders.py` (`DATA_LOADERS`, `TRAIN_DATASETS`/`TEST_DATASETS`,
`DATASET_DETAILS`, `DENORMALIZERS`, `CLASS_NAMES`, `LM_DATASET_TYPES`, `get_raw_texts`).
Adding a dataset means editing that file + a frontend node; users can't bring their own
data.

The requirement: a **flexible structure that fits ANY dataset type** (image
classification, text classification, next-token LM, extensible to
regression/autoencoder/tabular) — **not** an image-only MVP. Design the general
abstraction up front and avoid the parallel-registry mess growing per modality.

Key facts that make this tractable:
- HF `datasets` is **already** a dependency (`requirements.txt`), already used by
  imdb/ag_news via `load_dataset(...)`. No new dependency.
- The text pipeline already exists and is reused: `_tokenize`, `_build_vocab`,
  `_encode_texts` (`data_loaders.py`) and BPE (`bpe.py`: `BPETextDataset`, `BPELMDataset`).
- Every backend site already **finds the data node** by scanning `node["type"] in
  DATA_LOADERS`, so the node dict (with `properties`) is in scope everywhere.

## Core design

Two moves:

1. **A declarative `DatasetSpec` + one generic `CustomDataset` interpreter** — a custom
   dataset is fully described by two orthogonal axes, `input_kind` (image|text|vector) ×
   `task` (classification|language_model|regression|autoencoder), plus source (HF id/config/
   splits), column mapping, and per-kind transform params. One `CustomDataset` reads the
   spec and returns the same `(input, label)` tensors the built-ins produce.

2. **A node-aware resolution layer** — replace registry lookups keyed by the bare
   `dataset_type` **string** with resolver calls keyed by the **node dict**. `data.custom`
   → build from spec; every other type → the *exact same* existing registry call
   (built-ins stay byte-identical). `data.custom` is registered in `DATA_LOADERS` so all
   `in DATA_LOADERS` finder sites keep working untouched.

## Phase 1 — Spec + interpreter (new, self-contained)

New `backend/dataprep/custom_dataset.py`:

- **`DatasetSpec`** dataclass + `from_props(props, node_id)`. Fields: `input_kind`, `task`;
  source `hf_id`, `hf_config`, `train_split`, `test_split`, `trust_remote_code`;
  mapping `input_column`, `label_column`; image `image_size`, `channels`,
  `normalize_mean/std`; text `tokenizer`(char|word|bpe), `vocab_size`, `max_len`,
  `seq_len`; `class_names_override`, `batch_size`, and a `cache_key` derived from
  `node_id`+`hf_id` (so per-node vocab/BPE caches are stable, mirroring `_build_vocab`).
- **`CustomDataset(spec, split)`** with two `__getitem__` paths:
  - *row path* (classification/regression/autoencoder): one HF row → `_encode_input(row)` +
    `_make_target(row, x)`. image = torchvision `Resize/Grayscale/ToTensor/Normalize`;
    text = reuse `_encode_texts`/BPE; vector = `torch.tensor(...)`. Targets: class idx
    (long scalar) / float / `x` itself (autoencoder).
  - *corpus path* (language_model): concatenate + chunk `seq_len+1` → `(chunk[:-1],
    chunk[1:])`, exactly like `TinyShakespeareDataset`/`BPELMDataset`.
  - v1 wires the 9 sensible cells; the cells that don't fit (e.g. image+LM) raise
    `NotImplementedError` with a clear message. Covers the minimum
    (image+cls, text+cls, text+LM) and the cheap extras.
- **`infer_class_names(spec, hf_ds)`**: override → HF `ClassLabel.names` → sorted unique →
  `[]`. **`build_detail(spec)`**: same dict shape as `detail_mnist`/`detail_imdb`/
  `detail_tiny_shakespeare` (sampleImages / sampleTexts / isLanguageModel…).
  **`get_raw_text()`** + **`get_lm_vocab()`** for the BPE/step-through decode paths.

## Phase 2 — Resolution layer + call-site refactor (built-ins unchanged)

New `backend/dataprep/resolve.py` — node-aware resolvers, each `custom → spec; else →
existing registry keyed by node["type"]`:
`resolve_train_dataset`, `resolve_test_dataset`, `resolve_detail`, `resolve_class_names`,
`resolve_denormalizer` (custom image → `img*std+mean` closure or identity; else
`DENORMALIZERS.get`), `resolve_is_lm`, `resolve_raw_text`. The `else` branches preserve the
current signature-introspection (`load_dataset` passes matching props to built-in factories).

Register **`load_custom(props)`** in `DATA_LOADERS["data.custom"]` (builds a `CustomDataset`,
returns one batch) — satisfies the runner batch-load *and* every `in DATA_LOADERS` finder.

Refactor **only** the props-dependent sites (finders untouched):
- `backend/training/base.py`: `load_dataset`/`load_bpe_dataset` → resolvers; add
  `class_names` + `denorm` (resolved once) to `TrainingContext`; `build_epoch_result`
  classNames → `ctx.class_names`.
- `backend/training/__init__.py` `detect_training_mode`, `inference.py` LM guards →
  `resolve_is_lm(node)`.
- `inference.py` test set → `resolve_test_dataset`; classNames → `resolve_class_names`.
- **Signature change (the one non-mechanical bit):** `pick_tracked_samples`,
  `probe_tracked_samples`, `collect_misclassifications`, and the `_tensor_to_preview_image`
  helpers in `step_through.py`/`backprop_sim.py` take a **`denorm` callable** instead of a
  `dataset_type` string; callers pass `resolve_denormalizer(data_node)` (identical closure
  to today's hardcoded lambdas for built-ins).
- classNames sites in `detail.py`, `step_through.py`, `backprop_sim.py` →
  `resolve_class_names(node)`; `_pick_hard_sample` → `resolve_train_dataset(node)`.
- Custom text+BPE: add a `bpe` path so `load_bpe_dataset` wraps a custom text dataset
  (only genuinely new BPE integration; scope to text+classification).
- Secondary preview sites keyed only by a type string (`runners._add_image_preview`,
  `layers/data.py`) fall back to raw-clamp for custom in v1 (`# TODO custom-denorm`) — the
  important dashboard preview uses `ctx.denorm` correctly.

## Phase 3 — Detail endpoint + modal

`backend/api/data.py`: add `POST /dataset-detail` `{type, properties}` → `resolve_detail`
(the `GET /dataset/{type}` URL can't carry props); keep GET delegating for built-ins.
`src/ui/inspector/DatasetDetail.tsx`: fetch via POST with the node's `type`+`properties`;
render code already branches on isText/isLanguageModel/sampleImages/sampleTexts — no render
change. Thread the selected node's `properties` from the inspector caller.

## Phase 4 — Frontend node `data.custom`

New `src/domain/nodes/data/custom.ts` (append to `dataNodes` in `data/index.ts`).
Grouped properties with `visible` conditionals: Source (hfId, hfConfig, train/testSplit,
trustRemoteCode), Mapping (inputColumn, labelColumn), Task (inputKind, task, batchSize),
Image group (imageSize, channels, normalizeMean/std, `visible: inputKind==='image'`), Text
group (tokenizer, vocabSize, maxLen/seqLen, `visible: inputKind==='text'`), classNames
override. `inputKind`/`task` are `affects:'both'`. Ports `out`+`labels` (static, like every
built-in). **Shape executor computes output shape purely from props** (image →
`[B,C,H,W]`; text-cls → `[B,maxLen]`; LM → `[B,seqLen]`), no backend round-trip;
`metadata.datasetType:'data.custom'`.

## Files

Create: `backend/dataprep/custom_dataset.py`, `backend/dataprep/resolve.py`,
`src/domain/nodes/data/custom.ts`, `tests/backend/test_custom_dataset.py`.
Modify: `backend/dataprep/data_loaders.py` (register `load_custom`),
`backend/training/{base,__init__,standard,diffusion}.py`,
`backend/engine/graph_builder/{inference,runners,detail}.py`,
`backend/visualize/{step_through,backprop_sim}.py`, `backend/visualize/layers/data.py`,
`backend/api/data.py`, `src/domain/nodes/data/index.ts`,
`src/ui/inspector/DatasetDetail.tsx` (+ inspector caller).

## Verification

Commit-sized phases (each independently green):
1. Spec+interpreter — pytest asserts `CustomDataset(spec,'train')[0]` shapes for
   image-cls / text-cls / LM + `infer_class_names`.
2. Resolution layer — **existing `tests/backend/` suite passes unchanged** (built-in
   invariance) + a text-cls custom graph runs `build_training_context`→`run_training` 1
   epoch, classNames + trackedSamples populated.
3–4. Detail POST + `data.custom` node — `npm run typecheck`/build; drop node, conditional
   props toggle, shape propagates, detail modal hits POST.
5. End-to-end pytest, one graph per modality against a small HF set (image-cls,
   text-cls on `ag_news`, char-LM): accuracy/perplexity present, `detect_training_mode`
   returns `autoregressive` for LM via `resolve_is_lm`, `resolve_detail` shape matches.
Run backend in-process (no server); free port 8000 if ever started.

## Open decisions for review

- **Modality×task coverage in v1** — wire all 9 cells the interpreter cleanly supports, or
  restrict to the 3 required (image-cls, text-cls, text-LM)? Recommended: wire the 9 (the
  extras are a few lines each and prove the abstraction is genuinely general).
- **Data source** — HuggingFace Hub only in v1 (spec has a `source` field, so
  upload/URL can be added later without touching the resolvers). Recommended.
- **Trust model** — default `trust_remote_code=False` (non-scripted datasets only), with a
  per-node opt-in property. Recommended.
- **Custom image preview denorm** — v1 uses `ctx.denorm` for the dashboard (correct) and
  raw-clamp for a couple of secondary previews; thread the node there later if needed.
