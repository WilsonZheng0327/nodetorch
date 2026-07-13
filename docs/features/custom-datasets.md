# Custom Datasets (`data.custom`)

One generic data node that loads **any HuggingFace dataset** and standardizes it
to the same `(input, label)` tensor contract the built-in datasets produce — so
it flows through training, inference, test-set eval, and every visualization
unchanged. Implements [issue #7](https://github.com/WilsonZheng0327/nodetorch/issues/7).

## 1. The problem: datasets were keyed by a static type

Every built-in dataset is a hardcoded node type (`data.mnist`, `data.ag_news`, …)
whose behavior is spread across parallel registries in
[`backend/dataprep/data_loaders.py`](../../backend/dataprep/data_loaders.py):

| Registry | Signature | What it gives |
|----------|-----------|---------------|
| `DATA_LOADERS` | `(props) → {port: tensor}` | one batch (preview / shape) |
| `TRAIN_DATASETS` / `TEST_DATASETS` | `() → Dataset` | full split for the DataLoader |
| `DATASET_DETAILS` | `() → dict` | inspector card (labels, samples, stats) |
| `DENORMALIZERS` | `(img) → img` | undo normalization for preview |
| `CLASS_NAMES` | `dict[type, list[str]]` | class labels |
| `LM_DATASET_TYPES` | `set[str]` | which types are language models |

A built-in's behavior is fixed **by its type**. Adding a dataset meant editing
that file and adding a frontend node — there was no way for a user to bring their
own data.

## 2. Core design: one spec, one interpreter

A custom dataset is described **declaratively** by a `DatasetSpec` and interpreted
by one generic `CustomDataset` — both in
[`backend/dataprep/custom_dataset.py`](../../backend/dataprep/custom_dataset.py).

Two orthogonal axes fully classify a dataset:

- **`input_kind`** — how each sample is encoded: `image | text | vector`
- **`task`** — how the target is formed: `classification | language_model | regression | autoencoder`

The rest of the spec is the HuggingFace source (`hf_id`, `hf_config`, split names),
the column mapping (`input_column`, `label_column`), and per-kind transform params
(image resize/channels/normalize; text tokenizer/vocab/lengths). `DatasetSpec.from_props`
parses it from the node's properties.

`CustomDataset(spec, split)` has **two `__getitem__` paths**:

- **row path** (classification / regression / autoencoder) — one HF row per sample:
  `_preprocess_input(row)` builds the input tensor (image → resize/normalize;
  text → tokenize via the existing `_encode_texts` / BPE; vector → `torch.tensor`),
  and `_make_target(row, x)` forms the target (class id / float / the input itself).
- **corpus path** (language_model) — the whole corpus is tokenized into one flat
  id stream and cut into consecutive `seq_len`-token windows, returning
  `(chunk[:-1], chunk[1:])`. This mirrors the built-in `TinyShakespeareDataset` /
  `BPELMDataset` exactly.

The 3×4 grid gives 12 cells; v1 wires the 9 that map cleanly onto `(input, label)`
(the `_SUPPORTED` set). Unsupported cells (e.g. image + language_model) raise a
clear `NotImplementedError` at construction.

**Reuse, not reinvention:** HF `datasets` was already a dependency (imdb/ag_news
use it), and the text pipeline (`_tokenize`, `_build_vocab`, `_encode_texts`,
`bpe.py`) already existed. `CustomDataset` is a thin, general front over those.

## 3. The node-aware resolution layer

The registries above are keyed by a **type string** and can't express a
props-derived dataset. The fix is a layer of **resolvers** in
[`backend/dataprep/resolve.py`](../../backend/dataprep/resolve.py) — one per
dataset component:

```
resolve_train_dataset · resolve_test_dataset · resolve_detail · resolve_class_names
resolve_denormalizer  · resolve_is_lm        · resolve_raw_text
```

Each takes the **node dict** (which carries `properties`) and does the same
two-way branch:

```python
def resolve_class_names(node):
    if is_custom(node):
        return infer_class_names(_spec(node))     # build from the spec
    return dl.CLASS_NAMES.get(node["type"], [])   # the exact old registry call
```

So a call site writes `resolve_class_names(node)` instead of
`CLASS_NAMES.get(node["type"], [])`, and it works for **both** kinds without
knowing which it holds. Because the else-branch is the identical registry call,
**built-in datasets are byte-identical** — only `data.custom` takes the new path.

`data.custom` is also registered in `DATA_LOADERS` (as `load_custom`), so the many
`node["type"] in DATA_LOADERS` "find the data node" checks recognize it with no
change. The call sites that were migrated to resolvers:
`training/base.py`, `training/__init__.py`, `training/standard.py`,
`training/diffusion.py`, `engine/graph_builder/{inference,runners,detail}.py`,
`visualize/{step_through,backprop_sim}.py`.

One deliberate signature change: `pick_tracked_samples` / `probe_tracked_samples`
/ `collect_misclassifications` (in `inference.py`) took a `dataset_type` string to
look up the denormalizer; they now take a resolved `denorm` **callable**, threaded
from `ctx.denorm` (resolved once in `build_training_context`).

## 4. The frontend node

[`src/domain/nodes/data/custom.ts`](../../src/domain/nodes/data/custom.ts) — a
standard `NodeDefinition` registered in `data/index.ts`. Its properties are
grouped (Source / Task / Mapping / Image / Text) with `visible` predicates so the
panel shows only the relevant params:

- Image params (`imageSize`, `channels`, `normalize*`) — visible when `inputKind === 'image'`
- Text params (`tokenizer`, `vocabSize`) — visible when `inputKind === 'text'`
- `maxLen` for classification, `seqLen` for language_model
- `labelColumn` / `classNames` hidden for language_model

The **shape executor computes the output shape purely from properties** (image →
`[B,C,H,W]`; text-cls → `[B,maxLen]`; LM → `[B,seqLen]`) — no backend round-trip,
so shape mode stays instant and offline. It sets `metadata.datasetType:'data.custom'`,
which is what the inspector uses to show the "View Dataset Details" button.

## 5. The detail endpoint

`GET /dataset/{type}` can't carry properties, so a custom dataset's detail card
needs a new route. [`backend/api/data.py`](../../backend/api/data.py) adds
`POST /dataset-detail` with body `{type, id?, properties}`, routed through
`resolve_detail`. Both GET and POST go through the same resolver, so built-ins are
unaffected. The frontend modal
([`src/ui/inspector/DatasetDetail.tsx`](../../src/ui/inspector/DatasetDetail.tsx))
POSTs the node's `type` + `properties`; its render code already branches on
`isText` / `isLanguageModel` / `sampleImages` / `sampleTexts`, which
`build_detail` produces — so no render change was needed.

The AI assistant reuses this endpoint: its `get_dataset_info(nodeId)` tool
(see [agent.md](../explanation/agent.md)) POSTs the node's live properties and
summarizes the reply for the model — counts, class names, image/vocab facts, no
pixel arrays — so the agent can verify an `hfId` actually loads and size the rest
of the graph from the real class count. The node's property `help`/`group`
strings also reach the agent through the node catalog.

## 6. Adding a new modality or task cell

The abstraction is meant to grow along its two axes:

1. Add the cell to `_SUPPORTED` in `custom_dataset.py`.
2. Handle the new `input_kind` in `_preprocess_input` (and its `__init__` setup),
   or the new `task` in `_make_target` (or the corpus path).
3. Extend the frontend `shape` executor + any conditional properties.

Because the resolution layer and every call site are modality-agnostic, nothing
downstream changes — the new cell flows through training/inference/viz the moment
the interpreter produces the right `(input, label)` tensors.

## 7. v1 scope & known corners

- **Source:** HuggingFace Hub only. The spec has a `source` seam so upload / URL
  can be added later without touching the resolvers.
- **Trust:** `trust_remote_code` defaults **off** (non-scripted datasets); a
  per-node property opts in.
- **Preview denorm:** the training dashboard preview uses `ctx.denorm` (correct).
  Two secondary previews keyed only by a type string
  (`runners._add_image_preview`, `visualize/layers/data.py`) fall back to
  raw-clamp for custom images — a minor cosmetic gap only when custom
  normalization is on.
- **Vector output shape:** the feature dimension isn't known without loading, so
  the frontend shape is a placeholder for `inputKind === 'vector'`.

## Files

- Interpreter: `backend/dataprep/custom_dataset.py`
- Resolvers: `backend/dataprep/resolve.py`
- Batch loader registration: `backend/dataprep/data_loaders.py` (`load_custom`)
- Detail endpoint: `backend/api/data.py`
- Frontend node: `src/domain/nodes/data/custom.ts`
- Detail modal: `src/ui/inspector/DatasetDetail.tsx`
- Tests: `tests/backend/test_custom_dataset.py`, `tests/backend/test_resolve.py`,
  `tests/backend/test_custom_e2e.py`, `tests/frontend/domain/custom-node.test.ts`
