# `sequence_pool` "last" reads padding → text classifier stuck at 50%

- **Status:** fixed (2026-07-12)
- **Area:** `sequence_pool` node · text classification · padding
- **Symptom severity:** silent — trains without error, just never learns

## Symptom

An LSTM sentiment model on **Rotten Tomatoes** (binary, via the new `data.custom`
node) sat at **~50% accuracy** — pure chance for a balanced 2-class dataset —
across every epoch. No error, no warning; loss hovered around `ln 2 ≈ 0.69`. The
identical architecture on the built-in IMDb/AG-News presets trained fine, which is
what made it look like a custom-node bug at first.

## Root cause

`sequence_pool` with `mode: "last"` was literally:

```python
x[:, -1, :]   # take the hidden state at the final timestep
```

It has no awareness of padding. The pipeline right-pads every example to a fixed
length so batches are rectangular:

```text
tokens: [718, 6223, 73, ..., 1602, 0, 0, 0, ... 0]   # ~20 real tokens, then ~44 zeros (pad id 0)
        └────────── real review ──────────┘ └─────────── padding ───────────┘
```

Rotten Tomatoes reviews are ~20 words, padded to `maxLen=64`, so **~44 of 64
positions are padding**. The LSTM processes the real tokens *and then keeps
running over 44 pad tokens*, overwriting its state. `x[:, -1, :]` therefore reads
the hidden state **after 44 pad steps**, not after the last real word.

That final vector is nearly identical for every example (they all end in a long
pad run), so the classifier receives an almost-constant input, collapses to
predicting one class, and lands at exactly 50% on a balanced set.

### Why it stayed hidden

The built-in text presets use `"last"` too, but their text is long:

| Dataset | maxLen | typical length | padding | `"last"` reads |
|---|---|---|---|---|
| IMDb | 256 | ~230 words | little | a **real** token ✓ |
| AG News | 128 | ~40 words | some | mostly real ✓ |
| **Rotten Tomatoes** | 64 | **~20 words** | **~44 pad** | **padding** ✗ |

`"last"` only works when the text **fills** the sequence. The bug was latent in the
shipped presets and only surfaced once a *short*-text dataset could be plugged in —
which the `data.custom` node newly made possible.

### The deeper gap

Nothing in a text model carried **sequence-length / padding** information:

- `embedding` had no `padding_idx`, so the pad token `0` got a normal learned
  embedding, indistinguishable from a real word.
- `nn.LSTM` ran unpacked, so its `hidden` output (`h_n`) was *also* the
  post-padding state — i.e. connecting `lstm.hidden` directly had the same bug.
- `sequence_pool` saw only the `[B, T, H]` tensor, with no way to know where the
  real tokens ended.

So `"last"` wasn't uniquely broken — there was simply **no mask anywhere** for any
padding-correct reduction to use.

## Fix

Added an **optional `mask` input** to `sequence_pool`. Wire the data node's token
output to it (`data.out → pool.mask`) and pooling ignores pad positions; leave it
unwired and behavior is unchanged (backward compatible). The pool derives the mask
itself as `tokens != 0` (pad id `0` by convention, right-padded):

```python
last  → x[arange(B), (mask != 0).sum(1) - 1]      # each row's final REAL token
mean  → (x * valid).sum(1) / valid.sum(1)          # divide by real length, not T
max   → x.masked_fill(~valid, -inf).max(1).values  # a pad state can't win
```

Result on Rotten Tomatoes with masked `"last"`: **~82% val accuracy** (was 50%).

Implementation: `SequencePoolModule` (`backend/engine/node_builders.py`), the two
dispatch sites (`engine/forward_utils.py`, `engine/graph_builder/runners.py`), the
node's `mask` port (`src/domain/nodes/structural/sequence-pool.ts`), and
`tests/backend/test_sequence_pool.py`. The design keeps the mask a visible
Layer-5/6 wire — no engine/Layer-3 plumbing.

## Guidance (when is `"last"` even right?)

`"last"` is valid only when the **final position has seen the whole sequence** and
is a real token, not padding:

- **After a unidirectional LSTM/GRU** — valid *with the mask*, but redundant with
  the encoder's own `hidden` output.
- **After causal (GPT-style) attention** (`attention` node with `causalMask: true`)
  — valid and genuinely useful: a transformer has no `hidden` port, so slicing the
  last real token is the way to read the summary. Still needs the mask.
- **Not valid** after a plain `embedding` (last position is one word), a
  bidirectional RNN (the backward direction at the last position has seen almost
  nothing), or non-causal / BERT-style attention (no position is "the summary").

For everything else, prefer **masked `"mean"` / `"max"`** — they read the whole
sequence and don't depend on the encoder carrying information to the end.

## Lessons

- **A rectangular batch hides a ragged truth.** Any per-timestep reduction
  (`last`, and un-masked `mean`/`max`) is wrong under padding unless it's told
  where the real tokens end. Padding needs to be a first-class signal, not an
  invisible tensor-shape detail.
- **Silent degradation is the dangerous kind.** No crash, plausible loss — it just
  never learns. A model stuck at exactly chance on a balanced set is a tell that
  the representation carries no signal.
- **A latent bug surfaces when the input space widens.** `"last"` was quietly
  wrong for short text all along; adding arbitrary HuggingFace datasets is what
  finally fed it short text. New input surfaces re-test old assumptions.

## Follow-ups (not yet done)

- `padding_idx=0` on the `embedding` builder — zeroes pad embeddings (helps a
  plain `embedding → pool` model that has no RNN to learn around them).
- Wire the mask into the built-in `lstm-agnews` / `lstm-sentiment-imdb` presets —
  they still use unmasked `"last"` and only get away with it via longer text.
