"""Tokenizer preview — build vocab + frequencies from the upstream corpus.

Used by the layer-detail panel to show what a tokenizer node would produce
WITHOUT having to run training. Each function takes the dataset type and the
tokenizer's node properties, and returns:
  {
    vocab: [{id, token, freq}, ...],   # token list, ready to display
    stats: {numTokens, totalTokens, specialTokens, corpusChars, ...},
    merges?: [[a, b], ...],            # BPE only — learned merge rules
    sampleEncode?: [{id, token}, ...], # if `sample_text` supplied
  }

Results are cached behind the underlying vocab/BPE caches.
"""
from __future__ import annotations
import re
from collections import Counter

from bpe import get_bpe_tokenizer
from data_loaders import get_raw_texts


_CHAR_VOCAB_CACHE: dict[str, dict] = {}


def _split_words(text: str, lowercase: bool, split_punctuation: bool) -> list[str]:
    """Split into word-level tokens, matching the loader's regex."""
    if lowercase:
        text = text.lower()
    if split_punctuation:
        # Letters, numbers, single punctuation chars
        return re.findall(r"[a-zA-Z]+|[0-9]+|[^\s\w]", text)
    # Whitespace split, keep punctuation glued to words
    return [t for t in text.split() if t]


# --- Character ---

def preview_char_vocab(dataset_type: str, lowercase: bool, sample_text: str | None = None) -> dict:
    """Build a character vocab from the corpus."""
    raw = get_raw_texts(dataset_type)
    if not raw:
        return {"error": f"No corpus available for {dataset_type}"}

    cache_key = f"{dataset_type}_{'lc' if lowercase else 'cs'}"
    if cache_key in _CHAR_VOCAB_CACHE:
        cached = _CHAR_VOCAB_CACHE[cache_key]
    else:
        text = raw.lower() if lowercase else raw
        counter: Counter = Counter(text)
        # Reserve 0=pad, 1=unk so token IDs match the runtime behavior
        vocab_list = [
            {"id": 0, "token": "<pad>", "freq": 0, "special": True},
            {"id": 1, "token": "<unk>", "freq": 0, "special": True},
        ]
        idx = 2
        for ch, freq in counter.most_common():
            vocab_list.append({"id": idx, "token": ch, "freq": freq})
            idx += 1
        token_to_id = {v["token"]: v["id"] for v in vocab_list}
        cached = {
            "vocab": vocab_list,
            "token_to_id": token_to_id,
            "stats": {
                "numTokens": len(vocab_list),
                "specialTokens": 2,
                "corpusChars": len(text),
                "totalTokens": sum(counter.values()),
                "uniqueChars": len(counter),
            },
        }
        _CHAR_VOCAB_CACHE[cache_key] = cached

    result = {
        "vocab": cached["vocab"],
        "stats": cached["stats"],
    }
    if sample_text:
        sample = sample_text.lower() if lowercase else sample_text
        unk = 1
        result["sampleEncode"] = [
            {"id": cached["token_to_id"].get(ch, unk), "token": ch}
            for ch in sample
        ]
    return result


# --- Word ---

_WORD_VOCAB_CACHE: dict[str, dict] = {}


def preview_word_vocab(
    dataset_type: str,
    vocab_size: int,
    lowercase: bool,
    split_punctuation: bool,
    sample_text: str | None = None,
) -> dict:
    """Build a word-frequency vocab from the corpus."""
    raw = get_raw_texts(dataset_type)
    if not raw:
        return {"error": f"No corpus available for {dataset_type}"}

    cache_key = f"{dataset_type}_{vocab_size}_{'lc' if lowercase else 'cs'}_{'sp' if split_punctuation else 'nsp'}"
    if cache_key in _WORD_VOCAB_CACHE:
        cached = _WORD_VOCAB_CACHE[cache_key]
    else:
        tokens = _split_words(raw, lowercase, split_punctuation)
        counter: Counter = Counter(tokens)
        vocab_list = [
            {"id": 0, "token": "<pad>", "freq": 0, "special": True},
            {"id": 1, "token": "<unk>", "freq": 0, "special": True},
        ]
        idx = 2
        kept = counter.most_common(max(0, vocab_size - 2))
        for word, freq in kept:
            vocab_list.append({"id": idx, "token": word, "freq": freq})
            idx += 1

        oov_tokens = len(counter) - len(kept)
        oov_count = sum(counter.values()) - sum(f for _, f in kept)

        token_to_id = {v["token"]: v["id"] for v in vocab_list}
        cached = {
            "vocab": vocab_list,
            "token_to_id": token_to_id,
            "stats": {
                "numTokens": len(vocab_list),
                "specialTokens": 2,
                "totalTokens": sum(counter.values()),
                "uniqueWords": len(counter),
                "oovTokens": oov_tokens,
                "oovCount": oov_count,
                "coverage": round(1.0 - (oov_count / max(1, sum(counter.values()))), 4),
            },
        }
        _WORD_VOCAB_CACHE[cache_key] = cached

    result = {
        "vocab": cached["vocab"],
        "stats": cached["stats"],
    }
    if sample_text:
        sample_tokens = _split_words(sample_text, lowercase, split_punctuation)
        unk = 1
        result["sampleEncode"] = [
            {"id": cached["token_to_id"].get(t, unk), "token": t}
            for t in sample_tokens
        ]
    return result


# --- BPE ---

def preview_bpe_vocab(
    dataset_type: str,
    vocab_size: int,
    lowercase: bool,
    end_of_word_marker: str,
    sample_text: str | None = None,
) -> dict:
    """Train (or reuse cached) BPE tokenizer; return vocab + merges."""
    raw = get_raw_texts(dataset_type)
    if not raw:
        return {"error": f"No corpus available for {dataset_type}"}

    bpe = get_bpe_tokenizer(
        raw, vocab_size, cache_key=dataset_type,
        lowercase=lowercase, end_of_word_marker=end_of_word_marker,
    )

    # Build vocab list with frequencies from bpe.token_freqs.
    # Order by ID (vocab insertion order: <pad>, <unk>, base chars, then merges).
    inv_vocab = {v: k for k, v in bpe.vocab.items()}
    vocab_list = []
    for token_id in sorted(inv_vocab.keys()):
        token = inv_vocab[token_id]
        is_special = token in ("<pad>", "<unk>")
        freq = bpe.token_freqs.get(token, 0)
        vocab_list.append({
            "id": token_id,
            "token": token,
            "freq": freq,
            **({"special": True} if is_special else {}),
        })

    merges = [[a, b] for (a, b) in bpe.merges]

    result = {
        "vocab": vocab_list,
        "merges": merges,
        "stats": {
            "numTokens": bpe.vocab_size,
            "specialTokens": 2,
            "numMerges": len(bpe.merges),
            "baseChars": bpe.vocab_size - 2 - len(bpe.merges),
            "endOfWordMarker": bpe.end_of_word_marker,
            "lowercase": bpe.lowercase,
        },
    }
    if sample_text:
        ids = bpe.encode(sample_text)
        result["sampleEncode"] = [
            {"id": tid, "token": inv_vocab.get(tid, "?")}
            for tid in ids
        ]
    return result


# --- Dispatch ---

def preview_tokenizer(
    node_type: str,
    properties: dict,
    dataset_type: str,
    sample_text: str | None = None,
) -> dict:
    """Dispatch a tokenizer preview by node type."""
    if node_type == "ml.preprocessing.tokenizer_char":
        return preview_char_vocab(
            dataset_type=dataset_type,
            lowercase=properties.get("lowercase", False),
            sample_text=sample_text,
        )
    if node_type == "ml.preprocessing.tokenizer_word":
        return preview_word_vocab(
            dataset_type=dataset_type,
            vocab_size=int(properties.get("vocabSize", 10000)),
            lowercase=properties.get("lowercase", True),
            split_punctuation=properties.get("splitPunctuation", True),
            sample_text=sample_text,
        )
    if node_type == "ml.preprocessing.tokenizer_bpe":
        return preview_bpe_vocab(
            dataset_type=dataset_type,
            vocab_size=int(properties.get("vocabSize", 10000)),
            lowercase=properties.get("lowercase", True),
            end_of_word_marker=properties.get("endOfWordMarker", "</w>"),
            sample_text=sample_text,
        )
    return {"error": f"Unknown tokenizer type: {node_type}"}
