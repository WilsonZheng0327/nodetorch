# bpe.py — Byte-Pair Encoding tokenizer, learned from training data.
#
# Standard BPE algorithm:
#   1. Start with character-level tokens
#   2. Count all adjacent token pairs across the corpus
#   3. Merge the most frequent pair into a new token
#   4. Repeat until vocab_size is reached
#
# After training, the merge rules are applied in order to encode new text.
# Results are cached per (cache_key, vocab_size) so BPE is only learned once.

from __future__ import annotations
from collections import Counter
import re
import torch


class BPETokenizer:
    """Byte-Pair Encoding tokenizer that learns subword units from text."""

    def __init__(self, lowercase: bool = True, end_of_word_marker: str = "</w>"):
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}
        # Per-token corpus frequency, populated during training.
        self.token_freqs: dict[str, int] = {}
        self.lowercase = lowercase
        # Empty string disables the marker.
        self.end_of_word_marker = end_of_word_marker or ""

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _split_words(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(r'[a-zA-Z]+|[0-9]+|[^\s]', text)

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        if self.end_of_word_marker:
            return tuple(word) + (self.end_of_word_marker,)
        return tuple(word)

    def train(self, text: str, vocab_size: int, on_progress=None):
        """Learn BPE merge rules from a text corpus.

        Args:
            text: raw training text
            vocab_size: target vocabulary size (including <pad> and <unk>)
            on_progress: optional callback(current_vocab, target_vocab)
        """
        # Tokenize into words, track frequency of each word (as character tuple)
        words = self._split_words(text)
        word_freqs: dict[tuple[str, ...], int] = Counter()
        for word in words:
            word_freqs[self._word_to_symbols(word)] += 1

        # Initial vocab: special tokens + all unique characters
        self.vocab = {'<pad>': 0, '<unk>': 1}
        chars: set[str] = set()
        for word_tuple in word_freqs:
            for ch in word_tuple:
                chars.add(ch)
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # Iteratively merge the most frequent adjacent pair
        self.merges = []
        while len(self.vocab) < vocab_size:
            # Count all adjacent pairs
            pairs: Counter = Counter()
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pairs[(word_tuple[i], word_tuple[i + 1])] += freq

            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

            # Apply this merge to all words
            new_freqs: dict[tuple[str, ...], int] = {}
            for word_tuple, freq in word_freqs.items():
                new_word: list[str] = []
                i = 0
                while i < len(word_tuple):
                    if (i < len(word_tuple) - 1
                            and word_tuple[i] == best_pair[0]
                            and word_tuple[i + 1] == best_pair[1]):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_freqs[tuple(new_word)] = freq
            word_freqs = new_freqs

            if on_progress and len(self.vocab) % 100 == 0:
                on_progress(len(self.vocab), vocab_size)

        # Final per-token corpus frequencies (using the post-merge tokenization).
        self.token_freqs = {}
        for word_tuple, freq in word_freqs.items():
            for sym in word_tuple:
                self.token_freqs[sym] = self.token_freqs.get(sym, 0) + freq

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        """Encode text to BPE token IDs."""
        unk_id = self.vocab.get('<unk>', 1)
        words = self._split_words(text)
        ids: list[int] = []
        for word in words:
            symbols = list(self._word_to_symbols(word))
            # Apply merges in learned order
            for a, b in self.merges:
                merged = a + b
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols[i] = merged
                        del symbols[i + 1]
                    else:
                        i += 1
            ids.extend(self.vocab.get(s, unk_id) for s in symbols)
        if max_len is not None:
            ids = ids[:max_len]
            ids += [0] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        inv = {v: k for k, v in self.vocab.items()}
        tokens = [inv.get(i, '') for i in ids if i > 1]  # skip pad and unk
        marker = self.end_of_word_marker
        joined = ''.join(tokens)
        if marker:
            joined = joined.replace(marker, ' ')
        return joined.strip()

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID, preserving word boundaries as spaces."""
        inv = {v: k for k, v in self.vocab.items()}
        raw = inv.get(token_id, '')
        if not raw or token_id <= 1:
            return ''
        marker = self.end_of_word_marker
        if marker:
            return raw.replace(marker, ' ')
        return raw


# --- Cached instances ---
# BPE training is expensive, so we cache per (cache_key, vocab_size).

_bpe_cache: dict[str, BPETokenizer] = {}


def get_bpe_tokenizer(
    raw_text: str,
    vocab_size: int,
    cache_key: str,
    on_progress=None,
    lowercase: bool = True,
    end_of_word_marker: str = "</w>",
) -> BPETokenizer:
    """Get or train a cached BPE tokenizer.

    Args:
        raw_text: corpus text to learn from (only used on first call)
        vocab_size: target vocab size
        cache_key: unique key for caching (e.g. dataset name)
        on_progress: optional callback(current, target) during training
        lowercase: lowercase the corpus before training
        end_of_word_marker: token appended to each word; "" disables it
    """
    marker_key = end_of_word_marker or "none"
    lc_key = "lc" if lowercase else "cs"
    full_key = f"{cache_key}_{vocab_size}_{lc_key}_{marker_key}"
    if full_key in _bpe_cache:
        return _bpe_cache[full_key]

    tokenizer = BPETokenizer(lowercase=lowercase, end_of_word_marker=end_of_word_marker)
    tokenizer.train(raw_text, vocab_size, on_progress)
    _bpe_cache[full_key] = tokenizer
    return tokenizer
