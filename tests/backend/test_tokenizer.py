"""Tests for the Tokenizer preprocessing node (backend).

Tests the TokenizerModule from node_builders.py: builder function,
truncation, padding, and vocabulary clamping behavior.
"""

import torch
import torch.nn as nn
import pytest

from node_builders import NODE_BUILDERS, TokenizerModule


class TestTokenizerBuilder:
    def test_registered_in_builders(self):
        assert "ml.preprocessing.tokenizer" in NODE_BUILDERS

    def test_returns_module(self):
        props = {"vocabSize": 100, "maxLen": 32}
        module = NODE_BUILDERS["ml.preprocessing.tokenizer"](props, {})
        assert isinstance(module, nn.Module)

    def test_default_props(self):
        module = NODE_BUILDERS["ml.preprocessing.tokenizer"]({}, {})
        assert isinstance(module, TokenizerModule)
        assert module.vocab_size == 10000
        assert module.max_len == 256


class TestTokenizerTruncation:
    def test_truncates_long_sequences(self):
        tok = TokenizerModule(vocab_size=100, max_len=5)
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # length 10
        out = tok(x)
        assert list(out.shape) == [1, 5]
        assert out[0].tolist() == [1, 2, 3, 4, 5]

    def test_exact_length_unchanged(self):
        tok = TokenizerModule(vocab_size=100, max_len=5)
        x = torch.tensor([[10, 20, 30, 40, 50]])  # exactly max_len
        out = tok(x)
        assert list(out.shape) == [1, 5]
        assert out[0].tolist() == [10, 20, 30, 40, 50]


class TestTokenizerPadding:
    def test_pads_short_sequences(self):
        tok = TokenizerModule(vocab_size=100, max_len=8)
        x = torch.tensor([[5, 10, 15]])  # length 3
        out = tok(x)
        assert list(out.shape) == [1, 8]
        assert out[0].tolist() == [5, 10, 15, 0, 0, 0, 0, 0]

    def test_pads_with_zeros(self):
        tok = TokenizerModule(vocab_size=100, max_len=6)
        x = torch.tensor([[1]])  # length 1
        out = tok(x)
        assert out[0][0].item() == 1
        assert all(v == 0 for v in out[0][1:].tolist())


class TestTokenizerVocabClamping:
    def test_clamps_oov_to_unk(self):
        tok = TokenizerModule(vocab_size=50, max_len=5)
        x = torch.tensor([[10, 49, 50, 99, 200]])
        out = tok(x)
        # 10 and 49 are valid (< 50), 50/99/200 should become 1 (<unk>)
        assert out[0][0].item() == 10
        assert out[0][1].item() == 49
        assert out[0][2].item() == 1
        assert out[0][3].item() == 1
        assert out[0][4].item() == 1

    def test_zero_index_preserved(self):
        """Index 0 (<pad>) should pass through unchanged."""
        tok = TokenizerModule(vocab_size=50, max_len=3)
        x = torch.tensor([[0, 1, 2]])
        out = tok(x)
        assert out[0].tolist() == [0, 1, 2]


class TestTokenizerBatched:
    def test_handles_batch(self):
        tok = TokenizerModule(vocab_size=100, max_len=4)
        x = torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [10, 20, 30, 40, 50, 60],
        ])
        out = tok(x)
        assert list(out.shape) == [2, 4]
        assert out[0].tolist() == [1, 2, 3, 4]
        assert out[1].tolist() == [10, 20, 30, 40]

    def test_batch_padding(self):
        tok = TokenizerModule(vocab_size=100, max_len=6)
        x = torch.tensor([
            [1, 2],
            [3, 4],
        ])
        out = tok(x)
        assert list(out.shape) == [2, 6]
        assert out[0].tolist() == [1, 2, 0, 0, 0, 0]
        assert out[1].tolist() == [3, 4, 0, 0, 0, 0]


class TestTokenizerCombined:
    def test_truncate_and_clamp(self):
        """Sequences that are both too long AND have OOV tokens."""
        tok = TokenizerModule(vocab_size=10, max_len=3)
        x = torch.tensor([[5, 15, 3, 99, 7]])  # too long + OOV
        out = tok(x)
        assert list(out.shape) == [1, 3]
        assert out[0][0].item() == 5   # valid
        assert out[0][1].item() == 1   # 15 >= 10 → <unk>
        assert out[0][2].item() == 3   # valid

    def test_pad_and_clamp(self):
        """Short sequences with OOV tokens."""
        tok = TokenizerModule(vocab_size=10, max_len=5)
        x = torch.tensor([[5, 20]])  # short + OOV
        out = tok(x)
        assert list(out.shape) == [1, 5]
        assert out[0].tolist() == [5, 1, 0, 0, 0]  # 20→1, rest padded
