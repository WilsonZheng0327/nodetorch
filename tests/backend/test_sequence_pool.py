"""Masked sequence pooling: pooling must ignore right-padding when a mask
(token ids, pad=0) is supplied, and behave exactly as before when it isn't."""

import torch

from engine.node_builders import SequencePoolModule
from engine.forward_utils import execute_node

# Two sequences, T=4, H=2. Row 0 has 2 real tokens then padding; the pad
# positions carry deliberately huge values (100) so any leak is obvious.
X = torch.tensor([
    [[1.0, 1.0], [2.0, 2.0], [100.0, 100.0], [100.0, 100.0]],  # 2 real, 2 pad
    [[1.0, 0.0], [0.0, 3.0], [2.0, 2.0], [4.0, 1.0]],          # 4 real, 0 pad
])
# Mask as token ids: nonzero = real token, 0 = pad (right-padded).
MASK = torch.tensor([[5, 7, 0, 0],
                     [3, 4, 9, 1]])


def test_masked_mean_excludes_padding():
    out = SequencePoolModule('mean')(X, MASK)
    # row 0: mean of the two real steps only (not the 100s)
    assert torch.allclose(out[0], torch.tensor([1.5, 1.5]))
    # row 1: no padding -> mean of all four
    assert torch.allclose(out[1], X[1].mean(dim=0))


def test_masked_max_excludes_padding():
    out = SequencePoolModule('max')(X, MASK)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0]))   # not 100
    assert torch.allclose(out[1], X[1].max(dim=0).values)


def test_masked_last_picks_final_real_token():
    out = SequencePoolModule('last')(X, MASK)
    assert torch.allclose(out[0], X[0, 1])   # last real is index 1, not 3 (padding)
    assert torch.allclose(out[1], X[1, 3])   # full length -> genuine last


def test_no_mask_is_unmasked_original_behavior():
    # Without a mask, every mode matches the old plain-tensor ops.
    assert torch.allclose(SequencePoolModule('mean')(X), X.mean(dim=1))
    assert torch.allclose(SequencePoolModule('max')(X), X.max(dim=1).values)
    assert torch.allclose(SequencePoolModule('last')(X), X[:, -1, :])


def test_mask_ignored_when_length_mismatches():
    # A mask whose T doesn't match x is ignored (falls back to unmasked).
    bad = torch.ones(2, 3)
    assert torch.allclose(SequencePoolModule('mean')(X, bad), X.mean(dim=1))


def test_execute_node_routes_mask():
    # The forward-pass dispatch must forward the optional "mask" input.
    m = SequencePoolModule('last')
    with_mask = execute_node('ml.structural.sequence_pool', m, {'in': X, 'mask': MASK})
    assert torch.allclose(with_mask['out'][0], X[0, 1])
    # And still works with no mask edge connected.
    without = execute_node('ml.structural.sequence_pool', m, {'in': X})
    assert torch.allclose(without['out'][0], X[0, -1])
