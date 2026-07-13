"""The batch loaders (used for inference / node previews) must reuse a cached
Dataset instead of rebuilding it from disk on every call, while still drawing a
fresh (random) batch each time. Issue #2."""

import torch
import torchvision

import dataprep.data_loaders as dl


def test_cached_dataset_memoizes_by_key():
    dl._DATASET_CACHE.clear()
    calls = []

    def factory():
        calls.append(1)
        return object()

    a = dl._cached_dataset(("k",), factory)
    b = dl._cached_dataset(("k",), factory)
    assert a is b and len(calls) == 1          # built once, reused
    c = dl._cached_dataset(("k2",), factory)
    assert c is not a and len(calls) == 2      # different key -> rebuilt
    dl._DATASET_CACHE.clear()


def test_load_mnist_builds_dataset_once(monkeypatch):
    """Two load_mnist calls should construct the underlying dataset only once."""
    constructions = []

    class FakeMNIST(torch.utils.data.Dataset):
        def __init__(self, *a, **k):
            constructions.append(1)

        def __len__(self):
            return 16

        def __getitem__(self, i):
            return torch.zeros(1, 28, 28), i % 10

    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMNIST)
    dl._DATASET_CACHE.clear()
    try:
        b1 = dl.load_mnist({"batchSize": 4})
        b2 = dl.load_mnist({"batchSize": 4})
        assert len(constructions) == 1                     # not rebuilt per call
        assert b1["out"].shape[0] == 4 and b2["out"].shape[0] == 4  # batches still returned
    finally:
        dl._DATASET_CACHE.clear()                          # don't leak the fake to other tests
