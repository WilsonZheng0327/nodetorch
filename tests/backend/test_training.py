"""Tests for the training loop plugin system."""

import pytest
from training import TRAINING_LOOPS, detect_training_mode, run_training
from training.base import (
    TrainingContext, TrainingResult,
    build_optimizer, build_scheduler, init_weight_norms,
)
import torch
import torch.nn as nn


class TestRegistry:
    def test_standard_registered(self):
        assert "standard" in TRAINING_LOOPS

    def test_standard_is_callable(self):
        assert callable(TRAINING_LOOPS["standard"])

    def test_detect_standard(self):
        nodes = {"n1": {"type": "ml.layers.conv2d"}, "n2": {"type": "ml.loss.cross_entropy"}}
        assert detect_training_mode(nodes) == "standard"

    def test_detect_empty(self):
        assert detect_training_mode({}) == "standard"


class TestBuildOptimizer:
    def test_adam(self):
        params = [nn.Parameter(torch.randn(10))]
        opt_node = {"type": "ml.optimizers.adam", "properties": {"lr": 0.001}}
        opt = build_optimizer(opt_node, params)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw(self):
        params = [nn.Parameter(torch.randn(10))]
        opt_node = {"type": "ml.optimizers.adamw", "properties": {"lr": 0.001, "weightDecay": 0.01}}
        opt = build_optimizer(opt_node, params)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        params = [nn.Parameter(torch.randn(10))]
        opt_node = {"type": "ml.optimizers.sgd", "properties": {"lr": 0.01, "momentum": 0.9}}
        opt = build_optimizer(opt_node, params)
        assert isinstance(opt, torch.optim.SGD)


class TestBuildScheduler:
    def test_none(self):
        opt = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=0.01)
        assert build_scheduler(opt, "none", 10) is None

    def test_cosine(self):
        opt = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=0.01)
        sched = build_scheduler(opt, "cosine", 10)
        assert sched is not None

    def test_step(self):
        opt = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=0.01)
        sched = build_scheduler(opt, "step", 10)
        assert sched is not None

    def test_warmup(self):
        opt = torch.optim.SGD([nn.Parameter(torch.randn(10))], lr=0.01)
        sched = build_scheduler(opt, "warmup", 20)
        assert sched is not None


class TestInitWeightNorms:
    def test_returns_norms(self):
        modules = {"conv": nn.Conv2d(3, 16, 3)}
        norms = init_weight_norms(modules)
        assert "conv" in norms
        assert norms["conv"] > 0

    def test_empty_modules(self):
        norms = init_weight_norms({})
        assert norms == {}


class TestTrainingResult:
    def test_default_values(self):
        r = TrainingResult()
        assert r.epoch_results == []
        assert r.error is None
        assert r.training_mode == "standard"

    def test_error_result(self):
        r = TrainingResult(error="something broke")
        assert r.error == "something broke"
