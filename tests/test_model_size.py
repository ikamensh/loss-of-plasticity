"""Tests guarding against accidental network bloat.

The CIFAR-ready ``SimpleNet`` was deepened to improve accuracy while remaining
small enough for quick CPU experimentation. This test ensures the architecture
stays lightweight so training scripts do not become unexpectedly slow."""

import os
import sys

# Allow importing the script without installing the package.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from drift.dynamic_mnist_cbp import DATASETS, SimpleNet


def test_simple_net_param_count_under_limit():
    """``SimpleNet`` should stay below 100k parameters for fast CPU training."""
    config = DATASETS["cifar10"]
    model = SimpleNet(config=config, use_cbp=False)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params < 100_000

