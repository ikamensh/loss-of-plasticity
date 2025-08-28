"""Tests guarding against accidental network bloat.

The CIFAR-ready deep network balances accuracy with speed by keeping its
parameter count low. This regression test ensures the architecture stays
lightweight so CPU-only experiments do not become unexpectedly slow."""

import os
import sys

# Allow importing the script without installing the package.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from drift.dynamic_mnist_cbp import DATASETS
from drift.models import DeepNet


def test_deep_net_param_count_under_limit():
    """``DeepNet`` should stay below 100k parameters for fast CPU training."""
    config = DATASETS["cifar10"]
    model = DeepNet(config=config, use_cbp=False)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params < 100_000

