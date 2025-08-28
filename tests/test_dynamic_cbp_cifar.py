"""Tests for dataset flexibility in the dynamic CBP script."""

import os
import sys

import pytest
import torch

# Allow importing the script without installing the package.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import after we add dataset support; for now this test will fail before fix.
from drift.dynamic_mnist_cbp import DATASETS, NUM_CLASSES, SimpleNet


def test_simple_net_handles_cifar():
    """Ensure SimpleNet can process CIFAR inputs once dataset option is added.

    Historically ``SimpleNet`` was built solely for MNIST's ``1×28×28`` images.
    Attempting to run it on CIFAR10 ``3×32×32`` inputs triggered shape mismatch
    errors in the convolution and linear layers.  The updated implementation
    infers dimensions from a dataset configuration, so this forward pass should
    succeed and produce logits for every class.
    """
    config = DATASETS["cifar10"]
    model = SimpleNet(config=config, use_cbp=False)
    # Two random CIFAR-sized images.
    x = torch.randn(2, config.in_channels, config.image_size, config.image_size)
    out = model(x)
    # The model should output one logit vector per image.
    assert out.shape == (2, NUM_CLASSES)
