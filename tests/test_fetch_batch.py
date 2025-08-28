"""Tests for the ``fetch_batch`` helper across datasets."""

import os
import sys

import torch

# Allow importing the ``drift`` package without installing the repository.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from drift import dynamic_mnist_cbp


def test_fetch_batch_matches_mnist_per_item():
    """Vectorised MNIST batch retrieval mirrors per-item access without binarisation.

    A previous revision of :func:`fetch_batch` stochastically binarised MNIST
    pixels which made its output differ from the deterministic
    ``MNIST.__getitem__`` behaviour.  The randomness caused downstream tests to
    fail unpredictably and complicated debugging.  This regression test
    illustrates the issue by comparing the helper against a manual per-item
    loop, asserting they now return identical normalised tensors.
    """

    train, _, _ = dynamic_mnist_cbp.get_data("mnist")
    idxs = torch.randint(len(train), (8,))

    # Manual per-item retrieval replicating the old slow path.  ``ToTensor``
    # already scales pixels to ``[0, 1]`` so no further processing is required.
    manual_x, manual_y = [], []
    for i in idxs.tolist():
        x, y = train[i]
        manual_x.append(x)
        manual_y.append(y)
    manual_x = torch.stack(manual_x)
    manual_y = torch.tensor(manual_y)

    # Vectorised helper under test.  A fixed seed would be required for the
    # previous buggy implementation which binarised inputs; here it simply
    # normalises and should match ``manual_x`` exactly.
    torch.manual_seed(0)
    x, y = dynamic_mnist_cbp.fetch_batch(train, idxs)

    assert torch.equal(x, manual_x)
    assert torch.equal(y, manual_y)


def test_fetch_batch_matches_cifar_per_item():
    """Vectorised CIFAR batch retrieval mirrors per-item access.

    The original helper unsqueezed and binarised inputs assuming MNIST's
    ``1×28×28`` layout, which broke for CIFAR10's ``H×W×C`` NumPy arrays.  The
    updated version should simply normalise and permute channels, matching the
    manual per-item path that relies on ``ToTensor``.
    """

    train, _, _ = dynamic_mnist_cbp.get_data("cifar10")
    idxs = torch.randint(len(train), (4,))  # small batch for speed

    # Manual per-item retrieval replicating the slow but correct path.
    manual_x, manual_y = [], []
    for i in idxs.tolist():
        x, y = train[i]
        manual_x.append(x)
        manual_y.append(y)
    manual_x = torch.stack(manual_x)
    manual_y = torch.tensor(manual_y)

    # Vectorised helper under test.
    x, y = dynamic_mnist_cbp.fetch_batch(train, idxs)

    assert torch.equal(x, manual_x)
    assert torch.equal(y, manual_y)
