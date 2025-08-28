import os
import sys

import torch

# Allow tests to import the package without installation.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.drifting_sampler import DriftingClassSampler


def test_sampling_respects_fixed_weights():
    """Sampler draws classes in proportion to static weights.

    With ``step=0`` the weights do not drift.  Class ``0`` is given a weight of
    ``0.8`` while the others share ``0.2``.  Over 1000 draws the probability of
    observing fewer than 700 occurrences of class ``0`` is ~1e-69, so the test
    effectively never fails due to randomness.
    """
    torch.manual_seed(0)
    sampler = DriftingClassSampler(num_classes=3, step=0.0)
    sampler.weights = torch.tensor([0.8, 0.1, 0.1])
    counts = torch.zeros(3)
    for _ in range(1000):
        counts[sampler.sample_class()] += 1
    assert counts[0] / counts.sum() > 0.7


def test_drift_clamps_weights():
    """Weights remain within bounds despite aggressive random walk.

    A large step size makes the random walk prone to overshooting the bounds,
    so clamping must be applied.  The check is deterministic once a seed is
    fixed and cannot fail spuriously.
    """
    torch.manual_seed(0)
    sampler = DriftingClassSampler(num_classes=2, step=1.0)
    for _ in range(100):
        sampler.drift()
        assert torch.all((sampler.min_weight <= sampler.weights) & (sampler.weights <= sampler.max_weight))


def test_sample_indices_uses_provided_classes():
    """``sample_indices`` draws only from the class selected by ``weights``.

    Setting the weight of class ``1`` to ``1`` and class ``0`` to ``0`` ensures
    every sample should come from the second list of indices.
    """
    torch.manual_seed(0)
    sampler = DriftingClassSampler(num_classes=2, step=0.0)
    sampler.weights = torch.tensor([0.0, 1.0])  # always pick class 1
    class_indices = [torch.tensor([0, 1]), torch.tensor([2, 3])]
    idxs = sampler.sample_indices(class_indices, batch_size=10)
    assert all(i in {2, 3} for i in idxs)
