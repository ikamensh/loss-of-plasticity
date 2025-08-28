"""Utilities for sampling classes with drifting weights.

The module exposes :class:`DriftingClassSampler`, a minimal helper for creating
non‑stationary training streams.  Each class starts with weight ``1`` and, every
time a sample is requested, all weights are perturbed by small Gaussian noise
and clamped to ``[min_weight, max_weight]``.  The perturbed weights are then
renormalized to probabilities and a class index is drawn accordingly.  Because
the weights evolve after every draw, the class distribution drifts gradually
over time.

The ``main`` section at the bottom demonstrates the behaviour by printing class
counts after many draws.
"""

from __future__ import annotations

import torch
from typing import List, Sequence


class DriftingClassSampler:
    """Sample class indices whose probabilities perform a random walk.

    Instantiate with the number of classes.  Repeated calls to
    :meth:`sample_class` or :meth:`sample_indices` will emit class choices whose
    relative probabilities evolve slowly because the internal weights undergo a
    random walk.

    Parameters
    ----------
    num_classes:
        Number of distinct classes to draw.
    step:
        Standard deviation of the Gaussian noise added to each weight per draw.
    min_weight, max_weight:
        After drifting, weights are clamped to this interval.
    """

    def __init__(
        self,
        num_classes: int,
        step: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 1.0,
    ) -> None:
        self.step = step
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights = torch.ones(num_classes)

    def drift(self) -> None:
        """Perform a small random walk and clamp the weights."""
        noise = torch.randn(self.weights.shape) * self.step
        self.weights.add_(noise).clamp_(self.min_weight, self.max_weight)

    def sample_class(self) -> int:
        """Drift the weights and return a single class index."""
        self.drift()
        probs = self.weights / self.weights.sum()
        return int(torch.multinomial(probs, 1))

    def sample_indices(
        self, class_indices: Sequence[torch.Tensor], batch_size: int
    ) -> List[int]:
        """Return ``batch_size`` dataset indices drawn with drifting class weights.

        Parameters
        ----------
        class_indices:
            A sequence where ``class_indices[c]`` contains dataset indices for
            class ``c``.
        batch_size:
            Number of samples to draw.
        """
        batch = []
        for _ in range(batch_size):
            # Each draw mutates ``self.weights`` so we must sample sequentially
            # rather than vectorising the operation.
            c = self.sample_class()
            idx_tensor = class_indices[c]
            choice = torch.randint(len(idx_tensor), (1,))
            batch.append(int(idx_tensor[choice]))
        return batch


if __name__ == "__main__":  # pragma: no cover - example usage
    # Demonstrate that the sampler produces a non-uniform, drifting stream.
    torch.manual_seed(0)
    sampler = DriftingClassSampler(num_classes=10)
    class_indices = [torch.arange(c * 5, (c + 1) * 5) for c in range(10)]  # dummy indices
    counts = torch.zeros(10, dtype=torch.int64)
    for _ in range(1000):
        idx = sampler.sample_indices(class_indices, 1)[0]
        counts[idx // 5] += 1  # recover the class from the dummy index
    print("Sampled counts per class:", counts.tolist())
