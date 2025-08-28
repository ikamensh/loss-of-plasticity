"""Train a tiny ConvNet on a class-skewed image stream with optional CBP layers.

Instead of permuting pixels, this variant drifts the *class distribution* over
time. Each class maintains a sampling weight that performs a small random walk
between 0.01 and 1.0. For every training example we first perturb all weights,
then draw a class in proportion to those weights and sample an image of that
class. The result is a continually changing, nonstationary stream where some
classes become temporarily rare while others dominate.

Vanilla networks tend to over-specialize toward classes that appear early in
training. When Continuous Backprop (``--cbp``) is enabled, filters can be
replaced over time, helping the model adapt as the class proportions drift and
yielding more stable accuracy under this shift.  Both MNIST and CIFAR-10 are
supported via ``--dataset``.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow running the script directly without installing the package.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))


from lop.algos.cbp_conv import CBPConv
from lop.algos.cbp_linear import CBPLinear
from drift.drifting_sampler import DriftingClassSampler
import time
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = "data"
BATCH_SIZE = 64
LEARNING_RATE = 0.1
NUM_CLASSES = 10

CONV_OUT_CHANNELS = 16
CONV_KERNEL_SIZE = 5
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
HIDDEN_DIM = 32

CBP_REPLACEMENT_RATE = 1e-4
CBP_MATURITY_THRESHOLD = 100


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration parameters for each supported dataset."""

    dataset_cls: Type[datasets.VisionDataset]
    in_channels: int
    image_size: int


DATASETS = {
    "mnist": DatasetConfig(datasets.MNIST, 1, 28),
    "cifar10": DatasetConfig(datasets.CIFAR10, 3, 32),
}


class SimpleNet(nn.Module):
    """Minimal network with one convolutional and one linear layer.

    When ``use_cbp`` is True, CBPConv/CBPLinear wrappers are inserted to enable
    continual feature replacement; otherwise the plain layers are used.  Layer
    shapes are inferred from the provided :class:`DatasetConfig` to avoid magic
    numbers.
    """

    def __init__(self, config: DatasetConfig, use_cbp: bool):
        super().__init__()
        self.act = nn.ReLU()

        # Convolutional feature extractor.
        self.conv = nn.Conv2d(
            config.in_channels, CONV_OUT_CHANNELS, kernel_size=CONV_KERNEL_SIZE
        )
        self.pool = nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        # Infer the flattened feature dimension and the number of outputs per
        # convolutional filter for CBP from a dummy forward pass.  This keeps
        # the model agnostic to input resolution.
        with torch.no_grad():
            dummy = torch.zeros(
                1, config.in_channels, config.image_size, config.image_size
            )
            pooled = self.pool(self.act(self.conv(dummy)))
            flattened_dim = int(pooled.view(1, -1).size(1))
            last_filter_outputs = int(pooled[0, 0].numel())

        self.fc1 = nn.Linear(flattened_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

        if use_cbp:
            self.cbp_conv = CBPConv(
                in_layer=self.conv,
                out_layer=self.fc1,
                num_last_filter_outputs=last_filter_outputs,
                replacement_rate=CBP_REPLACEMENT_RATE,
                maturity_threshold=CBP_MATURITY_THRESHOLD,
            )
            self.cbp_fc = CBPLinear(
                in_layer=self.fc1,
                out_layer=self.fc2,
                replacement_rate=CBP_REPLACEMENT_RATE,
                maturity_threshold=CBP_MATURITY_THRESHOLD,
            )
        else:
            self.cbp_conv = None
            self.cbp_fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv(x)))
        if self.cbp_conv is not None:
            x = self.cbp_conv(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        if self.cbp_fc is not None:
            x = self.cbp_fc(x)
        x = self.fc2(x)
        return x


def get_data(dataset: str, batch_size: int = BATCH_SIZE):
    """Return the train dataset, test loader and configuration for ``dataset``."""

    config = DATASETS[dataset]
    transform = transforms.ToTensor()
    train = config.dataset_cls(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    test = config.dataset_cls(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test_loader, config


def fetch_batch(train: datasets.MNIST, idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalised images and labels for ``idxs`` in one shot.

    ``fetch_batch`` replaces the original per-item loop over indices, removing
    Python and ``ToTensor`` overhead by pulling data directly from the dataset's
    storage.  The helper understands both MNIST's ``N×H×W`` tensor layout and
    CIFAR's ``N×H×W×C`` NumPy arrays, returning tensors scaled to ``[0, 1]`` and
    permuted into channels-first format when necessary.
    """

    # ``MNIST`` stores data as a ``torch.Tensor`` whereas ``CIFAR10`` uses a
    # ``numpy.ndarray``.  Index accordingly and scale pixel values to ``[0, 1]``.
    if isinstance(train.data, torch.Tensor):
        raw = train.data[idxs].float().div(255.0)
    else:
        raw = torch.tensor(train.data[idxs.tolist()], dtype=torch.float32).div(255.0)

    if raw.ndim == 3:  # Grayscale images: N×H×W
        # Add a channel dimension for consistency with convolutional layers.
        x = raw.unsqueeze(1)
    elif raw.ndim == 4:  # Colour images: N×H×W×C
        # Rearrange to channels-first expected by PyTorch modules.
        x = raw.permute(0, 3, 1, 2)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported data shape {raw.shape}")

    # Targets may be a tensor (MNIST) or a Python list (CIFAR).
    if isinstance(train.targets, torch.Tensor):
        y = train.targets[idxs].clone()
    else:
        y = torch.tensor(train.targets)[idxs].clone()

    return x, y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on the given dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cbp", action="store_true", default=True, help="Enable CBP layers")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        default="mnist",
        help="Dataset to use",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on the training loop and print top stats",
    )
    parser.add_argument("--dynamic", action="store_true", default=True, help="Enable distribution shift")

    args = parser.parse_args()
    print(f"{args=}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_loader, cfg = get_data(args.dataset, BATCH_SIZE)
    model = SimpleNet(config=cfg, use_cbp=args.cbp).to(device)
    opt = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Pre-compute indices for each class to allow sampling with replacement.
    targets = (
        torch.tensor(train_set.targets)
        if isinstance(train_set.targets, list)
        else train_set.targets
    )
    class_indices = [torch.where(targets == i)[0] for i in range(NUM_CLASSES)]
    if args.dynamic:
        sampler = DriftingClassSampler(num_classes=NUM_CLASSES)
    else:
        sampler = DriftingClassSampler(num_classes=NUM_CLASSES, min_weight=1., max_weight=1.)

    steps_per_epoch = len(train_set) // BATCH_SIZE  # roughly one pass worth of samples
    prev_conv_resets, prev_fc_resets = 0, 0

    if args.profile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Track how many samples of each class were seen this epoch.
        epoch_counts = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        for _ in range(steps_per_epoch):
            batch_indices = torch.tensor(
                # ``DriftingClassSampler`` expects a concrete batch size.  The
                # script uses a fixed ``BATCH_SIZE`` constant, so pass it
                # directly rather than an undefined variable.
                sampler.sample_indices(class_indices, BATCH_SIZE)
            )
            x, y = fetch_batch(train_set, batch_indices)
            x = x.to(device)
            epoch_counts += torch.bincount(y, minlength=10)
            y = y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

        acc = evaluate(model, test_loader, device)
        history.append(acc)
        print(f"Epoch {epoch}: test accuracy {acc:.3f}")
        # Report current sampler weights and how many samples were drawn per class.
        print(f"  class weights: {[f"{x:.2}" for x in sampler.weights.tolist()]}")
        total_samples = int(epoch_counts.sum())
        rel_counts = [f"{x:.2%}" for x in (epoch_counts.float() / total_samples).tolist()]
        print(f"  samples per class: {epoch_counts.tolist()} (rel {rel_counts})")

        # Report how many features were reset by CBP layers this epoch.
        conv_total = model.cbp_conv.num_feature_resets if model.cbp_conv else 0
        fc_total = model.cbp_fc.num_feature_resets if model.cbp_fc else 0
        print(
            "  feature resets this epoch - conv: "
            f"{conv_total - prev_conv_resets}, dense: {fc_total - prev_fc_resets}"
        )
        prev_conv_resets, prev_fc_resets = conv_total, fc_total
        epoch_end = time.time()
        print(f"  Epoch time: {epoch_end-epoch_start:.2f}s")

    if args.profile:
        import pstats

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(10)

    end = time.time()
    print(f"Total time: {end-start:.2f}s")

    plt.plot(history)
    plt.show()
    plt.savefig("accuracy.png")


if __name__ == "__main__":
    main()
