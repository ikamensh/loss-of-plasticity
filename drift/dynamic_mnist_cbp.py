"""Train a tiny ConvNet on a class-skewed MNIST stream with optional CBP layers.

Instead of permuting pixels, this variant drifts the *class distribution* over
time. Each digit maintains a sampling weight that performs a small random walk
between 0.01 and 1.0. For every training example we first perturb all weights,
then draw a class in proportion to those weights and sample an image of that
class. The result is a continually changing, nonstationary stream where some
digits become temporarily rare while others dominate.

Vanilla networks tend to over-specialize toward classes that appear early in
training. When Continuous Backprop (``--cbp``) is enabled, filters can be
replaced over time, helping the model adapt as the class proportions drift and
yielding more stable accuracy under this shift.
"""

import argparse
import os
import sys
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


class SimpleNet(nn.Module):
    """Minimal network with one convolutional and one linear layer.

    When ``use_cbp`` is True, CBPConv/CBPLinear wrappers are inserted to enable
    continual feature replacement; otherwise the plain layers are used.
    """

    def __init__(self, use_cbp: bool):
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Conv2d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        # After conv (28->24) and pooling (24->12)
        self.fc1 = nn.Linear(16 * 12 * 12, 32)
        self.fc2 = nn.Linear(32, 10)
        self.act = nn.ReLU()

        if use_cbp:
            self.cbp_conv = CBPConv(
                in_layer=self.conv,
                out_layer=self.fc1,
                num_last_filter_outputs=12 * 12,
                replacement_rate=1e-4,
                maturity_threshold=100,
            )
            self.cbp_fc = CBPLinear(
                in_layer=self.fc1,
                out_layer=self.fc2,
                replacement_rate=1e-4,
                maturity_threshold=100,
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


def get_data(batch_size: int = 64):
    """Return the MNIST train dataset and a test loader."""
    transform = transforms.ToTensor()
    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test_loader


def fetch_batch(train: datasets.MNIST, idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binarised images and labels for ``idxs`` in one shot.

    The original training loop iterated over indices and called
    ``MNIST.__getitem__`` for each element. That repeated Python and transform
    overhead dominated runtime according to profiling. Gathering the raw pixel
    data and targets in bulk lets PyTorch handle the heavy lifting in C and
    eliminates the per-item overhead.
    """
    x = train.data[idxs].float().div(255.0).unsqueeze(1)
    x = torch.bernoulli(x)
    y = train.targets[idxs].clone()
    return x, y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on binarized MNIST images."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = torch.bernoulli(x)  # binarize at evaluation time
            x, y = x.to(device), y.to(device)
            preds = model(x)
            correct += (preds.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbp", action="store_true", help="Enable CBP layers")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on the training loop and print top stats",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    train_set, test_loader = get_data(batch_size)
    model = SimpleNet(use_cbp=args.cbp).to(device)
    opt = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    # Pre-compute indices for each class to allow sampling with replacement.
    class_indices = [torch.where(train_set.targets == i)[0] for i in range(10)]
    sampler = DriftingClassSampler(num_classes=10)

    steps_per_epoch = len(train_set) // batch_size  # roughly one pass worth of samples
    prev_conv_resets, prev_fc_resets = 0, 0

    if args.profile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    for epoch in range(1, args.epochs + 1):
        # Track how many samples of each class were seen this epoch.
        epoch_counts = torch.zeros(10, dtype=torch.int64)
        for _ in range(steps_per_epoch):
            batch_indices = torch.tensor(
                sampler.sample_indices(class_indices, batch_size)
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

    if args.profile:
        import pstats

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(10)


if __name__ == "__main__":
    main()
