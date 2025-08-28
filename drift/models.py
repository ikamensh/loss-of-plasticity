"""Neural network architectures for the drift experiments.

This module contains both the original shallow network used in the
project's early days and a slightly deeper variant that performs better
on CIFAR-10 while remaining lightweight.  Both networks optionally use
Continuous Backprop layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lop.algos.cbp_conv import CBPConv
from lop.algos.cbp_linear import CBPLinear

# ---------------------------------------------------------------------------
# Shared CBP hyperparameters
# ---------------------------------------------------------------------------

CBP_REPLACEMENT_RATE = 1e-4
CBP_MATURITY_THRESHOLD = 100

# ---------------------------------------------------------------------------
# Shallow network
# ---------------------------------------------------------------------------

# Dimensions mirror the very first version of the training script: one
# ``16@5×5`` convolution feeding a ``32``-unit hidden layer.
SHALLOW_CONV_OUT = 16
SHALLOW_CONV_KERNEL = 5
SHALLOW_POOL_KERNEL = 2
SHALLOW_POOL_STRIDE = 2
SHALLOW_HIDDEN = 32


class ShallowNet(nn.Module):
    """Tiny network with one convolution and one fully connected layer."""

    def __init__(self, config, use_cbp: bool):
        super().__init__()
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(
            config.in_channels, SHALLOW_CONV_OUT, kernel_size=SHALLOW_CONV_KERNEL
        )
        self.pool = nn.MaxPool2d(SHALLOW_POOL_KERNEL, SHALLOW_POOL_STRIDE)

        # Determine flattened dimension and outputs per filter for CBP.
        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.image_size, config.image_size)
            pooled = self.pool(self.act(self.conv(dummy)))
            flattened_dim = int(pooled.view(1, -1).size(1))
            last_filter_outputs = int(pooled[0, 0].numel())

        self.fc1 = nn.Linear(flattened_dim, SHALLOW_HIDDEN)
        self.fc2 = nn.Linear(SHALLOW_HIDDEN, 10)

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


# ---------------------------------------------------------------------------
# Deeper network
# ---------------------------------------------------------------------------

# Expanded dimensions for better CIFAR-10 performance while keeping the
# parameter count below ``100k``.
CONV1_OUT_CHANNELS = 16
CONV1_KERNEL_SIZE = 5
CONV2_OUT_CHANNELS = 32
CONV2_KERNEL_SIZE = 3
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 32


class DeepNet(nn.Module):
    """Two conv blocks followed by two hidden layers."""

    def __init__(self, config, use_cbp: bool):
        super().__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(
            config.in_channels, CONV1_OUT_CHANNELS, kernel_size=CONV1_KERNEL_SIZE
        )
        self.conv2 = nn.Conv2d(
            CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=CONV2_KERNEL_SIZE
        )
        self.pool = nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.image_size, config.image_size)
            h = self.pool(self.act(self.conv1(dummy)))
            h = self.pool(self.act(self.conv2(h)))
            flattened_dim = int(h.view(1, -1).size(1))
            last_filter_outputs = int(h[0, 0].numel())

        self.fc1 = nn.Linear(flattened_dim, HIDDEN_DIM1)
        self.fc2 = nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        self.fc3 = nn.Linear(HIDDEN_DIM2, 10)

        if use_cbp:
            self.cbp_conv = CBPConv(
                in_layer=self.conv2,
                out_layer=self.fc1,
                num_last_filter_outputs=last_filter_outputs,
                replacement_rate=CBP_REPLACEMENT_RATE,
                maturity_threshold=CBP_MATURITY_THRESHOLD,
            )
            self.cbp_fc = CBPLinear(
                in_layer=self.fc2,
                out_layer=self.fc3,
                replacement_rate=CBP_REPLACEMENT_RATE,
                maturity_threshold=CBP_MATURITY_THRESHOLD,
            )
        else:
            self.cbp_conv = None
            self.cbp_fc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        if self.cbp_conv is not None:
            x = self.cbp_conv(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        if self.cbp_fc is not None:
            x = self.cbp_fc(x)
        x = self.fc3(x)
        return x


# Convenience map used by the training script.
MODELS = {
    "shallow": ShallowNet,
    "deep": DeepNet,
}

