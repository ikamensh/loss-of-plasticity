"""Tests that dynamic training script stores results in run-specific folders.

Historically the script wrote `accuracy.png` directly to the working directory,
meaning subsequent experiments overwrote previous plots and lost their arguments.
The new behaviour logs each run under `runs/DATE` with the CLI arguments and
accuracy plot, enabling batch experiment management.
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Allow importing the script without installing the package.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from drift import dynamic_mnist_cbp


@pytest.mark.usefixtures("tmp_path")
def test_saves_run_outputs(monkeypatch, tmp_path):
    """Ensure a run writes args and plot into a unique results directory.

    Prior to the fix the script always saved `accuracy.png` in the current
    directory.  This test verifies a regression by checking that after running
    the script once, a new folder under `runs/` appears containing both the
    arguments used for the run and the accuracy plot.
    """
    # Run in an isolated directory so test artifacts don't pollute the repo.
    monkeypatch.chdir(tmp_path)

    # Stub data loading with a tiny synthetic dataset to keep the test fast and
    # avoid network downloads.
    def fake_get_data(dataset: str, batch_size: int = 64):
        data = dynamic_mnist_cbp.torch.zeros((1, 1, 28, 28))
        targets = dynamic_mnist_cbp.torch.zeros(1, dtype=dynamic_mnist_cbp.torch.long)

        class Dummy:
            def __len__(self):
                return len(self.data)

        train = Dummy()
        train.data = data
        train.targets = targets
        test_loader = [(data, targets)]
        cfg = dynamic_mnist_cbp.DATASETS["mnist"]
        return train, test_loader, cfg

    monkeypatch.setattr(dynamic_mnist_cbp, "get_data", fake_get_data)
    monkeypatch.setattr(dynamic_mnist_cbp.plt, "show", lambda: None)

    # Execute script with zero epochs to skip the training loop.
    sys.argv = ["dynamic_mnist_cbp.py", "--epochs", "0"]
    dynamic_mnist_cbp.main()

    run_root = Path("runs")
    # The script should create exactly one timestamp-named directory.
    subdirs = [p for p in run_root.iterdir() if p.is_dir()]
    assert len(subdirs) == 1, "expected one run directory"
    run_dir = subdirs[0]

    # Both args.json and accuracy.png should be saved inside that directory.
    args_path = run_dir / "args.json"
    plot_path = run_dir / "accuracy.png"
    assert args_path.is_file()
    assert plot_path.is_file()

    # Args should round-trip from the JSON file.
    with args_path.open() as fh:
        saved_args = json.load(fh)
    assert saved_args["epochs"] == 0
