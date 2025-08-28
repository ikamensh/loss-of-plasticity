"""Ensure both shallow and deep nets can be built from the training script.

The model architecture used by ``dynamic_mnist_cbp.py`` was previously
hard-coded, making it impossible to switch back to the original shallow
network once the deeper variant was introduced.  Experiments requiring
the smaller architecture could not be reproduced, constituting a
regression in configurability.

This test exercises a planned ``build_model`` helper that should return
either architecture based on a ``name`` argument.  It documents the
regression and will fail until the helper is implemented.
"""

import os
import sys
import pytest

# Allow importing the script without installing the package.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from drift.dynamic_mnist_cbp import DATASETS, build_model  # pragma: no cover - will fail until implemented
from drift import models  # pragma: no cover - will fail until implemented


def test_build_model_selects_architecture():
    """Requesting ``shallow`` or ``deep`` should yield the respective class.

    Prior to the fix this will raise ``AttributeError`` because the
    training script exposes only a single hard-coded model.
    """
    cfg = DATASETS["mnist"]
    shallow = build_model("shallow", cfg, use_cbp=False)
    deep = build_model("deep", cfg, use_cbp=False)
    assert shallow.__class__ is models.ShallowNet
    assert deep.__class__ is models.DeepNet
