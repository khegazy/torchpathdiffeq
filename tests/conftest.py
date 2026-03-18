from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure tests/ is on sys.path so _helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import SEED


@pytest.fixture()
def seed():
    """Set a deterministic random seed for reproducibility."""
    torch.manual_seed(SEED)
