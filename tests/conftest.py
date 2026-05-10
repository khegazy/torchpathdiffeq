from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure tests/ is on sys.path so _helpers can be imported
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import SEED

from torchpathdiffeq import UNIFORM_METHODS

# CLAUDE.md notes: "UNIFORM_METHODS are global singletons — tests that mutate
# dtype must save/restore original tableau tensors (float16 truncation is
# irreversible)." A few tests (test_data_types.py in particular) construct
# solvers at lower dtypes which mutates the shared singletons, leaving them
# in a precision-degraded state for the next test. Pin the original float64
# tableau values at session start and restore them before each test.
_ORIGINAL_TABLEAU_VALUES = {
    name: {
        "c": method.tableau.c.clone(),
        "b": method.tableau.b.clone(),
        "b_error": method.tableau.b_error.clone(),
    }
    for name, method in UNIFORM_METHODS.items()
}


@pytest.fixture(autouse=True)
def _restore_uniform_methods_dtype():
    """Reset UNIFORM_METHODS singleton tableaux to original float64 state
    before every test. Eliminates cross-test contamination from dtype
    mutation. Removable once Phase 4 eliminates the singletons.
    """
    for name, method in UNIFORM_METHODS.items():
        original = _ORIGINAL_TABLEAU_VALUES[name]
        method.tableau.c = original["c"].clone()
        method.tableau.b = original["b"].clone()
        method.tableau.b_error = original["b_error"].clone()


@pytest.fixture
def seed():
    """Set a deterministic random seed for reproducibility."""
    torch.manual_seed(SEED)
