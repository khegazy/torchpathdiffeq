"""Shared constants and helpers for the test suite."""

from __future__ import annotations

import torch
from torch import nn

from torchpathdiffeq import (
    UNIFORM_METHODS,
    VARIABLE_METHODS,
    ODE_dict,
    get_parallel_RK_solver,
    steps,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 2025

# Tight tolerances for accuracy tests
ATOL_TIGHT = 1e-12
RTOL_TIGHT = 1e-10

# Loose tolerances for adaptivity / dtype tests
ATOL_LOOSE = 1e-5
RTOL_LOOSE = 1e-5

# Medium tolerances
ATOL_MED = 1e-9
RTOL_MED = 1e-7

T_INIT = torch.tensor([0], dtype=torch.float64)
T_FINAL = torch.tensor([1], dtype=torch.float64)

REMOVE_CUT = 0.1

# ---------------------------------------------------------------------------
# Parametrize helpers — usable as @pytest.mark.parametrize values
# ---------------------------------------------------------------------------

UNIFORM_METHOD_NAMES = list(UNIFORM_METHODS.keys())
VARIABLE_METHOD_NAMES = list(VARIABLE_METHODS.keys())
INTEGRAND_NAMES = list(ODE_dict.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_uniform_solver(method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT, **kwargs):
    """Create a parallel uniform-sampling RK solver."""
    return get_parallel_RK_solver(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
        **kwargs,
    )


def make_solver_for_unit_test(method_name="bosh3", atol=1e-6, rtol=1e-6):
    """Create a minimal solver for testing internal methods (no ode_fxn needed)."""
    return get_parallel_RK_solver(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
    )


def make_variable_solver_for_unit_test(
    method_name="adaptive_heun", atol=1e-6, rtol=1e-6
):
    """Create a minimal variable solver for testing internal methods."""
    return get_parallel_RK_solver(
        sampling_type=steps.ADAPTIVE_VARIABLE,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=REMOVE_CUT,
    )


def constant_ode_fxn(t, *args):
    """f(t) = 1 for all t. Returns shape [N, 1]."""
    if len(t.shape) == 1:
        return torch.ones(1, dtype=t.dtype, device=t.device)
    return torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)


def assert_time_ordering(integral_output):
    """Assert that all time points in the output are non-decreasing."""
    t_flat = torch.flatten(integral_output.t, start_dim=0, end_dim=1)
    assert torch.all(
        t_flat[1:] - t_flat[:-1] >= 0
    ), "Time points are not non-decreasing"


def assert_optimal_mesh_ordering(integral_output):
    """Assert that the optimal mesh time points are non-decreasing."""
    t_optimal_flat = torch.flatten(integral_output.t_optimal, start_dim=0, end_dim=1)
    assert torch.all(
        t_optimal_flat[1:] - t_optimal_flat[:-1] >= 0
    ), "Optimal mesh time points are not non-decreasing"


def assert_step_continuity(integral_output):
    """Assert that consecutive steps share boundary points (end of step i == start of step i+1)."""
    assert torch.allclose(
        integral_output.t[1:, 0, :], integral_output.t[:-1, -1, :]
    ), "Consecutive steps do not share boundary points"


# ---------------------------------------------------------------------------
# Parameterized nn.Module integrands for gradient tests
# ---------------------------------------------------------------------------


class ScaledIntegrand(nn.Module):
    """f(t) = scale * t^2, with learnable scale. Returns [N, 1]."""

    __name__ = "ScaledIntegrand"

    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=torch.float64))

    def forward(self, t, *args):
        while len(t.shape) < 2:
            t = t.unsqueeze(0)
        return self.scale * t**2
