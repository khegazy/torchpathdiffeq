"""Tests using the Wolf-Schlegel potential energy surface (multi-dimensional integrand)."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_LOOSE,
    REMOVE_CUT,
    RTOL_LOOSE,
    SEED,
    T_FINAL,
    T_INIT,
    UNIFORM_METHOD_NAMES,
)

from torchpathdiffeq import (
    SerialAdaptiveStepsizeSolver,
    get_parallel_RK_solver,
    steps,
    wolf_schlegel,
)

# Map variable method names to their serial equivalents
_SERIAL_METHOD_MAP = {"generic3": "bosh3"}


@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
def test_wolf_schlegel_parallel_vs_serial(method_name):
    """Parallel integration of Wolf-Schlegel matches serial within tolerance."""
    torch.manual_seed(SEED)
    parallel_solver = get_parallel_RK_solver(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method=method_name,
        atol=ATOL_LOOSE,
        rtol=RTOL_LOOSE,
        remove_cut=REMOVE_CUT,
        ode_fxn=wolf_schlegel,
    )

    serial_method = _SERIAL_METHOD_MAP.get(method_name, method_name)
    serial_solver = SerialAdaptiveStepsizeSolver(
        serial_method,
        ATOL_LOOSE,
        RTOL_LOOSE,
        ode_fxn=wolf_schlegel,
    )

    parallel_output = parallel_solver.integrate(t_init=T_INIT, t_final=T_FINAL)
    serial_output = serial_solver.integrate(t_init=T_INIT, t_final=T_FINAL)

    error = torch.abs(parallel_output.integral - serial_output.integral)
    # Scale tolerance by the number of steps (each step contributes up to atol + rtol*|y|)
    error_tolerance = (
        ATOL_LOOSE + RTOL_LOOSE * torch.abs(serial_output.integral)
    ) * len(parallel_output.t)
    assert error < error_tolerance, (
        f"{method_name}: parallel={parallel_output.integral.item():.6f}, "
        f"serial={serial_output.integral.item():.6f}, "
        f"error={error.item():.2e}, tolerance={error_tolerance.item():.2e}"
    )
