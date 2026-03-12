"""Tests using the Wolf-Schlegel potential energy surface (multi-dimensional integrand)."""
from __future__ import annotations

import pytest
import torch

from torchpathdiffeq import (
    SerialAdaptiveStepsizeSolver,
    get_parallel_RK_solver,
    steps,
    UNIFORM_METHODS,
)

from _helpers import ATOL_LOOSE, RTOL_LOOSE, SEED, UNIFORM_METHOD_NAMES, REMOVE_CUT, T_INIT, T_FINAL

# ---------------------------------------------------------------------------
# Wolf-Schlegel potential energy surface along a linear interpolation path
# ---------------------------------------------------------------------------

_WS_MIN_INIT = torch.tensor([1.133, -1.486])
_WS_MIN_FINAL = torch.tensor([-1.166, 1.477])


def _wolf_schlegel(t, y=None):
    """Evaluate the Wolf-Schlegel 2D potential along a linear path in [0, 1]."""
    assert torch.all(t >= 0) and torch.all(t <= 1)
    while len(t.shape) < 2:
        t = t.unsqueeze(0)
    interpolate = _WS_MIN_INIT.to(t.device) + t * (_WS_MIN_FINAL - _WS_MIN_INIT).to(t.device)
    x = interpolate[:, 0].unsqueeze(-1)
    y = interpolate[:, 1].unsqueeze(-1)
    return 10 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + x * y + 0.2 * x + 0.1 * y)


class _WolfSchlegelCallable:
    """Callable wrapper for the serial solver.

    Note: torchdiffeq's adaptive stepper may evaluate slightly outside [0, 1]
    during trial steps, so we do NOT assert t ∈ [0, 1] here (unlike the
    parallel version which controls evaluation points directly).
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, t, y=None):
        while len(t.shape) < 2:
            t = t.unsqueeze(0)
        interpolate = _WS_MIN_INIT.to(t.device) + t * (_WS_MIN_FINAL - _WS_MIN_INIT).to(t.device)
        x = interpolate[:, 0].unsqueeze(-1)
        y = interpolate[:, 1].unsqueeze(-1)
        self.calls += 1
        return 10 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + x * y + 0.2 * x + 0.1 * y)


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
        ode_fxn=_wolf_schlegel,
    )

    serial_method = _SERIAL_METHOD_MAP.get(method_name, method_name)
    wf_instance = _WolfSchlegelCallable()
    serial_solver = SerialAdaptiveStepsizeSolver(
        serial_method, ATOL_LOOSE, RTOL_LOOSE, ode_fxn=wf_instance,
    )

    parallel_output = parallel_solver.integrate(t_init=T_INIT, t_final=T_FINAL)
    serial_output = serial_solver.integrate(t_init=T_INIT, t_final=T_FINAL)

    error = torch.abs(parallel_output.integral - serial_output.integral)
    # Scale tolerance by the number of steps (each step contributes up to atol + rtol*|y|)
    error_tolerance = (ATOL_LOOSE + RTOL_LOOSE * torch.abs(serial_output.integral)) * len(
        parallel_output.t
    )
    assert error < error_tolerance, (
        f"{method_name}: parallel={parallel_output.integral.item():.6f}, "
        f"serial={serial_output.integral.item():.6f}, "
        f"error={error.item():.2e}, tolerance={error_tolerance.item():.2e}"
    )
