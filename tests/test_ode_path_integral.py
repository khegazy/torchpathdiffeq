"""Tests that the ode_path_integral() wrapper produces identical results to direct solver usage."""

from __future__ import annotations

import pytest
import torch
from _helpers import ATOL_MED, RTOL_MED, SEED, T_FINAL, T_INIT, UNIFORM_METHOD_NAMES

from torchpathdiffeq import (
    RKParallelUniformAdaptiveStepsizeSolver,
    SerialAdaptiveStepsizeSolver,
    ode_path_integral,
)


def _integrand(t, y=0):
    """Test integrand: modulated Gaussian * cosine chirp."""
    return torch.exp(-5 * (t - 0.5) ** 2) * 4 * torch.cos(3 * t**2)


@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
def test_wrapper_matches_direct_solver_uniform(method_name):
    """ode_path_integral(computation='parallel', sampling='uniform') matches direct RK solver."""
    torch.manual_seed(SEED)

    wrapper_output = ode_path_integral(
        ode_fxn=_integrand,
        method=method_name,
        computation="parallel",
        sampling="uniform",
        atol=ATOL_MED,
        rtol=RTOL_MED,
        t_init=T_INIT,
        t_final=T_FINAL,
        y0=torch.tensor([0], dtype=torch.float64),
        t=None,
    )

    torch.manual_seed(SEED)

    direct_solver = RKParallelUniformAdaptiveStepsizeSolver(
        method=method_name,
        atol=ATOL_MED,
        rtol=RTOL_MED,
        ode_fxn=_integrand,
    )
    direct_output = direct_solver.integrate(t_init=T_INIT, t_final=T_FINAL)

    assert torch.allclose(wrapper_output.integral, direct_output.integral), (
        f"Integral mismatch for {method_name}: "
        f"wrapper={wrapper_output.integral.item()}, direct={direct_output.integral.item()}"
    )
    assert torch.allclose(
        wrapper_output.integral_error, direct_output.integral_error
    ), f"Integral error mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.t_optimal, direct_output.t_optimal
    ), f"Optimal mesh mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.y, direct_output.y
    ), f"y values mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.t, direct_output.t
    ), f"t values mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.h, direct_output.h
    ), f"Step sizes mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.sum_steps, direct_output.sum_steps
    ), f"sum_steps mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.sum_step_errors, direct_output.sum_step_errors
    ), f"sum_step_errors mismatch for {method_name}"
    assert torch.allclose(
        wrapper_output.error_ratios, direct_output.error_ratios
    ), f"error_ratios mismatch for {method_name}"


SERIAL_METHODS = ["adaptive_heun", "fehlberg2", "bosh3", "rk4", "dopri5"]


@pytest.mark.parametrize("method_name", SERIAL_METHODS)
def test_wrapper_matches_direct_solver_serial(method_name):
    """ode_path_integral(computation='serial') matches direct SerialAdaptiveStepsizeSolver."""
    wrapper_output = ode_path_integral(
        ode_fxn=_integrand,
        method=method_name,
        computation="serial",
        atol=ATOL_MED,
        rtol=RTOL_MED,
        t_init=T_INIT,
        t_final=T_FINAL,
        y0=torch.tensor([0], dtype=torch.float64),
        t=None,
    )

    direct_solver = SerialAdaptiveStepsizeSolver(
        method=method_name,
        atol=ATOL_MED,
        rtol=RTOL_MED,
        ode_fxn=_integrand,
    )
    direct_output = direct_solver.integrate(t_init=T_INIT, t_final=T_FINAL)

    assert torch.allclose(wrapper_output.integral, direct_output.integral), (
        f"Serial integral mismatch for {method_name}: "
        f"wrapper={wrapper_output.integral.item()}, direct={direct_output.integral.item()}"
    )
    assert torch.allclose(
        wrapper_output.t, direct_output.t
    ), f"Serial t mismatch for {method_name}"
