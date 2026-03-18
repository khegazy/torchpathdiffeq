"""Unit tests for SerialAdaptiveStepsizeSolver."""

from __future__ import annotations

import torch

from torchpathdiffeq.base import IntegralOutput
from torchpathdiffeq.serial_solver import SerialAdaptiveStepsizeSolver


def _make_serial_solver(method="dopri5", atol=1e-8, rtol=1e-6):
    """Create a serial solver for testing."""
    return SerialAdaptiveStepsizeSolver(method=method, atol=atol, rtol=rtol)


class TestSerialSolverIntegrate:
    """Tests for SerialAdaptiveStepsizeSolver.integrate."""

    def test_constant_integrand(self):
        """∫₀¹ 1 dt = 1."""
        solver = _make_serial_solver()
        result = solver.integrate(
            ode_fxn=lambda t, y: torch.ones_like(y),
            y0=torch.tensor([0.0], dtype=torch.float64),
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([1.0], dtype=torch.float64),
        )
        assert torch.allclose(
            result.integral, torch.tensor([1.0], dtype=torch.float64), atol=1e-6
        )

    def test_linear_integrand(self):
        """∫₀¹ t dt = 0.5."""
        solver = _make_serial_solver()
        result = solver.integrate(
            ode_fxn=lambda t, y: t.unsqueeze(-1) if t.dim() == 0 else t,
            y0=torch.tensor([0.0], dtype=torch.float64),
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([1.0], dtype=torch.float64),
        )
        assert torch.allclose(
            result.integral, torch.tensor([0.5], dtype=torch.float64), atol=1e-6
        )

    def test_returns_integral_output(self):
        """Result is an IntegralOutput instance."""
        solver = _make_serial_solver()
        result = solver.integrate(
            ode_fxn=lambda t, y: torch.ones_like(y),
            y0=torch.tensor([0.0], dtype=torch.float64),
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([1.0], dtype=torch.float64),
        )
        assert isinstance(result, IntegralOutput)
        assert result.t is not None

    def test_custom_bounds(self):
        """∫₂⁵ 1 dt = 3."""
        solver = _make_serial_solver()
        result = solver.integrate(
            ode_fxn=lambda t, y: torch.ones_like(y),
            y0=torch.tensor([0.0], dtype=torch.float64),
            t_init=torch.tensor([2.0], dtype=torch.float64),
            t_final=torch.tensor([5.0], dtype=torch.float64),
        )
        assert torch.allclose(
            result.integral, torch.tensor([3.0], dtype=torch.float64), atol=1e-6
        )

    def test_nonzero_y0(self):
        """∫₀¹ 1 dt with y0=5 gives 6."""
        solver = _make_serial_solver()
        result = solver.integrate(
            ode_fxn=lambda t, y: torch.ones_like(y),
            y0=torch.tensor([5.0], dtype=torch.float64),
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([1.0], dtype=torch.float64),
        )
        assert torch.allclose(
            result.integral, torch.tensor([6.0], dtype=torch.float64), atol=1e-6
        )

    def test_set_solver_dtype_noop(self):
        """_set_solver_dtype does nothing (no error)."""
        solver = _make_serial_solver()
        solver._set_solver_dtype(torch.float32)
        solver._set_solver_dtype(torch.float64)
