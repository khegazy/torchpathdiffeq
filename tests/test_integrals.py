"""Tests for numerical integral accuracy against known analytical solutions."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_TIGHT,
    INTEGRAND_NAMES,
    RTOL_TIGHT,
    SEED,
    T_FINAL,
    T_INIT,
    UNIFORM_METHOD_NAMES,
    assert_optimal_mesh_ordering,
    assert_step_continuity,
    assert_time_ordering,
    make_uniform_solver,
)

from torchpathdiffeq import ODE_dict


@pytest.mark.parametrize("method_name", UNIFORM_METHOD_NAMES)
@pytest.mark.parametrize("integrand_name", INTEGRAND_NAMES)
class TestUniformIntegralAccuracy:
    """Verify that each uniform RK method correctly integrates each test function."""

    def _integrate(self, method_name, integrand_name):
        """Run integration for the given method and integrand, return (output, correct, cutoff)."""
        ode_fxn, solution_fxn, cutoff = ODE_dict[integrand_name]
        correct = solution_fxn(t_init=T_INIT, t_final=T_FINAL)
        torch.manual_seed(SEED)
        solver = make_uniform_solver(method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT)
        output = solver.integrate(ode_fxn, t_init=T_INIT, t_final=T_FINAL)
        return output, correct, cutoff

    def test_integral_value(self, method_name, integrand_name):
        """Computed integral matches the analytical solution within the error cutoff."""
        output, correct, cutoff = self._integrate(method_name, integrand_name)
        rel_error = torch.abs((output.integral.cpu() - correct) / correct)
        assert rel_error < cutoff, (
            f"{method_name} failed on {integrand_name}: "
            f"got {output.integral.item()}, expected {correct.item()}, "
            f"rel_error={rel_error.item():.2e} >= cutoff={cutoff:.2e}"
        )

    def test_time_ordering(self, method_name, integrand_name):
        """All time points in the integration output are non-decreasing."""
        output, _, _ = self._integrate(method_name, integrand_name)
        assert_time_ordering(output)

    def test_optimal_mesh_ordering(self, method_name, integrand_name):
        """Optimal mesh time points are non-decreasing."""
        output, _, _ = self._integrate(method_name, integrand_name)
        assert_optimal_mesh_ordering(output)

    def test_step_continuity(self, method_name, integrand_name):
        """Consecutive integration steps share boundary points."""
        output, _, _ = self._integrate(method_name, integrand_name)
        assert_step_continuity(output)
