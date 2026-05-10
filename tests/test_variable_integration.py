"""Integration accuracy tests for variable-sampling methods.

CLAUDE.md notes (line 69) that "Variable sampling integration tests
are not currently enabled (only uniform methods are tested)".
Investigation while writing this file revealed the deeper reason:
``_VariableAdaptiveQuadratureBase`` is missing the
``_t_step_interpolate`` method (uniform has it at parallel_solver.py
line 1443; variable has only a docstring-stubbed ``_initial_t_steps``
in lines 1620-1671). Calling ``integrate(...)`` on the variable
solver immediately ``AttributeError``s at parallel_solver.py:1193.

This means the variable code path is non-functional in current code.
The user's research in ``examples/pode/`` does NOT use variable
sampling (only imports ``VARIABLE_METHODS`` for reference), so the
breakage hasn't surfaced. The Phase 4 ``QuadratureMethod`` ABC
redesign should re-implement the variable solver path; when it
does, the strict-xfails in this file will become xpasses and
require unmarking (pytest enforces this).

Phase 0 of the quadrature alignment plan.
"""

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
    VARIABLE_METHOD_NAMES,
    assert_optimal_mesh_ordering,
    assert_step_continuity,
    assert_time_ordering,
)

from torchpathdiffeq import ODE_dict, adaptive_quadrature, steps


def _make_variable_solver(method_name, atol=ATOL_TIGHT, rtol=RTOL_TIGHT):
    """Construct a parallel variable-sampling solver."""
    return adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_VARIABLE,
        method=method_name,
        atol=atol,
        rtol=rtol,
        remove_cut=0.1,
    )


VARIABLE_BROKEN_REASON = (
    "Variable solver is currently non-functional: "
    "_VariableAdaptiveQuadratureBase is missing _t_step_interpolate. "
    "Phase 4 of the quadrature alignment plan restores the variable code "
    "path as part of the QuadratureMethod ABC redesign."
)


@pytest.mark.xfail(reason=VARIABLE_BROKEN_REASON, strict=True)
@pytest.mark.parametrize("method_name", VARIABLE_METHOD_NAMES)
@pytest.mark.parametrize("integrand_name", INTEGRAND_NAMES)
class TestVariableIntegralAccuracy:
    """Each variable method correctly integrates each ODE_dict integrand."""

    def _integrate(self, method_name, integrand_name):
        """Run integration; return (output, correct, cutoff)."""
        f, solution_fxn, cutoff = ODE_dict[integrand_name]
        correct = solution_fxn(mesh_init=T_INIT, mesh_final=T_FINAL)
        torch.manual_seed(SEED)
        solver = _make_variable_solver(method_name)
        output = solver.integrate(f, mesh_init=T_INIT, mesh_final=T_FINAL)
        return output, correct, cutoff

    def test_integral_value(self, method_name, integrand_name):
        """Computed integral matches the analytical solution within cutoff."""
        output, correct, cutoff = self._integrate(method_name, integrand_name)
        rel_error = torch.abs((output.integral.cpu() - correct) / correct)
        # Variable methods are 2nd or 3rd order so we relax the cutoff
        # by 100x relative to the uniform-method cutoff in ODE_dict.
        # The uniform cutoff was tuned for dopri5 (5th order); variable
        # methods cannot match that on the same atol/rtol setting.
        adjusted_cutoff = cutoff * 100.0
        assert rel_error < adjusted_cutoff, (
            f"{method_name} (variable) on {integrand_name}: "
            f"got {output.integral.item()}, expected {correct.item()}, "
            f"rel_error={rel_error.item():.2e} >= adjusted_cutoff={adjusted_cutoff:.2e}"
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
