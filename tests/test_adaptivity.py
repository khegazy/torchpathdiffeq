"""Tests for adaptive step adding (refinement) and removal (pruning)."""

from __future__ import annotations

import pytest
import torch
from _helpers import (
    ATOL_LOOSE,
    ATOL_MED,
    RTOL_LOOSE,
    RTOL_MED,
    T_FINAL,
    T_INIT,
    assert_step_continuity,
    assert_time_ordering,
)

from torchpathdiffeq import RKParallelUniformAdaptiveStepsizeSolver


def _integrand(t):
    """Test integrand: modulated Gaussian × cosine chirp."""
    return torch.exp(-5 * (t - 0.5) ** 2) * 4 * torch.cos(3 * t**2)


# ---------------------------------------------------------------------------
# Step Adding (refinement) tests
# ---------------------------------------------------------------------------

ADDING_METHODS = ["adaptive_heun", "dopri5"]


@pytest.mark.parametrize("method_name", ADDING_METHODS)
class TestStepAdding:
    """Starting from a coarse mesh, verify the solver adds steps to meet tolerance."""

    def _make_solver(self, method_name):
        return RKParallelUniformAdaptiveStepsizeSolver(
            method=method_name,
            ode_fxn=_integrand,
            atol=ATOL_MED,
            rtol=RTOL_MED,
        )

    def test_steps_added_on_coarse_mesh(self, method_name):
        """A minimal mesh (Cm1+1 points) should grow after integration."""
        solver = self._make_solver(method_name)
        t = torch.linspace(0, 1.0, solver.Cm1 + 1).unsqueeze(1)
        output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
        assert len(t) < len(output.t_optimal), (
            f"{method_name}: coarse mesh ({len(t)} points) should produce "
            f"a larger optimal mesh, but got {len(output.t_optimal)} points"
        )

    def test_time_ordering_after_refinement(self, method_name):
        """Time points remain ordered after adaptive refinement."""
        solver = self._make_solver(method_name)
        t = torch.linspace(0, 1.0, solver.Cm1 + 1).unsqueeze(1)
        output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
        assert_time_ordering(output)

    def test_step_continuity_after_refinement(self, method_name):
        """Consecutive steps share boundaries after adaptive refinement."""
        solver = self._make_solver(method_name)
        t = torch.linspace(0, 1.0, solver.Cm1 + 1).unsqueeze(1)
        output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
        assert_step_continuity(output)

    def test_mesh_stabilizes_after_repeated_refinement(self, method_name):
        """Feeding t_optimal back in should stabilize (mesh stops growing)."""
        solver = self._make_solver(method_name)
        t = torch.linspace(0, 1.0, solver.Cm1 + 1).unsqueeze(1)
        for idx in range(3):
            output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
            t_optimal = output.t_optimal
            if idx == 0:
                # First iteration: mesh must grow
                assert len(t) < len(
                    t_optimal
                ), f"Iteration {idx}: mesh should grow from {len(t)} points"
            else:
                # Subsequent iterations: mesh should not shrink
                assert len(t) <= len(
                    t_optimal
                ), f"Iteration {idx}: mesh should not shrink from {len(t)} points"
            t = t_optimal


# ---------------------------------------------------------------------------
# Step Removal (pruning) tests
# ---------------------------------------------------------------------------


class TestStepRemoval:
    """Starting from an over-resolved mesh, verify the solver prunes excess steps."""

    def _make_solver(self):
        return RKParallelUniformAdaptiveStepsizeSolver(
            method="dopri5",
            ode_fxn=_integrand,
            atol=ATOL_LOOSE,
            rtol=RTOL_LOOSE,
        )

    def test_steps_removed_on_dense_mesh(self):
        """A very dense mesh (997 points) should be pruned after integration."""
        solver = self._make_solver()
        t = torch.linspace(0, 1, 997).unsqueeze(1)
        output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
        assert len(t) > len(output.t_optimal), (
            f"Dense mesh ({len(t)} points) should be pruned, "
            f"but got {len(output.t_optimal)} points"
        )

    def test_time_ordering_after_pruning(self):
        """Time points remain ordered after pruning."""
        solver = self._make_solver()
        t = torch.linspace(0, 1, 997).unsqueeze(1)
        output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
        assert_time_ordering(output)

    def test_step_continuity_after_pruning(self):
        """Consecutive steps share boundaries after pruning."""
        solver = self._make_solver()
        t = torch.linspace(0, 1, 997).unsqueeze(1)
        output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
        assert_step_continuity(output)

    def test_mesh_stabilizes_after_repeated_pruning(self):
        """Feeding t_optimal back in should stabilize (pruning converges)."""
        solver = self._make_solver()
        t = torch.linspace(0, 1, 997).unsqueeze(1)
        for idx in range(3):
            output = solver.integrate(t=t, t_init=T_INIT, t_final=T_FINAL)
            t_optimal = output.t_optimal
            if idx == 0:
                # First iteration: mesh must shrink
                assert len(t) > len(
                    t_optimal
                ), f"Iteration {idx}: mesh should shrink from {len(t)} points"
            else:
                # Subsequent iterations: mesh should not grow
                assert len(t) >= len(
                    t_optimal
                ), f"Iteration {idx}: mesh should not grow from {len(t)} points"
            t = t_optimal
