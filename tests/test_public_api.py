"""Smoke test of the public API.

Phase 0 of the quadrature alignment plan. This test imports every
documented public symbol and exercises each at its simplest signature.
Its job is to catch accidental rename omissions during the Phase 3
public-API rename pass: if any export is dropped or any documented
field disappears, this file will fail.

This test pins the *current* (ODE-framed) public API. After Phase 3
the new (quadrature-framed) public API will replace these assertions
in this same file.
"""

from __future__ import annotations

import math

import torch

import torchpathdiffeq
from torchpathdiffeq import (
    UNIFORM_METHODS,
    VARIABLE_METHODS,
    IntegralOutput,
    ODE_dict,
    RKParallelUniformAdaptiveStepsizeSolver,
    RKParallelVariableAdaptiveStepsizeSolver,
    SerialAdaptiveStepsizeSolver,
    get_parallel_RK_solver,
    ode_path_integral,
    setup_logging,
    steps,
    wolf_schlegel,
)


class TestPublicExports:
    """Every documented public symbol exists at the package level."""

    def test_classes_exist(self):
        assert RKParallelUniformAdaptiveStepsizeSolver is not None
        assert RKParallelVariableAdaptiveStepsizeSolver is not None
        assert SerialAdaptiveStepsizeSolver is not None
        assert IntegralOutput is not None

    def test_functions_exist(self):
        assert callable(ode_path_integral)
        assert callable(get_parallel_RK_solver)
        assert callable(setup_logging)

    def test_data_exports_exist(self):
        assert isinstance(UNIFORM_METHODS, dict)
        assert len(UNIFORM_METHODS) > 0
        assert isinstance(VARIABLE_METHODS, dict)
        assert len(VARIABLE_METHODS) > 0
        assert isinstance(ODE_dict, dict)
        assert len(ODE_dict) > 0
        assert callable(wolf_schlegel)

    def test_steps_enum(self):
        assert hasattr(steps, "ADAPTIVE_UNIFORM")
        assert hasattr(steps, "ADAPTIVE_VARIABLE")
        assert hasattr(steps, "FIXED")

    def test_module_has_dunder_attributes(self):
        assert hasattr(torchpathdiffeq, "__name__")


class TestOdePathIntegralEntryPoint:
    """The free function `ode_path_integral` works for the simplest case."""

    def test_simple_sine_integral(self):
        result = ode_path_integral(
            ode_fxn=torch.sin,
            method="dopri5",
            atol=1e-8,
            rtol=1e-6,
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([math.pi], dtype=torch.float64),
        )
        assert isinstance(result, IntegralOutput)
        # int_0^pi sin(t) dt = 2
        assert abs(result.integral.item() - 2.0) < 1e-6

    def test_constant_integral(self):
        # int_0^1 1 dt = 1
        result = ode_path_integral(
            ode_fxn=torch.ones_like,
            method="bosh3",
            atol=1e-10,
            rtol=1e-10,
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([1.0], dtype=torch.float64),
        )
        assert abs(result.integral.item() - 1.0) < 1e-9


class TestIntegralOutputFields:
    """The IntegralOutput dataclass exposes the documented fields.

    Phase 3 of the plan renames these to value/error/mesh_*; this test
    locks the current names so a rename omission is caught immediately.
    """

    def _run(self):
        return ode_path_integral(
            ode_fxn=torch.sin,
            method="dopri5",
            atol=1e-8,
            rtol=1e-6,
            t_init=torch.tensor([0.0], dtype=torch.float64),
            t_final=torch.tensor([math.pi], dtype=torch.float64),
        )

    def test_documented_fields_present(self):
        r = self._run()
        # Public-facing fields per current docs.
        for name in (
            "integral",
            "loss",
            "gradient_taken",
            "t_optimal",
            "t",
            "h",
            "y",
            "sum_steps",
            "integral_error",
            "sum_step_errors",
            "error_ratios",
            "t_init",
            "t_final",
            "y0",
        ):
            assert hasattr(r, name), f"IntegralOutput missing field {name!r}"


class TestSolverFactory:
    """The factory function dispatches correctly to uniform/variable solvers."""

    def test_uniform_dispatch(self):
        solver = get_parallel_RK_solver(
            sampling_type=steps.ADAPTIVE_UNIFORM,
            method="dopri5",
            atol=1e-6,
            rtol=1e-6,
        )
        assert isinstance(solver, RKParallelUniformAdaptiveStepsizeSolver)

    def test_variable_dispatch(self):
        solver = get_parallel_RK_solver(
            sampling_type=steps.ADAPTIVE_VARIABLE,
            method="generic3",
            atol=1e-6,
            rtol=1e-6,
        )
        assert isinstance(solver, RKParallelVariableAdaptiveStepsizeSolver)


class TestMethodRegistries:
    """The method registries contain the documented entries."""

    def test_uniform_methods_present(self):
        for name in ("adaptive_heun", "fehlberg2", "bosh3", "dopri5"):
            assert name in UNIFORM_METHODS, f"missing uniform method {name!r}"

    def test_variable_methods_present(self):
        for name in ("adaptive_heun", "generic3"):
            assert name in VARIABLE_METHODS, f"missing variable method {name!r}"
