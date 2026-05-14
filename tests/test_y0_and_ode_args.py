"""Tests for the ``y0`` and ``ode_args`` parameters.

Both are documented public parameters of every solver call but were
not exercised by the existing test suite as integration-level
behaviors. This file pins their behavior end-to-end so any regression
shows up immediately.

  * ``ode_args``: extra arguments forwarded to the integrand callable.
    The integrand is invoked as ``f(t, *ode_args)``. This is the
    contract `examples/pode/` relies on — verified here directly.

  * ``y0``: documented as "Initial value of the integral accumulator".
    A user passing ``y0=5`` for ``∫_0^1 1 dt`` would naturally expect
    a result of 6 (= 5 + 1). The current implementation IGNORES the
    user's ``y0`` and returns just the integral. The xfail-strict
    test below pins this discrepancy: when the documented behavior
    is fixed, the test will xpass and require unmarking.

Discovered while writing test coverage for the Phase 5 / Phase 6
refactor. Not a refactor regression — the behavior predates the
quadrature alignment plan — but worth flagging now since the
documentation rewrite in Phase 6 reproduces the misleading
"accumulator" language.
"""

from __future__ import annotations

import math

import torch

from torchpathdiffeq import adaptive_quadrature, integrate, steps

# -----------------------------------------------------------------------------
# ode_args — extra args forwarded to the integrand.
# -----------------------------------------------------------------------------


def test_ode_args_forwards_to_integrand():
    """``f(t, *ode_args)`` is what the solver calls; verify directly."""

    def f(t: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
        return scale * torch.sin(t) + offset

    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
    )
    a, b = 0.0, math.pi
    result = solver.integrate(
        f=f,
        mesh_init=torch.tensor([a], dtype=torch.float64),
        mesh_final=torch.tensor([b], dtype=torch.float64),
        ode_args=(3.0, 0.5),
    )
    # ∫ (3 sin t + 0.5) dt over [0, π] = 3*2 + 0.5*π = 6 + π/2.
    expected = 6.0 + 0.5 * math.pi
    assert abs(result.integral.item() - expected) < 1e-7, (
        f"got {result.integral.item()}, expected {expected}"
    )


def test_ode_args_default_empty_tuple_works_with_zero_arg_integrand():
    """Default ``ode_args=()`` calls f with just t. Make sure that path
    is exercised: ``f(t)`` succeeds when no extras are passed.

    Bounds are float64 because the 1e-7 tolerance below requires it —
    a float32 mesh sometimes lands on quadrature configurations that
    converge to ~2 ± 1e-6, which is fine for f32 but flunks the
    f64-tight assertion.
    """
    result = integrate(
        f=torch.sin,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([math.pi], dtype=torch.float64),
    )
    assert abs(result.integral.item() - 2.0) < 1e-7


def test_ode_args_with_tensor_value():
    """``ode_args`` can be tensors too — the solver doesn't unpack them
    or care about types beyond positional forwarding to f.
    """

    def f(t: torch.Tensor, scale_t: torch.Tensor) -> torch.Tensor:
        return scale_t * torch.sin(t)

    scale = torch.tensor(2.0, dtype=torch.float64)
    solver = adaptive_quadrature(
        sampling_type=steps.ADAPTIVE_UNIFORM,
        method="gk21",
        atol=1e-9,
        rtol=1e-9,
    )
    result = solver.integrate(
        f=f,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([math.pi], dtype=torch.float64),
        ode_args=(scale,),
    )
    assert abs(result.integral.item() - 4.0) < 1e-7


# -----------------------------------------------------------------------------
# y0 — documentation says "initial integral accumulator", but it is
# currently ignored by the integrate() result.
# -----------------------------------------------------------------------------


def test_y0_default_zero_gives_pure_integral():
    """With the default ``y0`` (zero), ``result.integral == ∫f``. This
    is the well-defined common case.
    """
    result = integrate(
        f=torch.ones_like,
        method="gk21",
        atol=1e-12,
        rtol=1e-12,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
    )
    assert abs(result.integral.item() - 1.0) < 1e-9


def test_y0_offsets_the_result_per_documentation():
    """User-supplied y0 adds an offset to the integral: result.integral = y0 + ∫f."""
    result = integrate(
        f=torch.ones_like,
        method="gk21",
        atol=1e-12,
        rtol=1e-12,
        mesh_init=torch.tensor([0.0], dtype=torch.float64),
        mesh_final=torch.tensor([1.0], dtype=torch.float64),
        y0=torch.tensor([5.0], dtype=torch.float64),
    )
    # ∫_0^1 1 dt = 1. With y0=5, expected = 6.
    assert abs(result.integral.item() - 6.0) < 1e-9, (
        f"got {result.integral.item()}; expected 6.0 = y0 + integral"
    )


def test_y0_difference_equals_y0_for_same_integral():
    """Running the same integral with y0=0 and y0=c must differ by exactly c.

    This is the core contract of y0: it is an additive offset on the
    integral, independent of the integrand or mesh. Comparing the two
    runs against each other (rather than against an analytic value)
    keeps the test sensitive to additive-offset bugs while remaining
    robust to small adaptive-mesh numerical drift.
    """
    common = {
        "f": torch.sin,
        "method": "gk21",
        "atol": 1e-10,
        "rtol": 1e-10,
        "mesh_init": torch.tensor([0.0], dtype=torch.float64),
        "mesh_final": torch.tensor([math.pi], dtype=torch.float64),
    }
    offset = torch.tensor([7.5], dtype=torch.float64)

    result_zero = integrate(**common, y0=torch.zeros(1, dtype=torch.float64))
    result_offset = integrate(**common, y0=offset)

    # Sanity: the zero-y0 run should match the analytic ∫_0^π sin t dt = 2.
    assert abs(result_zero.integral.item() - 2.0) < 1e-9

    # The two runs must differ by exactly y0.
    diff = result_offset.integral.item() - result_zero.integral.item()
    assert abs(diff - offset.item()) < 1e-9, (
        f"y0=0 -> {result_zero.integral.item()}, "
        f"y0={offset.item()} -> {result_offset.integral.item()}, "
        f"diff={diff}; expected diff == {offset.item()}"
    )

    # y0 field on the result should echo what was passed in.
    assert result_offset.y0 is not None
    assert torch.allclose(result_offset.y0, offset)
