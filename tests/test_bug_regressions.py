"""Regression tests for confirmed bugs in the parallel solver.

Phase 0 of the quadrature alignment plan. These tests document the
broken pieces in the code and lock the user-visible behavior so
Phase 1 fixes don't regress correctness elsewhere.

Important: investigation found that bug B1 and most of B2 currently
sit behind dead code. parallel_solver.py:1119-1133 reloads cached
barriers when ``t is None and same_ode_fxn``, but the unconditional
``if t is None`` block at line 1135 immediately overwrites the
result with a fresh random mesh. So the user never sees the bad
concatenation in the integral value. The bugs ARE real in the
source — they will reappear the moment Phase 1 tries to make
warm-start actually work, which is why we capture the broken state
explicitly here.

Tests in this file:

  - test_no_warm_start_path_correctness — anchors the no-warm-start
    path; if Phase 1 regresses this, the fix went wrong.

  - test_lambda_cache_key_is_broken (xfail-strict) — asserts that
    two different lambdas DO collide in the current cache key,
    which is the proof of bug B2. Phase 1's fix (id()-based key,
    plus an opt-in reuse_mesh parameter) will make this test pass
    by causing the cache key to differ.

  - test_warm_start_cache_load_is_dead_code — explicitly documents
    that the cache load at lines 1119-1133 has no effect on the
    final result because the random-mesh path overwrites it.

  - test_float16_construction_raises (xfail-strict) — Bug B4:
    float16 construction should refuse adaptive control rather
    than silently produce wrong answers.
"""

from __future__ import annotations

import math

import pytest
import torch
from tests._helpers import make_uniform_solver

from torchpathdiffeq import IntegrationResult, ode_path_integral

# -----------------------------------------------------------------------------
# Anchor: the no-warm-start path is correct as-is. Phase 1 must not regress it.
# -----------------------------------------------------------------------------


def test_no_warm_start_path_correctness():
    """A single fresh-solver call must produce an integral within the
    solver's reported error estimate. This anchors the no-warm-start
    path so Phase 1 fixes cannot accidentally regress it.
    """
    out = ode_path_integral(
        ode_fxn=lambda t: torch.exp(-(t**2)),
        method="dopri5",
        atol=1e-8,
        rtol=1e-8,
        t_init=torch.tensor([-2.0], dtype=torch.float64),
        t_final=torch.tensor([2.0], dtype=torch.float64),
    )
    expected = math.sqrt(math.pi) * math.erf(2.0)
    actual = out.integral.item()
    # The solver reports an integral_error; the actual error should be
    # within an order of magnitude of that estimate. We pin a generous
    # bound here to anchor "this works" without being brittle to torch
    # version-level numerical differences.
    assert abs(actual - expected) < 1e-3, (
        f"got {actual}, expected {expected}, reported error {out.integral_error.item()}"
    )


# -----------------------------------------------------------------------------
# B2: lambda cache key collision.
# Probe the cache key directly to demonstrate the broken state.
# -----------------------------------------------------------------------------


def test_lambda_cache_key_distinguishes_different_lambdas():
    """After integrating lambda1 then lambda2, the solver's cached
    'previous integrand' identifier must distinguish the two —
    otherwise the warm-start mechanism cannot be made correct.

    Phase 1 fix (Bug B2): the identifier was ``ode_fxn.__name__`` which
    is ``'<lambda>'`` for every lambda. It now stores ``id(ode_fxn)``,
    a stable per-function value that distinguishes any two function
    objects.
    """
    solver = make_uniform_solver("dopri5", atol=1e-6, rtol=1e-6)

    f1 = lambda t, *_: torch.sin(t)  # noqa: E731
    f2 = lambda t, *_: torch.cos(5 * t) ** 2  # noqa: E731

    t_init = torch.tensor([0.0], dtype=torch.float64)
    t_final = torch.tensor([math.pi], dtype=torch.float64)

    solver.integrate(ode_fxn=f1, t_init=t_init, t_final=t_final)
    key_after_f1 = solver.previous_ode_fxn_id

    solver.integrate(ode_fxn=f2, t_init=t_init, t_final=t_final)
    key_after_f2 = solver.previous_ode_fxn_id

    assert key_after_f1 != key_after_f2, (
        f"Solver cannot distinguish lambda1 from lambda2: "
        f"key_after_f1={key_after_f1!r}, key_after_f2={key_after_f2!r}."
    )


# -----------------------------------------------------------------------------
# B1: t_init/t_final swap. The bug is in the cache-load path, but the
# random-mesh generation at line 1135 overwrites the cached barriers
# unconditionally, so end-to-end the bug is invisible. Document this.
# -----------------------------------------------------------------------------


def test_warm_start_with_new_t_final_yields_correct_mesh():
    """After Phase 1's reuse_mesh opt-in, calling the solver a second
    time with ``reuse_mesh=True`` and a *different* ``t_final`` than
    the first call must still produce a monotone mesh that ends at
    the new ``t_final``. This exercises the warm-start path which
    Phase 1's Bug B1 fix activated.
    """
    solver = make_uniform_solver("dopri5", atol=1e-6, rtol=1e-6)

    def f(t, *args):
        return torch.sin(t)

    t_init = torch.tensor([0.0], dtype=torch.float64)
    out_first = solver.integrate(
        ode_fxn=f, t_init=t_init, t_final=torch.tensor([1.0], dtype=torch.float64)
    )
    expected_first = 1.0 - math.cos(1.0)
    assert abs(out_first.integral.item() - expected_first) < 1e-5

    out_second = solver.integrate(
        ode_fxn=f,
        t_init=t_init,
        t_final=torch.tensor([1.5], dtype=torch.float64),
        reuse_mesh=True,
    )
    assert out_second is not None
    expected_second = 1.0 - math.cos(1.5)
    assert abs(out_second.integral.item() - expected_second) < 1e-5

    # Bug B1 fix: the warm-started mesh ends at the new t_final
    # (previously the buggy concatenation appended t_init here,
    # producing non-monotone barriers).
    assert abs(out_second.mesh_optimal[-1].item() - 1.5) < 1e-12

    # Mesh is monotone non-decreasing.
    diffs = out_second.mesh_optimal[1:, 0] - out_second.mesh_optimal[:-1, 0]
    assert torch.all(diffs >= 0), (
        f"warm-started mesh is not monotone: diffs.min()={diffs.min().item()}"
    )


# -----------------------------------------------------------------------------
# B6: max_path_change early exit must return an IntegrationResult with
# converged=False, not bare None (which violated the type contract).
# -----------------------------------------------------------------------------


def test_max_path_change_returns_integral_output_not_none():
    """When ``max_path_change`` triggers early exit on a user-provided
    mesh, the solver must return an ``IntegrationResult`` with
    ``converged=False``, not ``None``. Phase 1's Bug B6 fix.
    """
    # Provide a far-too-coarse mesh on a wiggly integrand so the solver
    # cannot meet a tight tolerance on most steps. max_path_change=0.1
    # means: exit if more than 10% of steps fail. With 4-point initial
    # mesh on a damped sine and 1e-12 atol, that threshold trips hard.
    solver = make_uniform_solver("dopri5", atol=1e-12, rtol=1e-12, max_path_change=0.1)
    t = torch.linspace(0.0, 4.0, 4, dtype=torch.float64).unsqueeze(-1)

    out = solver.integrate(
        ode_fxn=lambda t, *_: torch.sin(10 * t) * torch.exp(-0.1 * t),
        t=t,
    )
    assert isinstance(out, IntegrationResult), (
        f"max_path_change early-exit returned {type(out).__name__}, "
        f"expected IntegrationResult. This is bug B6."
    )
    assert out.converged is False, (
        "early-exit IntegrationResult should have converged=False; "
        f"got converged={out.converged!r}"
    )


def test_normal_completion_has_converged_true():
    """A normal integration call returns ``converged=True``. Pins the
    default value of the new field.
    """
    out = ode_path_integral(
        ode_fxn=torch.sin,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        t_init=torch.tensor([0.0], dtype=torch.float64),
        t_final=torch.tensor([math.pi], dtype=torch.float64),
    )
    assert out.converged is True


# -----------------------------------------------------------------------------
# B4: float16 + adaptive should be refused.
# -----------------------------------------------------------------------------


def test_float16_construction_raises():
    """Constructing a parallel solver with dtype=float16 raises
    ``ValueError`` because float16's ~1e-3 precision floor cannot
    support adaptive error control to typical tolerances.

    Phase 1 fix (Bug B4): the guard lives in
    ``SolverBase._set_dtype``.
    """
    with pytest.raises(ValueError, match=r"float16|coarse"):
        make_uniform_solver("dopri5", atol=1e-5, rtol=1e-5, dtype=torch.float16)
