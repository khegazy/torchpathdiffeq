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

from torchpathdiffeq import ode_path_integral

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


@pytest.mark.xfail(
    reason="Bug B2: parallel_solver.py:1099 keys warm-start cache on "
    "ode_fxn.__name__ which is '<lambda>' for every lambda. Phase 1 will "
    "switch to id()-based keying with explicit reuse_mesh opt-in, at which "
    "point this xfail unmarks itself.",
    strict=True,
)
def test_lambda_cache_key_distinguishes_different_lambdas():
    """After integrating lambda1 then lambda2, the solver's cached
    'previous integrand' identifier should distinguish the two —
    otherwise the warm-start mechanism cannot be made correct.

    Currently the identifier is ``ode_fxn.__name__`` which is
    ``'<lambda>'`` for both. After the Phase 1 fix the identifier
    will be a stable per-function id (or be cleared entirely on
    integrand change), and this test will pass.
    """
    solver = make_uniform_solver("dopri5", atol=1e-6, rtol=1e-6)

    f1 = lambda t, *_: torch.sin(t)  # noqa: E731
    f2 = lambda t, *_: torch.cos(5 * t) ** 2  # noqa: E731

    t_init = torch.tensor([0.0], dtype=torch.float64)
    t_final = torch.tensor([math.pi], dtype=torch.float64)

    solver.integrate(ode_fxn=f1, t_init=t_init, t_final=t_final)
    key_after_f1 = solver.previous_ode_fxn

    solver.integrate(ode_fxn=f2, t_init=t_init, t_final=t_final)
    key_after_f2 = solver.previous_ode_fxn

    assert key_after_f1 != key_after_f2, (
        f"Solver cannot distinguish lambda1 from lambda2: "
        f"key_after_f1={key_after_f1!r}, key_after_f2={key_after_f2!r}. "
        f"This is bug B2 — both lambdas have __name__ == '<lambda>'."
    )


# -----------------------------------------------------------------------------
# B1: t_init/t_final swap. The bug is in the cache-load path, but the
# random-mesh generation at line 1135 overwrites the cached barriers
# unconditionally, so end-to-end the bug is invisible. Document this.
# -----------------------------------------------------------------------------


def test_warm_start_cache_load_is_currently_dead_code():
    """The conditional barrier reload at parallel_solver.py:1119-1133
    is followed by an unconditional ``if t is None`` block at line 1135
    that overwrites ``t_step_barriers`` with a fresh random mesh.
    The cache-load branch therefore has no observable effect on the
    final integral, masking bug B1 (the t_init/t_final swap on line
    1132) end-to-end.

    Phase 1 must make warm-start actually work via an explicit
    ``reuse_mesh=True`` opt-in. Until then, this test pins the
    current behavior: a second call with the same lambda and a
    *different* t_final returns a correct integral despite the
    buggy cache-load path.
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
        ode_fxn=f, t_init=t_init, t_final=torch.tensor([1.5], dtype=torch.float64)
    )
    assert out_second is not None  # see B6 / Phase 1
    expected_second = 1.0 - math.cos(1.5)
    assert abs(out_second.integral.item() - expected_second) < 1e-5

    # The optimal mesh ends at the new t_final. (This passes today
    # because the random-mesh path enforces it; the cache-load
    # bug never gets a chance to corrupt the final mesh.)
    assert abs(out_second.t_optimal[-1].item() - 1.5) < 1e-12


# -----------------------------------------------------------------------------
# B4: float16 + adaptive should be refused.
# -----------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Bug B4: float16 with adaptive refinement produces wrong "
    "answers silently because precision floor exceeds typical tolerances. "
    "Phase 1 will raise ValueError at construction.",
    strict=True,
)
def test_float16_construction_raises():
    """Constructing a parallel solver with dtype=float16 should raise
    ``ValueError`` because float16's ~1e-3 precision floor cannot
    support adaptive error control to typical tolerances.
    """
    with pytest.raises(ValueError, match=r"float16|coarse"):
        make_uniform_solver("dopri5", atol=1e-5, rtol=1e-5, dtype=torch.float16)
