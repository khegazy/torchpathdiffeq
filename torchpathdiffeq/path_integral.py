"""
High-level public API for numerical path integration.

Provides ``ode_path_integral()``, the main entry point for users who want
to compute a definite integral without manually constructing solver objects.
This function creates the appropriate solver, runs the integration, and
returns the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import IntegralOutput, steps
from .runge_kutta import get_parallel_RK_solver
from .serial_solver import SerialAdaptiveStepsizeSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch


def ode_path_integral(
    ode_fxn: Callable,
    method: str,
    computation: str = "parallel",
    sampling: str = "uniform",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    t: torch.Tensor | None = None,
    t_init: torch.Tensor | None = None,
    t_final: torch.Tensor | None = None,
    y0: torch.Tensor | None = None,
    remove_cut: float = 0.1,
    total_mem_usage: float = 0.9,
    use_absolute_error_ratio: bool = True,
    device: str | None = None,
    **kwargs,
) -> IntegralOutput:
    """
    Compute the definite integral of ode_fxn from t_init to t_final.

    This is the main public API. It creates a solver of the requested type,
    runs the integration, and returns the result. Supports both parallel
    (GPU-optimized, batch evaluation) and serial (sequential, torchdiffeq-backed)
    computation modes.

    The integrand ode_fxn should depend only on time t (not on accumulated
    state y). This independence between evaluation points is what enables
    the parallel solver to evaluate many steps simultaneously.

    Example::

        result = ode_path_integral(
            ode_fxn=lambda t: torch.sin(t),
            method='dopri5',
            atol=1e-8,
            rtol=1e-6,
            t_init=torch.tensor([0.0]),
            t_final=torch.tensor([3.14159]),
        )
        print(result.integral)  # Should be close to 2.0

    Args:
        ode_fxn: The integrand function. For parallel mode, takes time points
            of shape [N, T] and returns values of shape [N, D]. For serial
            mode, follows the torchdiffeq convention f(t, y).
        method: Name of the RK integration method. Options for parallel mode:
            'adaptive_heun', 'fehlberg2', 'bosh3', 'dopri5' (uniform sampling),
            or 'adaptive_heun', 'generic3' (variable sampling).
        computation: Whether to use 'parallel' (GPU-batched) or 'serial'
            (sequential via torchdiffeq) integration.
        sampling: For parallel computation only. 'uniform' uses fixed quadrature
            point positions within each step; 'variable' computes weights
            dynamically based on actual point positions.
        atol: Absolute error tolerance. A step is accepted when its estimated
            error < atol + rtol * |integral_value|.
        rtol: Relative error tolerance. See atol.
        t: Optional initial time mesh (step barrier positions). If None, the
            solver generates an initial mesh automatically. Shape: [N, T] for
            step barriers.
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].
        y0: Initial value of the integral accumulator. Shape: [D].
        remove_cut: Threshold for merging consecutive steps whose combined
            error ratio is below this value. Must be < 1. Lower values are
            more conservative (keep more steps). Only used in parallel mode.
        total_mem_usage: Fraction of total GPU/CPU memory the solver is
            allowed to use (0 < value <= 1). The solver dynamically adjusts
            batch size to stay within this budget. Only used in parallel mode.
        use_absolute_error_ratio: If True (default), error ratios use the
            total integral value as the reference. If False, uses the
            cumulative integral up to each step (more like traditional ODE
            error control). Only used in parallel mode.
        device: Device to run on (e.g. 'cuda', 'cpu'). If None, auto-detects.
        **kwargs: Additional keyword arguments forwarded to the solver
            constructor (e.g. max_batch, max_path_change, error_calc_idx).

    Returns:
        IntegralOutput containing the computed integral, error estimates,
        time mesh, and optimization diagnostics.

    Raises:
        ValueError: If computation is not 'parallel' or 'serial', or if
            sampling is not 'uniform' or 'variable'.

    Note:
        **Parallel mode**: If t is None, the solver creates an initial mesh
        of ~sqrt(N_init_steps) barriers with random sub-divisions in
        [t_init, t_final]. Steps are adaptively added (split) or removed
        (merged) based on error estimates until all steps meet the tolerance.

        **Serial mode**: Delegates to torchdiffeq.odeint which handles its
        own adaptive stepping internally.
    """
    if computation.lower() == "parallel":
        # Select the sampling strategy for the parallel solver
        if sampling.lower() == "uniform":
            sampling_type = steps.ADAPTIVE_UNIFORM
        elif sampling.lower() == "variable":
            sampling_type = steps.ADAPTIVE_VARIABLE
        else:
            raise ValueError(
                f"Sampling method must be either 'uniform' or 'variable', instead got {sampling}"
            )

        # Create the parallel RK solver and run integration
        integrator = get_parallel_RK_solver(
            sampling_type=sampling_type,
            method=method,
            ode_fxn=ode_fxn,
            atol=atol,
            rtol=rtol,
            remove_cut=remove_cut,
            t_init=t_init,
            t_final=t_final,
            use_absolute_error_ratio=use_absolute_error_ratio,
            device=device,
            **kwargs,
        )

        integral_output = integrator.integrate(
            y0=y0, t=t, t_init=t_init, t_final=t_final, total_mem_usage=total_mem_usage
        )
    elif computation.lower() == "serial":
        # Create the serial (torchdiffeq-backed) solver and run integration
        integrator = SerialAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final,
            device=device,
            **kwargs,
        )

        integral_output = integrator.integrate(
            y0=y0,
            t=t,
            t_init=t_init,
            t_final=t_final,
        )
    else:
        raise ValueError(
            f"Path integral computation type must be 'parallel' or 'serial', not {computation}."
        )

    return integral_output
