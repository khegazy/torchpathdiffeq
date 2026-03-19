"""
Sequential (non-parallel) integration solver.

Wraps the ``torchdiffeq.odeint`` function to provide a sequential integration
fallback using the same SolverBase interface as the parallel solvers. Unlike
the parallel solvers which evaluate many steps simultaneously, this solver
processes steps one at a time in the traditional sequential manner.

Note: The serial solver uses torchdiffeq's ODE interface where the integrand
is called as f(t, y). This differs from the parallel solvers where the
integrand is called as f(t) since steps are independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchdiffeq import odeint

from .base import IntegralOutput, SolverBase

if TYPE_CHECKING:
    from collections.abc import Callable


class SerialAdaptiveStepsizeSolver(SolverBase):
    """
    Sequential adaptive-stepsize solver using torchdiffeq as the backend.

    This solver evaluates integration steps one at a time (sequentially),
    delegating the actual stepping to torchdiffeq's ``odeint`` function.
    It serves as a baseline comparison for the parallel solvers and as a
    fallback when parallelization is not beneficial.

    The solver supports the same RK methods as torchdiffeq (e.g. dopri5,
    bosh3, adaptive_heun, fehlberg2).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the serial solver wrapping torchdiffeq."""
        super().__init__(*args, **kwargs)

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """No-op: the serial solver has no method-specific tensors to convert."""

    def _calculate_integral(self, _t, _y, _y0):
        """Not used: the serial solver delegates to torchdiffeq.odeint."""
        msg = "Serial solver does not use _calculate_integral"
        raise NotImplementedError(msg)

    def integrate(
        self,
        ode_fxn: Callable | None = None,
        y0: torch.Tensor | None = None,
        t_init: torch.Tensor | None = None,
        t_final: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> IntegralOutput:
        """
        Perform sequential numerical integration using torchdiffeq.odeint.

        Evaluates the integral of ode_fxn from t_init to t_final using
        the adaptive-stepsize ODE solver from torchdiffeq. The solver
        internally chooses step sizes to meet the specified tolerances.

        Args:
            ode_fxn: The integrand function f(t, y). If None, uses the
                function provided at construction time. Shape of input:
                t is scalar, y is [D]. Returns shape [D].
            y0: Initial value of the integral. Shape: [D]. If None, uses
                the value from construction.
            t_init: Lower integration bound. Shape: [T]. If None, uses
                the value from construction.
            t_final: Upper integration bound. Shape: [T]. If None, uses
                the value from construction.
            t: Time points at which to return the integral value. If None,
                defaults to [t_init, t_final]. Shape: [N, T].

        Returns:
            IntegralOutput with the computed integral at the final time
            point and the time mesh used.

        Note:
            The integral is evaluated within [t[0], t[-1]] and the returned
            integral value corresponds to the final time point t[-1].
        """
        # Fill in any missing arguments with stored defaults
        ode_fxn, t_init, t_final, y0 = self._check_variables(
            ode_fxn, t_init, t_final, y0, None
        )
        assert ode_fxn is not None, (
            "Must specify ode_fxn or pass it during class initialization."
        )
        if t is None:
            t = torch.tensor([t_init, t_final], dtype=torch.float64, device=self.device)
        else:
            assert len(t.shape) == 2

        # Delegate to torchdiffeq's sequential ODE solver
        integral = odeint(
            func=ode_fxn,
            y0=y0,
            t=t,
            method=self.method_name,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Return the integral at the final time point
        return IntegralOutput(
            integral=integral[-1],
            t=t,
        )
