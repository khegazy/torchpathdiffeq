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

import time
import torch
from typing import Callable, Optional, Tuple
from torchdiffeq import odeint

from .base import SolverBase, IntegralOutput


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
        super().__init__(*args, **kwargs)

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """No-op: the serial solver has no method-specific tensors to convert."""
        pass

    def integrate(
            self,
            ode_fxn: Optional[Callable] = None,
            y0: Optional[torch.Tensor] = None,
            t_init: Optional[torch.Tensor] = None,
            t_final: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            ode_args: Optional[tuple] = None,
            max_batch: Optional[int] = None
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
            ode_args: Unused (present for interface compatibility with
                parallel solvers).
            max_batch: Unused (present for interface compatibility with
                parallel solvers).

        Returns:
            IntegralOutput with the computed integral at the final time
            point and the time mesh used.

        Note:
            The integral is evaluated within [t[0], t[-1]] and the returned
            integral value corresponds to the final time point t[-1].
        """
        # Fill in any missing arguments with stored defaults
        ode_fxn, t_init, t_final, y0 = self._check_variables(
            ode_fxn, t_init, t_final, y0
        )
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        if t is None:
            t=torch.tensor(
                [t_init, t_final], dtype=torch.float64, device=self.device
            )
        else:
            assert len(t.shape) == 2

        # Delegate to torchdiffeq's sequential ODE solver
        integral = odeint(
            func=ode_fxn,
            y0=y0,
            t=t,
            method=self.method_name,
            rtol=self.rtol,
            atol=self.atol
        )

        # Return the integral at the final time point
        return IntegralOutput(
            integral=integral[-1],
            t=t,
        )