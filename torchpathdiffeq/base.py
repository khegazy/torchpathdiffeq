"""
Base classes and data structures for the torchpathdiffeq integration library.

Defines the core abstractions that all solvers build on:

- ``steps`` enum: the three integration step strategies (fixed, adaptive uniform,
  adaptive variable).
- ``IntegralOutput`` / ``MethodOutput``: dataclasses carrying results through the
  integration pipeline.
- ``SolverBase``: abstract base class providing dtype/device management, default
  parameter handling, and the ``integrate()`` / ``_calculate_integral()`` interface
  that concrete solvers must implement.
"""

from __future__ import annotations

asdfasdfasdf
dddddlklklklklklklklakdslkjasdlfk;jasd;lkfjas;dklfjal;skfejkfhlkajdsfhlakjsdflkasdjfkajsddfhslasdfasdfaf;awdfj;oasidjflaskdfn;lkasdhg;lkasdjf
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from .distributed import DistributedEnvironment

if TYPE_CHECKING:
    from collections.abc import Callable


class steps(Enum):
    """
    Enumeration of integration step strategies.

    - FIXED: Fixed step sizes (not adaptive).
    - ADAPTIVE_UNIFORM: Adaptive step sizes with quadrature points placed at
      fixed fractional positions within each step (tableau c values are constant).
    - ADAPTIVE_VARIABLE: Adaptive step sizes with quadrature points at arbitrary
      positions. Tableau b weights are recomputed dynamically based on actual
      point positions.
    """

    FIXED = 0
    ADAPTIVE_UNIFORM = 1
    ADAPTIVE_VARIABLE = 2


def get_sampling_type(sampling_type: str) -> steps:
    """
    Convert a string sampling type name into the corresponding ``steps`` enum.

    Accepts both short names ('uniform', 'variable') and full names
    ('adaptive_uniform', 'adaptive_variable').

    Args:
        sampling_type: One of 'fixed', 'uniform', 'adaptive_uniform',
            'variable', or 'adaptive_variable'.

    Returns:
        The corresponding ``steps`` enum value.

    Raises:
        KeyError: If sampling_type is not a recognized name.
    """
    types = {
        "fixed": steps.FIXED,
        "adaptive_uniform": steps.ADAPTIVE_UNIFORM,
        "uniform": steps.ADAPTIVE_UNIFORM,
        "adaptive_variable": steps.ADAPTIVE_VARIABLE,
        "variable": steps.ADAPTIVE_VARIABLE,
    }
    return types[sampling_type]


@dataclass
class IntegralOutput:
    """
    Complete output of a numerical integration run.

    Returned by all solver ``integrate()`` methods. Contains the computed
    integral value along with diagnostics about step sizes, errors, and
    the optimized time mesh for potential reuse.

    Attributes:
        integral: The computed integral value. Shape: [D].
        loss: Loss value computed by the loss function (defaults to the
            integral itself). Shape: scalar or [D].
        gradient_taken: Whether backpropagation was performed during
            this integration run.
        t_optimal: Optimized time step barriers for reuse in subsequent
            integration calls with the same integrand. Shape: [M, T].
        t: Time points at which the integrand was evaluated, organized by
            integration step. Shape: [N, C, T].
        h: Step sizes (t_right - t_left) for each integration step.
            Shape: [N, T].
        y: Integrand evaluations at each time point. Shape: [N, C, D].
        sum_steps: Weighted RK contribution of each step to the total
            integral (h * sum(b_i * y_i) per step). Shape: [N, D].
        integral_error: Estimated total error (sum of step errors) using
            the embedded lower-order method. Shape: [D].
        sum_step_errors: Per-step error estimates from the difference between
            the primary and embedded RK methods. Shape: [N, D].
        error_ratios: Per-step error ratios (error / tolerance). Values > 1
            indicate the step did not meet accuracy requirements. Shape: [N].
        t_init: Lower integration bound used. Shape: [T].
        t_final: Upper integration bound used. Shape: [T].
        y0: Initial integral value used. Shape: [D].
    """

    integral: torch.Tensor
    loss: torch.Tensor = None
    gradient_taken: bool = None
    t_optimal: torch.Tensor = None
    t: torch.Tensor = None
    h: torch.Tensor = None
    y: torch.Tensor = None
    sum_steps: torch.Tensor = None
    integral_error: torch.Tensor = None
    sum_step_errors: torch.Tensor = None
    error_ratios: torch.Tensor = None
    t_init: torch.Tensor = None
    t_final: torch.Tensor = None
    y0: torch.Tensor = None


@dataclass
class MethodOutput:
    """
    Output from a single batch of RK integration steps.

    Produced by ``_calculate_integral()`` in concrete solver subclasses.
    Contains both the primary integral estimate and the embedded error
    estimate, broken down per step.

    Attributes:
        integral: Total integral contribution from this batch of steps
            (sum of sum_steps). Shape: [D].
        integral_error: Total error estimate from this batch (sum of
            sum_step_errors). Shape: [D].
        sum_steps: Per-step integral contributions. Each entry is
            h_i * sum(b_j * y_ij) for step i. Shape: [N, D].
        sum_step_errors: Per-step error estimates from the difference between
            order-p and order-(p-1) methods. Shape: [N, D].
        h: Step sizes (t_right - t_left) for each step. Shape: [N, T].
    """

    integral: torch.Tensor
    integral_error: torch.Tensor
    sum_steps: torch.Tensor
    sum_step_errors: torch.Tensor
    h: torch.Tensor


class SolverBase(DistributedEnvironment):
    """
    Abstract base class for all numerical integration solvers.

    Provides common infrastructure inherited by both the serial solver
    (which wraps torchdiffeq) and the parallel adaptive solvers. Handles:

    - Dtype management: converts all tensors to a consistent dtype, and sets
      dtype-appropriate assertion tolerances (atol_assert, rtol_assert) used
      for internal sanity checks (e.g., verifying time ordering).
    - Device management: inherited from DistributedEnvironment.
    - Default parameter storage: ode_fxn, y0, t_init, t_final can be set at
      construction and reused across multiple integrate() calls.
    - Warm-start caching: stores t_step_barriers_previous and previous_ode_fxn
      so that repeated integration of the same function can reuse the optimized
      time mesh from the prior run.

    Subclasses must implement:
        - ``_set_solver_dtype(dtype)``: update solver-specific state when dtype changes.
        - ``_calculate_integral(t, y, y0)``: compute integral + error for a batch of steps.
        - ``integrate(...)``: run the full integration procedure.
    """

    def __init__(
        self,
        method: str,
        atol: float,
        rtol: float,
        y0: torch.Tensor = torch.tensor([0], dtype=torch.float64),
        ode_fxn: Callable | None = None,
        t_init: torch.Tensor = torch.tensor([0], dtype=torch.float64),
        t_final: torch.Tensor = torch.tensor([1], dtype=torch.float64),
        dtype: torch.dtype = torch.float64,
        eval: bool = False,
        device: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the solver base with integration parameters.

        Args:
            method: Name of the RK integration method (e.g. 'dopri5', 'bosh3').
                Converted to lowercase.
            atol: Absolute error tolerance for adaptive step control. Steps with
                error below atol + rtol*|integral| are accepted.
            rtol: Relative error tolerance for adaptive step control.
            y0: Initial value of the integral (accumulator starting point).
                Shape: [D].
            ode_fxn: The integrand function f(t) -> Tensor. Takes time points
                of shape [N, T] and returns evaluations of shape [N, D].
                Can also be provided at integrate() call time.
            t_init: Lower bound of integration. Shape: [T].
            t_final: Upper bound of integration. Shape: [T].
            dtype: Floating point precision for all computations. Supported:
                torch.float64, torch.float32, torch.float16.
            eval: If True, disables training mode (no gradient computation).
            device: Device string (e.g. 'cuda', 'cpu'). Passed to
                DistributedEnvironment for device assignment.
            *args: Additional arguments forwarded to DistributedEnvironment.
            **kwargs: Additional keyword arguments forwarded to
                DistributedEnvironment.
        """
        super().__init__(*args, **kwargs, device_type=device)

        self.method_name = method.lower()
        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.y0 = y0.to(self.device)
        self.t_init = t_init.to(self.device)
        self.t_final = t_final.to(self.device)
        self.training = not eval
        # Cached time barriers from last integration for warm-starting
        self.t_step_barriers_previous = None
        # Name of the last integrated function (for warm-start matching)
        self.previous_ode_fxn = None

        self._set_dtype(dtype)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        """
        Set the floating-point precision for all solver tensors.

        Converts y0, t_init, t_final, and cached barriers to the new dtype.
        Also sets dtype-appropriate assertion tolerances used for internal
        sanity checks (e.g., verifying that time points are monotonically
        ordered despite floating-point rounding).

        The assertion tolerances by dtype are:
            - float64: atol_assert=1e-15, rtol_assert=1e-7
            - float32: atol_assert=1e-7,  rtol_assert=1e-5
            - float16: atol_assert=1e-3,  rtol_assert=1e-1

        Args:
            dtype: Target dtype. Must be float64, float32, or float16.

        Raises:
            ValueError: If dtype is not one of the three supported types.
        """
        if hasattr(self, "dtype") and dtype == self.dtype:
            return

        self.dtype = dtype
        self.y0 = self.y0.to(self.dtype)
        self.t_init = self.t_init.to(self.dtype)
        self.t_final = self.t_final.to(self.dtype)
        if self.t_step_barriers_previous is not None:
            self.t_step_barriers_previous = self.t_step_barriers_previous.to(self.dtype)

        # Set assertion tolerances appropriate for this precision level.
        # These are used in internal sanity checks (e.g., time ordering),
        # NOT for integration error control (that uses self.atol/self.rtol).
        if self.dtype == torch.float64:
            self.atol_assert = 1e-15
            self.rtol_assert = 1e-7
        elif self.dtype == torch.float32:
            self.atol_assert = 1e-7
            self.rtol_assert = 1e-5
        elif self.dtype == torch.float16:
            self.atol_assert = 1e-3
            self.rtol_assert = 1e-1
        else:
            raise ValueError(
                "Given dtype must be torch.float64, torch.float32, or torch.float16"
            )

        self._set_solver_dtype(self.dtype)

    def set_dtype_by_input(
        self,
        t: torch.Tensor | None = None,
        t_init: torch.Tensor | None = None,
        t_final: torch.Tensor | None = None,
    ) -> None:
        """
        Infer and set the solver dtype from the dtype of input tensors.

        Checks t first, then t_init, then t_final. The first non-None
        tensor's dtype is used. This allows the solver to automatically
        match the precision of user-provided inputs.

        Args:
            t: Optional time points tensor.
            t_init: Optional lower integration bound tensor.
            t_final: Optional upper integration bound tensor.
        """
        if t is not None:
            self._set_dtype(t.dtype)
        elif t_init is not None:
            self._set_dtype(t_init.dtype)
        elif t_final is not None:
            self._set_dtype(t_final.dtype)

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """
        Update solver-specific state when dtype changes.

        Called by ``_set_dtype()`` after converting base tensors. Subclasses
        must implement this to convert their own tensors (e.g., RK tableaux).

        Args:
            dtype: The new dtype to convert to.
        """
        raise NotImplementedError

    def _check_variables(
        self,
        ode_fxn: Callable | None = None,
        t_init: torch.Tensor | None = None,
        t_final: torch.Tensor | None = None,
        y0: torch.Tensor | None = None,
    ) -> tuple[
        Callable | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """
        Fill in missing arguments with stored defaults and move to correct device.

        Any argument that is None is replaced with the value stored at
        construction time. Tensor arguments are cast to the solver's dtype
        and moved to the solver's device.

        Args:
            ode_fxn: Integrand function, or None to use self.ode_fxn.
            t_init: Lower integration bound, or None to use self.t_init.
            t_final: Upper integration bound, or None to use self.t_final.
            y0: Initial integral value, or None to use self.y0.

        Returns:
            Tuple of (ode_fxn, t_init, t_final, y0) with defaults filled
            in and tensors on the correct device/dtype.
        """
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        t_init = self.t_init if t_init is None else t_init
        t_final = self.t_final if t_final is None else t_final
        y0 = self.y0 if y0 is None else y0

        if t_init is not None:
            t_init = t_init.to(self.dtype).to(self.device)
        if t_final is not None:
            t_final = t_final.to(self.dtype).to(self.device)
        if y0 is not None:
            y0 = y0.to(self.dtype).to(self.device)
        return ode_fxn, t_init, t_final, y0

    def train(self) -> None:
        """Enable training mode (gradients will be computed during integration)."""
        self.training = True

    def eval(self) -> None:
        """Enable evaluation mode (no gradient computation during integration)."""
        self.training = False

    def _calculate_integral(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        y0: torch.Tensor = torch.tensor([0], dtype=torch.float64),
    ) -> MethodOutput:
        """
        Compute the integral and error estimate for a batch of integration steps.

        This is the core numerical method that subclasses must implement.
        Given time points and integrand evaluations at those points, applies
        the specific RK quadrature rule to produce integral contributions
        and error estimates for each step.

        Args:
            t: Time points for each step, with C quadrature points per step.
                Shape: [N, C, T] where N=steps, C=quadrature points, T=time dims.
            y: Integrand evaluations at the time points in t.
                Shape: [N, C, D] where D=output dimensionality of the integrand.
            y0: Starting value for the integral accumulator. Shape: [D].

        Returns:
            MethodOutput containing the integral, error, per-step contributions,
            per-step errors, and step sizes.
        """
        raise NotImplementedError

    def _integral_loss(self, integral: IntegralOutput, *args, **kwargs) -> torch.Tensor:
        """
        Default loss function: returns the integral value itself.

        Used when no custom loss_fxn is provided. Can be overridden or
        replaced at integrate() call time to compute a custom loss for
        gradient-based optimization.

        Args:
            integral: The current IntegralOutput from this batch.

        Returns:
            Loss tensor for backpropagation. Shape: same as integral.integral.
        """
        return integral.integral

    def integrate(
        self,
        ode_fxn: Callable,
        y0: torch.Tensor = torch.tensor([0], dtype=torch.float64),
        t_init: torch.Tensor = torch.tensor([0], dtype=torch.float64),
        t_final: torch.Tensor = torch.tensor([1], dtype=torch.float64),
        t: torch.Tensor | None = None,
        ode_args: tuple = (),
    ) -> IntegralOutput:
        """
        Perform numerical path integration of ode_fxn from t_init to t_final.

        This is the main entry point that users call. Subclasses implement
        the actual integration logic (serial or parallel adaptive).

        The integrand ode_fxn should be a function of time only: f(t) -> Tensor.
        It does NOT depend on accumulated state y (this is numerical quadrature,
        not ODE solving). This independence between steps is what enables
        parallel evaluation.

        Args:
            ode_fxn: The integrand function. Takes time points of shape [N, T]
                and returns values of shape [N, D].
            y0: Initial value of the integral (accumulator start). Shape: [D].
            t_init: Lower integration bound. Shape: [T].
            t_final: Upper integration bound. Shape: [T].
            t: Optional initial time points. If provided, these serve as the
                starting mesh for adaptive refinement. Shape depends on solver:
                [N, T] for step barriers, [N, C, T] for full quadrature points.
            ode_args: Extra arguments passed to ode_fxn after the time tensor.

        Returns:
            IntegralOutput containing the computed integral, error estimates,
            time mesh, and optimization diagnostics.

        Note:
            Handling of the input time ``t`` differs between parallel and serial
            solvers. See each solver's documentation for details.
        """
        raise NotImplementedError

    def __del__(self) -> None:
        """Destructor that cleans up the distributed process group."""
        self.end_process()
