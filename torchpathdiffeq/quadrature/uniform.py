"""
Uniform-sampling adaptive quadrature: ``_UniformAdaptiveQuadratureBase``.

In uniform sampling, quadrature points within each step are placed at
fixed fractional positions defined by the method's tableau.c values
(e.g. dopri5 always uses ``c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]``). The
b weights are also constant.

When the adaptive controller splits a step, the new sub-steps get
fresh evaluations at the standard c positions; the old evaluations
are discarded. When merging two steps, a new combined step is built
spanning ``[A.start, B.end]`` and re-evaluated at the standard c
positions.

This intermediate base class holds the uniform-specific overrides of
the abstract methods declared in ``AdaptiveQuadrature``. Concrete
solvers subclass this with a specific RK family for the
``_calculate_integral`` implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange

from torchpathdiffeq.base import steps
from torchpathdiffeq.methods import UNIFORM_METHODS, _get_method

from .base import AdaptiveQuadrature

if TYPE_CHECKING:
    from collections.abc import Callable


class _UniformAdaptiveQuadratureBase(AdaptiveQuadrature):
    """
    Parallel solver using uniform-sampling Runge-Kutta methods.

    In uniform sampling, quadrature points within each step are placed at
    fixed fractional positions defined by the tableau's c values. For example,
    dopri5 always evaluates at c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1] of each
    step's width. The b weights are also fixed constants.

    This is the simpler and more common solver variant. When a step is split,
    the new sub-steps get fresh evaluations at the standard c positions.

    Supported methods: 'adaptive_heun', 'fehlberg2', 'bosh3', 'dopri5'.

    Attributes:
        method: The MethodClass instance containing the Butcher tableau.
        order: Convergence order of the RK method.
        C: Number of quadrature points per step (len(tableau.c)).
        Cm1: C - 1, used for indexing and step calculations.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the uniform solver and set up its RK method."""
        super().__init__(*args, **kwargs)
        self._setup_method(self.dtype)

    def _setup_method(self, dtype: torch.dtype) -> None:
        """
        Initialize the RK method and extract its properties.

        Loads the named method from UNIFORM_METHODS, moves its tableau
        tensors to the correct device and dtype, and stores the method's
        order and number of quadrature points (C).

        Args:
            dtype: Floating-point dtype for the tableau tensors.

        Raises:
            AssertionError: If method_name is not in UNIFORM_METHODS.
        """
        error_message = f"Cannot find method '{self.method_name}' in supported method: {list(UNIFORM_METHODS.keys())}"
        assert self.method_name in UNIFORM_METHODS, error_message
        self.method = _get_method(
            steps.ADAPTIVE_UNIFORM, self.method_name, self.device, dtype
        )
        self.order = self.method.order
        self.C = len(self.method.tableau.c)
        self.Cm1 = self.C - 1

    def _set_solver_dtype(self, dtype: torch.dtype) -> None:
        """Re-initialize the method when the solver's dtype changes."""
        self._setup_method(dtype)

    """
    def _initial_t_steps(self,
            t,
            mesh_init=None,
            mesh_final=None
        ):
        Creates an initial time sampling tensor either from scratch or from a
        tensor of time points with dimension d.

        Args:
            t (Tensor): Input time, either None or tensor starting and
                ending at the integration bounds
            mesh_init (Tensor, optional): Minimum of integral range
            mesh_final (Tensor, optional): Maximum of integral range

        Shapes:
            t : [N, T] will populate intermediate evaluations according to
                integration method; [N, C, T] will return t if C is the same
                as the number of evaluations per step, otherwise it will create
                C steps between the first and last values in the second dim
            mesh_init: [T]
            mesh_final: [T]

        # Get variables or populate with default values, send to correct device
        _, mesh_init, mesh_final, _ = self._check_variables(
            None, mesh_init, mesh_final, None
        )
        if t is None:
            t = torch.linspace(0, 1., 7*self.Cm1 + 1, device=self.device).unsqueeze(-1)
            t = mesh_init + t*(mesh_final - mesh_init)
        elif len(t.shape) == 3:
            if t.shape[1] == self.C:
                return t
            else:
                if len(t) > 1:
                    logger.debug("t values: %s", t[:,:,0])
                    assert torch.allclose(t[:-1,-1], t[1:,0], atol=self.atol_assert, rtol=self.rtol_assert)
                t = t[:,torch.tensor([0,-1], dtype=torch.int, device=self.device),:]
                t = torch.flatten(t, start_dim=0, end_dim=1)
        return self._t_step_interpolate(t[:-1], t[1:])
    """

    def _t_step_interpolate(
        self, t_left: torch.Tensor, t_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Place quadrature points within each step using the tableau's c values.

        For uniform sampling, quadrature points are at fixed fractional positions
        within each step: t_i = t_left + c_i * (t_right - t_left). For example,
        with c = [0, 0.5, 1], points are placed at the start, midpoint, and end.

        Args:
            t_left: Left boundary of each step. Shape: [N, T].
            t_right: Right boundary of each step. Shape: [N, T].

        Returns:
            Quadrature point positions. Shape: [N, C, T] where C is the
            number of tableau c values.
        """
        # Compute step width and scale by tableau c positions
        dt = (t_right - t_left).unsqueeze(1)
        steps = self.method.tableau.c.unsqueeze(-1) * dt
        return t_left.unsqueeze(1) + steps

    def _evaluate_adaptive_y(
        self,
        f: Callable,
        idxs_add: torch.Tensor,
        _y: torch.Tensor,
        nodes: torch.Tensor,
        ode_args: tuple = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split failed steps at their midpoints and evaluate the integrand.

        Each step that failed the error check is divided into two sub-steps
        at its midpoint. Fresh quadrature points are placed in each sub-step
        using the tableau's c values, and the integrand is evaluated at all
        new points.

        Args:
            f: The integrand function f(t). Takes [N, T], returns [N, D].
            idxs_add: Indices of steps that need splitting. Shape: [R].
            _y: Current integrand evaluations (unused here, present for
                interface compatibility with the variable solver). Shape: [N, C, D].
            nodes: Current quadrature point positions. Shape: [N, C, T].
            ode_args: Extra arguments passed to f.

        Returns:
            Tuple of (y_add, nodes_new) where:
                - y_add: Integrand values at new quadrature points.
                    Shape: [2*R, C, D] (two sub-steps per split step).
                - nodes_new: New quadrature point positions.
                    Shape: [2*R, C, T].
        """
        T = nodes.shape[-1]
        # Compute the midpoint of each failed step
        t_mid = (nodes[idxs_add, -1] + nodes[idxs_add, 0]) / 2.0
        # Build left and right boundaries for the two new sub-steps
        t_left = torch.concatenate([nodes[idxs_add, None, 0], t_mid[:, None]], dim=1)
        t_right = torch.concatenate([t_mid[:, None], nodes[idxs_add, None, -1]], dim=1)
        # Place quadrature points in each sub-step and evaluate
        nodes_new = self._t_step_interpolate(t_left.view(-1, T), t_right.view(-1, T))
        y_add = f(nodes_new.view(-1, T), *ode_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", C=self.C)
        return y_add, nodes_new

    def _merge_excess_nodes(
        self,
        nodes: torch.Tensor,
        sum_steps: torch.Tensor,
        sum_step_errors: torch.Tensor,
        remove_idxs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of consecutive over-resolved steps into single larger steps.

        For uniform sampling, merging replaces two consecutive steps [A, B]
        with a single step spanning [A.start, B.end]. New quadrature points
        are placed at the standard tableau c positions within the merged step.
        The integral contributions and errors of both steps are summed.

        Args:
            nodes: Quadrature point positions for all steps. Shape: [N, C, T].
            sum_steps: Integral contribution of each step. Shape: [N, D].
            sum_step_errors: Error estimate of each step. Shape: [N, D].
            remove_idxs: Indices of the first step in each pair to merge.
                The pair (remove_idxs[i], remove_idxs[i]+1) is merged.
                Shape: [R].

        Returns:
            Tuple of (nodes_pruned, sum_steps_pruned, sum_step_errors_pruned)
            with the merged steps replacing the original pairs. Each has
            N-R entries along the first dimension.
        """
        if len(remove_idxs) == 0 or len(nodes) == 1:
            return nodes, sum_steps, sum_step_errors

        # Create merged steps spanning from the left edge of the first step
        # to the right edge of the second step, with new c-based quadrature points
        nodes_replace = self._t_step_interpolate(
            nodes[remove_idxs, 0], nodes[remove_idxs + 1, -1]
        )

        # Sum integral contributions and errors of the merged pair
        sum_steps_replace = sum_steps[remove_idxs] + sum_steps[remove_idxs + 1]
        sum_step_errors_replace = (
            sum_step_errors[remove_idxs] + sum_step_errors[remove_idxs + 1]
        )

        # Remove the first step of each pair from the arrays
        remove_mask = torch.ones(len(nodes), device=self.device, dtype=torch.bool)
        remove_mask[remove_idxs] = False
        nodes_pruned = nodes[remove_mask]
        sum_steps_pruned = sum_steps[remove_mask]
        sum_step_errors_pruned = sum_step_errors[remove_mask]

        # Place the merged step data at the position of the second (kept) step,
        # adjusting indices to account for earlier removals shifting positions
        remove_idxs_shifted = remove_idxs - torch.arange(
            len(remove_idxs), device=self.device
        )
        nodes_pruned[remove_idxs_shifted] = nodes_replace
        sum_steps_pruned[remove_idxs_shifted] = sum_steps_replace
        sum_step_errors_pruned[remove_idxs_shifted] = sum_step_errors_replace

        # Verify time ordering is preserved after merging
        nodes_pruned_flat = torch.flatten(nodes_pruned, start_dim=0, end_dim=1)
        assert torch.all(
            nodes_pruned_flat[1:] - nodes_pruned_flat[:-1] + self.atol_assert >= 0
        )
        nodes_flat = torch.flatten(nodes, start_dim=0, end_dim=1)
        assert torch.all(nodes_flat[1:] - nodes_flat[:-1] + self.atol_assert >= 0)

        return nodes_pruned, sum_steps_pruned, sum_step_errors_pruned
