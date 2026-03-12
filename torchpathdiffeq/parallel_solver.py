"""
Parallel adaptive-stepsize integration solvers.

This module contains the core parallel integration engine. Unlike traditional
sequential integrators that must evaluate one step at a time (because each step
depends on the previous result), this solver evaluates many integration steps
simultaneously in a batch. This is possible because the integrand f(t) depends
only on time, not on accumulated state -- each step's contribution to the
integral is independent and can be computed in parallel.

The integration domain [t_init, t_final] is divided by "barriers" into steps.
Within each step, C quadrature points are placed (per the RK tableau). The
solver adaptively refines this mesh: steps with too much error are split into
smaller steps; consecutive steps with very little error are merged.

Key concepts:

- **t_step_barriers**: Boundary points dividing [t_init, t_final] into steps.
  Shape: [M, T] where M is the number of barriers. Step i spans from
  barrier[i] to barrier[i+1].

- **t_step_trackers**: Boolean array of length M. True means the step starting
  at that barrier still needs to be evaluated (or re-evaluated after splitting).

- **Batching**: When there are more steps than fit in GPU memory, the solver
  processes them in batches. Batch size is determined dynamically by measuring
  the memory footprint of the integrand function.

Class hierarchy (defined here):

- ``ParallelAdaptiveStepsizeSolver``: Abstract base with the main integrate()
  loop, adaptive step management, error computation, and memory management.

- ``ParallelUniformAdaptiveStepsizeSolver``: Concrete subclass for methods with
  fixed tableau c values (quadrature points at constant fractional positions).

- ``ParallelVariableAdaptiveStepsizeSolver``: Concrete subclass for methods
  where quadrature points can be at arbitrary positions within each step.
"""

import psutil
import time
import torch
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from einops import rearrange
from dataclasses import asdict as dataclass_asdict

from .methods import _get_method, UNIFORM_METHODS, VARIABLE_METHODS
from .base import SolverBase, IntegralOutput, MethodOutput, steps


class ParallelAdaptiveStepsizeSolver(SolverBase):
    """
    Base class for parallel adaptive-stepsize numerical integration.

    Implements the main integration loop that:
    1. Initializes a mesh of time step barriers across the integration domain.
    2. Evaluates the integrand at quadrature points within each step (in batches).
    3. Computes integral contributions and error estimates per step using RK methods.
    4. Adaptively refines the mesh: splits high-error steps, merges low-error pairs.
    5. Repeats until all steps meet the error tolerance.
    6. Computes an optimal mesh for potential reuse.

    Subclasses must implement:
        - ``_t_step_interpolate(t_left, t_right)``: Place quadrature points within steps.
        - ``_evaluate_adaptive_y(...)``: Evaluate integrand at refined points.
        - ``_merge_excess_t(...)``: Merge consecutive low-error steps.
        - ``_calculate_integral(t, y, y0)``: Compute RK integral + error for a batch.

    Attributes:
        remove_cut: Error ratio threshold for merging steps (default 0.1).
        max_batch: Maximum number of integrand evaluations per batch. If None,
            determined dynamically from available memory.
        total_mem_usage: Fraction of total memory to use (0 < value <= 1).
        max_path_change: If set, stops integration when the fraction of failing
            steps exceeds this value (used with pre-specified time meshes).
        use_absolute_error_ratio: If True, uses the total integral for error
            normalization. If False, uses cumulative integral up to each step.
        error_calc_idx: If set, only this output dimension is used for error
            computation (useful for multi-dimensional integrands where only
            one dimension should drive adaptivity).
        method: The RK method object (set by subclass).
        order: Convergence order of the RK method (set by subclass).
        C: Number of quadrature points per integration step (set by subclass).
        Cm1: C - 1, used frequently in index calculations (set by subclass).
    """

    def __init__(
            self,
            remove_cut: float = 0.1,
            max_batch: Optional[int] = None,
            total_mem_usage: float = 0.9,
            max_path_change: Optional[float] = None,
            use_absolute_error_ratio: bool = True,
            error_calc_idx: Optional[int] = None,
            *args,
            **kwargs) -> None:
        """
        Initialize the parallel adaptive solver.

        Args:
            remove_cut: Error ratio threshold below which consecutive step pairs
                are merged. Must be < 1. Lower values keep more steps (more
                conservative). Default: 0.1.
            max_batch: Maximum number of integrand evaluations per batch. If None,
                batch size is determined dynamically based on available memory.
            total_mem_usage: Fraction of total device memory the solver may use
                for batched evaluations (0 < value <= 1). Default: 0.9.
            max_path_change: If set, and the user provides a time mesh (t is not
                None), integration stops early if this fraction of steps fail
                the error tolerance. Useful for iterative optimization.
            use_absolute_error_ratio: If True, error ratios use the total
                (converging) integral value. If False, uses cumulative sum up
                to each step. Default: True.
            error_calc_idx: If set, only this dimension index of the integrand
                output is used for error-based step decisions.
            *args: Forwarded to SolverBase (and DistributedEnvironment).
            **kwargs: Forwarded to SolverBase (and DistributedEnvironment).
        """
        
        super().__init__(*args, **kwargs)
        assert remove_cut < 1.
        self.remove_cut = remove_cut
        self.max_batch = max_batch
        self.max_path_change = max_path_change
        self.use_absolute_error_ratio = use_absolute_error_ratio

        self.method = None
        self.order = None
        self.C = None
        self.Cm1 = None
        self.error_calc_idx = error_calc_idx
        self.total_mem_usage = total_mem_usage


    def _initial_t_steps(
            self,
            t,
            t_init=None,
            t_final=None
        ):
        """
        Creates an initial time sampling tensor either from scratch or from a
        tensor of time points with dimension d.

        Args:
            t (Tensor): Input time, either None or tensor starting and
                ending at the integration bounds
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
        
        Shapes:
            t : [N, T] will populate intermediate evaluations according to
                integration method, [N, C, T] will return t
            t_init: [T]
            t_final: [T]
        """
        raise NotImplementedError
    

    def _evaluate_adaptive_y(
            self,
            ode_fxn: Callable,
            idxs_add: torch.Tensor,
            y: torch.Tensor,
            t: torch.Tensor,
            ode_args: tuple = ()
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the integrand at new points created by splitting failed steps.

        When steps fail the error tolerance, they are split into two smaller
        steps. This method evaluates the integrand at the new quadrature points
        needed for the smaller steps. The implementation differs between uniform
        and variable solvers.

        Args:
            ode_fxn: The integrand function.
            idxs_add: Indices of steps that need to be split. Shape: [n_add].
            y: Current integrand evaluations for all steps. Shape: [N, C, D].
            t: Current time points for all steps. Shape: [N, C, T].
            ode_args: Extra arguments passed to ode_fxn.

        Returns:
            Tuple of (y_new, t_new): integrand evaluations and time points
            for the replacement (split) steps.
        """
        raise NotImplementedError
    

    def _merge_excess_t(self, t, sum_steps, sum_step_errors, remove_idxs):
        """
        Merges neighboring time steps or removes and one time steps and extends
        its neighbor to cover the same range.

        Args:
            t (Tensor): Integration time steps
            remove_idxs (Tensor): First index of neighboring steps needed to be
                merged, or remove at given index and extend the following step
        
        Shapes:
            t : [N, C, T]
            removed_idxs : [n]
        """
        raise NotImplementedError
    

    def _error_norm(self, error: torch.Tensor) -> torch.Tensor:
        """
        Compute the RMS (root-mean-square) norm of per-dimension errors.

        For multi-dimensional integrands (D > 1), reduces the error across
        all output dimensions to a single scalar per step using the L2 norm.
        For 1D integrands, this is equivalent to torch.abs().

        Args:
            error: Per-dimension error values. Shape: [N, D] or [N].

        Returns:
            RMS error per step. Shape: [N].
        """
        return torch.sqrt(torch.mean(error**2, -1))
    

    def _get_new_eval_times(
            self,
            t: Optional[torch.Tensor],
            error_ratios: Optional[torch.Tensor] = None,
            t_init: Optional[torch.Tensor] = None,
            t_final: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which time points need new integrand evaluations.

        On the first call (t=None or error_ratios=None), creates an initial mesh
        and returns all evaluation points. On subsequent calls, identifies steps
        where error_ratio > 1 and returns their midpoints as new evaluation times.

        Args:
            t: Current time evaluations. Shape: [N, C, T], or None for initial call.
            error_ratios: Per-step error ratios from the last batch. Shape: [N].
                None on the first call.
            t_init: Lower integration bound. Shape: [T].
            t_final: Upper integration bound. Shape: [T].

        Returns:
            Tuple of (idxs_add, t_add):
                - idxs_add: Indices of steps to add/replace. Shape: [n_add].
                - t_add: Time points for new evaluations. Shape varies.
        """
        if t is None or error_ratios is None:
            if t is not None and len(t.shape) == 1:
                t = t.unsqueeze(-1)
            t_steps = self._initial_t_steps(
                t, t_init=t_init, t_final=t_final
            ).to(self.dtype)
            N, C, _ = t_steps.shape

            # Time points to evaluate, remove repetitive time points at the end
            # of each step to minimize evaluations
            t_add = torch.concatenate(
                [t_steps[0], t_steps[1:,1:].reshape((-1, *(t_steps.shape[2:])))],
                dim=0
            )
            idxs_add = torch.arange(N, device=self.device)
        else: 
            idxs_add = torch.where(error_ratios > 1.)[0]
            t_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
        
        return idxs_add, t_add
        

    def _get_initial_t_steps(
            self,
            t: Optional[torch.Tensor],
            t_init: torch.Tensor,
            t_final: torch.Tensor,
            inforce_endpoints: bool = False
        ) -> torch.Tensor:
        """
        Initialize the time mesh with C quadrature points per step.

        Takes raw time points and produces the full [N, C, T] tensor with
        quadrature points interpolated within each step. Optionally enforces
        that the mesh starts at t_init and ends at t_final.

        Args:
            t: Input time points. Shape: [N, T] (barriers only) or
                [N, C, T] (already has quadrature points). None to auto-generate.
            t_init: Lower integration bound. Shape: [T].
            t_final: Upper integration bound. Shape: [T].
            inforce_endpoints: If True, clips the mesh to [t_init, t_final]
                by removing/adjusting boundary steps.

        Returns:
            Time mesh with quadrature points. Shape: [N, C, T].
        """
        if t is not None and len(t.shape) == 1:
            t = t.unsqueeze(-1)
        if t is None or len(t.shape) != 3 or t.shape[1] != self.C:
            if t is not None:
                print(t.shape)
            t = self._initial_t_steps(
                t, t_init=t_init, t_final=t_final
            ).to(self.dtype)

        if inforce_endpoints:
            if t_init != t[0,0]:
                # Remove time steps where first point is less than t_init
                t = t[t[:,-1,0] > t_init[0]]
                # First step should start at t_init
                inp = torch.tensor([t_init.unsqueeze(0), t[0,-1].unsqueeze(0)], device=self.device)
                if t.shape[-1] == 1:
                    inp = inp.unsqueeze(-1)
                t[0] = self._initial_t_steps(
                    inp, t_init=t_init, t_final=t_final
                ).to(self.dtype)
            if t_final != t[-1,-1]:
                # Remove time steps where last point is greater than t_final
                t = t[t[:,0,0] < t_final[0]]
                # Last step should end at t_final
                inp = torch.tensor([t[-1,0].unsqueeze(0), t_final.unsqueeze(0)], device=self.device)
                if t.shape[-1] == 1:
                    inp = inp.unsqueeze(-1)
                t[-1] = self._initial_t_steps(
                    inp, t_init=t_init, t_final=t_final
                ).to(self.dtype)
        return t


    def _adaptively_add_steps(
            self,
            method_output: Optional[MethodOutput],
            error_ratios: torch.Tensor,
            y_step_eval: Optional[torch.Tensor],
            t_step_eval: Optional[torch.Tensor],
            t_step_barriers: torch.Tensor,
            t_step_barrier_idxs: torch.Tensor,
            t_step_trackers: torch.Tensor,
        ) -> Tuple[Optional[MethodOutput], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accept accurate steps and split inaccurate ones.

        This is the core adaptive refinement operation. For each evaluated step:
        - If error_ratio < 1.0: ACCEPT the step. Mark it as done in t_step_trackers.
          Keep its integral contribution.
        - If error_ratio >= 1.0: REJECT the step. Insert a new midpoint barrier
          between its start and end, splitting it into two smaller steps. These
          new steps will be evaluated in the next iteration.

        The midpoint barrier is placed at the average of the two neighboring
        barriers: t_new = (t_left + t_right) / 2.

        Args:
            method_output: RK results from the current batch (may be None when
                called during post-convergence optimization). If present, rejected
                steps are removed from it.
            error_ratios: Per-step error ratios. Shape: [N_batch].
            y_step_eval: Integrand evaluations for current batch. Shape: [N_batch, C, D].
            t_step_eval: Time points for current batch. Shape: [N_batch, C, T].
            t_step_barriers: All barrier positions. Shape: [M, T].
            t_step_barrier_idxs: Indices into t_step_barriers for the steps
                in the current batch. Shape: [N_batch].
            t_step_trackers: Boolean array tracking which steps need evaluation.
                Shape: [M].

        Returns:
            Tuple of (method_output, y_step_eval, t_step_eval,
            t_step_barriers_new, t_step_trackers_new, error_ratios_kept):
                - method_output: Updated with rejected steps removed.
                - y_step_eval: Kept evaluations only.
                - t_step_eval: Kept time points only.
                - t_step_barriers_new: Barriers with new midpoints inserted.
                - t_step_trackers_new: Updated tracker with new steps marked True.
                - error_ratios_kept: Error ratios for accepted steps only.
        """
        # Steps that pass the error tolerance are accepted (done)
        keep_mask = error_ratios < 1.0
        t_step_trackers[t_step_barrier_idxs[keep_mask]] = False

        # Steps that fail the error tolerance need to be split
        remove_mask = error_ratios >= 1.0
        N_t_add = torch.sum(remove_mask)
        # Allocate new barriers array with room for inserted midpoints
        t_step_barriers_new = torch.nan*torch.ones(
            (N_t_add + len(t_step_barriers), t_step_barriers.shape[-1]),
            dtype=self.dtype,
            device=self.device
        )
        t_step_trackers_new = torch.ones(
            N_t_add + len(t_step_barriers),
            dtype=bool,
            device=self.device
        )

        # Transfer existing barriers to their new positions in the expanded array.
        # Each rejected step causes a +1 offset for all subsequent barriers
        # (because a midpoint is being inserted). idx_offset tracks this shift.
        idx_offset = torch.zeros(
            len(t_step_barriers), dtype=torch.long, device=self.device
        )
        idx_offset[t_step_barrier_idxs[remove_mask]+1] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idxs_transfer = idx_offset\
            + torch.arange(len(t_step_barriers), device=self.device)
        t_step_barriers_new[idxs_transfer] = t_step_barriers.clone()
        t_step_trackers_new[idxs_transfer] = t_step_trackers.clone()

        # Insert new midpoint barriers between the start and end of rejected steps.
        # The midpoint is placed at (left_barrier + right_barrier) / 2.
        idxs_new = t_step_barrier_idxs[remove_mask]\
            + torch.arange(N_t_add, device=self.device) + 1
        t_add_barriers = 0.5*(
            t_step_barriers_new[idxs_new-1] + t_step_barriers_new[idxs_new+1]
        )
        t_step_barriers_new[idxs_new] = t_add_barriers
        assert torch.sum(torch.isnan(t_step_barriers_new)) == 0
        assert len(idxs_new) + len(idxs_transfer) == len(t_step_barriers_new)

        if method_output is not None:
            method_output.sum_steps = method_output.sum_steps[keep_mask]
            method_output.sum_step_errors = method_output.sum_step_errors[keep_mask]
            method_output.h = method_output.h[keep_mask]
            method_output.integral = torch.sum(method_output.sum_steps, 0)
            method_output.integral_error = torch.sum(method_output.sum_step_errors, 0)
        if y_step_eval is not None:
            y_step_eval = y_step_eval[keep_mask] 
        if t_step_eval is not None:
            t_step_eval = t_step_eval[keep_mask] 
        return method_output, y_step_eval, t_step_eval, t_step_barriers_new, t_step_trackers_new, error_ratios[keep_mask]


    def prune_excess_t(self, t, sum_steps, sum_step_errors, error_ratios_2steps):
        """
        Remove a single integration time step where
        error_ratios_2steps < remove_cut by merging two neighboring time steps,
        error_ratios_2steps corresponds to the first time step of the pair.
        This function only alters t, where remove_fxn merges the two steps.

        Args:
            t (Tensor): Current time evaluations in the path integral
            error_ratios_2steps (Tensor): The merged errors of neighboring time
                steps, these indices align with the first step of the pair
                (error_ratios_2steps[i] -> t[i])
        
        Shapes:
            t: [N, C, T]
            error_ratios_2steps: [N-1]
        """
        
        if len(error_ratios_2steps) == 0:
            return t, sum_steps, sum_step_errors
        # Since error ratios encompasses 2 RK steps each neighboring element shares
        # a step, we cannot remove that same step twice and therefore remove the 
        # first in pair of steps that it appears in
        ratio_idxs_cut = torch.where(
            self._rec_remove(error_ratios_2steps < self.remove_cut)
        )[0] # Index for first interval of 2
        assert not torch.any(ratio_idxs_cut[:-1] + 1 == ratio_idxs_cut[1:])

        if len(ratio_idxs_cut) == 0:
            return t, sum_steps, sum_step_errors
        
        return self._merge_excess_t(
            t, sum_steps, sum_step_errors, ratio_idxs_cut
        )

    def _get_optimal_t_step_barriers(
            self,
            record: Dict[str, torch.Tensor],
            t_step_barriers: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute an optimized time mesh from the converged integration results.

        After the integration loop converges, this method produces a refined
        mesh that can be reused for subsequent integrations of the same function.
        It does two things:
        1. Prunes: merges consecutive step pairs whose combined error ratio
           is below remove_cut (they were over-resolved).
        2. Adds: inserts midpoints for steps that still have high error ratios
           relative to the final integral value.

        This produces a mesh tailored to the difficulty of the integrand at
        different positions along the integration domain.

        Args:
            record: Dictionary of converged results including 't', 'sum_steps',
                'sum_step_errors', and 'integral'.
            t_step_barriers: Current barrier positions. Shape: [M, T].

        Returns:
            Optimized barrier positions. Shape: [M_opt, T].
        """
        # Prune steps with excess accuracy (over-resolved regions)
        _, error_ratios_2steps = self._compute_error_ratios(
            sum_step_errors=record['sum_step_errors'],
            sum_steps=record['sum_steps'],
            integral=record['integral'].detach()
        )
        t_pruned, sum_steps_pruned, sum_step_errors_pruned = self.prune_excess_t(
            record['t'],
            record['sum_steps'],
            record['sum_step_errors'],
            error_ratios_2steps
        )
        t_step_barriers_pruned = torch.concatenate(
            [t_pruned[:,0,:], t_step_barriers[-1].unsqueeze(0)],
            dim=0
        )

        # Add new t steps using converged integral value
        error_ratios, error_ratios_2steps = self._compute_error_ratios(
            sum_step_errors=sum_step_errors_pruned,
            sum_steps=sum_steps_pruned,
            integral=record['integral'].detach()
        )
        adaptive_step = self._adaptively_add_steps(
            method_output=None,
            error_ratios=error_ratios,
            y_step_eval=None,
            t_step_eval=None,
            t_step_barriers=t_step_barriers_pruned,
            t_step_barrier_idxs=torch.arange(
                len(t_step_barriers_pruned)-1, device=self.device
            ),
            t_step_trackers=torch.zeros(
                len(t_step_barriers_pruned), dtype=bool, device=self.device
            )
        ) 
        _, _, _, t_step_barriers_optimal, _, _ = adaptive_step

        return t_step_barriers_optimal


    def _rec_remove(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure no two adjacent True values exist in a boolean mask.

        When merging step pairs, two adjacent steps cannot both be merged
        (they share a boundary). This recursively resolves conflicts by
        keeping the first flagged step in any adjacent pair and un-flagging
        the second.

        Example: [True, True, False, True] -> [True, False, False, True]

        Args:
            mask: Boolean mask where True = flagged for removal. Shape: [N].

        Returns:
            Modified mask with no adjacent True values. Shape: [N].
        """
        
        mask2 = mask[:-1]*mask[1:]
        if not torch.any(mask2):
            return mask
        
        # Must keep the first integration step
        if mask2[0]:
            mask[1] = False
        
        # Mask is too small to remove points
        if len(mask) <= 2:
            return mask
        
        return self._rec_remove(torch.concatenate(
            [
                mask[:2],
                mask2[1:]*mask[:-2] + (~mask2[1:])*mask[2:]
            ]
        ))
        
        # if torch.any(mask2):
        #     if mask2[0]:
        #         mask[1] = False
        #     if len(mask) > 2:
        #         return self._rec_remove(torch.concatenate(
        #             [
        #                 mask[:2],
        #                 mask2[1:]*mask[:-2] + (~mask2[1:])*mask[2:]
        #             ]
        #         ))
        #     else:
        #         return mask
        # else:
        #     return mask


    def _compute_error_ratios(self, sum_step_errors, sum_steps=None, cum_sum_steps=None, integral=None):
        """
        Computes the ratio of the difference between chosen method of order p
        and a method of order p-1, and the error tolerance determined by atol,
        rtol, and the value of the integral. Integration steps of order p-1 
        use the same points.

        Args:
            sum_step_errors (Tensor): Similar to sum_steps but evaluated with
                and error tableau made of the differences between a method of
                order p and one of order p-1
            sum_steps (Tensor): Sum over all t and y evaluations in a single
                RK step multiplied by the total delta t for that step (h)
            Integral (Tensor): The evaluated path integral
        
        Shapes:
            sum_step_errors: [N, D]
            sum_steps: [N, D]
            integral: [D]
        """
        if self.error_calc_idx is not None:
            sum_step_errors = sum_step_errors[:,self.error_calc_idx, None]
            integral = integral[self.error_calc_idx, None]
            #y = y[:,:,self.error_calc_idx, None]
            if sum_steps is not None:
                sum_steps = sum_steps[:,self.error_calc_idx, None]
            if cum_sum_steps is not None:
                cum_sum_steps = cum_sum_steps[:,self.error_calc_idx, None]
                #DEBUG: add y0 to cum_steps to get get integral values at different times?

        if self.use_absolute_error_ratio:
            return self._compute_error_ratios_absolute(
                sum_step_errors, integral
            )
        else:
            return self._compute_error_ratios_cumulative(
                sum_step_errors, sum_steps=sum_steps, cum_sum_steps=cum_sum_steps
            )
    

    def _compute_error_ratios_absolute(self, sum_step_errors, integral):
        """
        Computes the ratio of the difference between chosen method of order p
        and a method of order p-1, and the error tolerance determined by atol,
        rtol, and the value of the integral. Integration steps of order p-1 
        use the same points.

        Args:
            sum_step_errors (Tensor): Similar to sum_steps but evaluated with
                and error tableau made of the differences between a method of
                order p and one of order p-1
            Integral (Tensor): The evaluated path integral
        
        Shapes:
            sum_step_errors: [N, D]
            integral: [D]
        """
        error_tol = self.atol + self.rtol*torch.abs(integral)
        error_estimate = torch.abs(sum_step_errors)
        error_ratio = self._error_norm(error_estimate/error_tol)

        error_estimate_2steps = error_estimate[:-1] + error_estimate[1:]
        error_ratio_2steps= self._error_norm(
            error_estimate_2steps/error_tol
        )
        
        return error_ratio, error_ratio_2steps   
    
    
    def _compute_error_ratios_cumulative(self, sum_step_errors, sum_steps=None, cum_sum_steps=None):
        """
        Computes the ratio of the difference between chosen method of order p
        and a method of order p-1, and the error tolerance determined by atol,
        rtol, and the value of the integral up to the current step. This method
        is more similar to ODE error calculation methods but is less suitable
        for path integrals where the total integral is known. Integration
        steps of order p-1 use the same points.

        Args:
            sum_step_errors (Tensor): Similar to sum_steps but evaluated with
                and error tableau made of the differences between a method of
                order p and one of order p-1
            sum_steps (Tensor): Sum over all t and y evaluations in a single
                RK step multiplied by the total delta t for that step (h)
        
        Shapes:
            sum_steps: [N, D]
            sum_step_errors: [N, D]
        """
        if cum_sum_steps is not None:
            cum_steps = cum_sum_steps
        elif sum_steps is not None:
            cum_steps = torch.cumsum(sum_steps, dim=0)
        else:
            raise ValueError("Must give sum_steps or cum_sum_steps")
        error_estimate = torch.abs(sum_step_errors)
        error_tol = self.atol + self.rtol*torch.abs(cum_steps)
        error_ratio = self._error_norm(error_estimate/error_tol).abs()

        error_estimate_2steps = error_estimate[:-1] + error_estimate[1:]
        error_tol_2steps = self.atol + self.rtol*torch.max(
            torch.stack(
                [cum_steps[:-1].abs(), cum_steps[1:].abs()]
            ),
            dim=0
        )[0]
        error_ratio_2steps= self._error_norm(
            error_estimate_2steps/error_tol_2steps
        ).abs() 
        
        return error_ratio, error_ratio_2steps
 
    def _get_sorted_indices(
            self,
            record: torch.Tensor,
            result: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute indices for merging new results into an existing sorted record.

        Uses binary search (searchsorted) to find where new results should be
        inserted, then computes the indices for both the existing and new entries
        in the merged array.

        Args:
            record: Sorted 1D tensor of existing values (e.g. start times of
                recorded steps). Shape: [N_record].
            result: New values to insert. Shape: [N_result].

        Returns:
            Tuple of (idxs_keep, idxs_input):
                - idxs_keep: Where existing record entries go in the merged array.
                - idxs_input: Where new result entries go in the merged array.
        """
        idxs_sorted = torch.searchsorted(record, result)
        idxs_input = idxs_sorted + torch.arange(len(result), device=self.device)
        idxs_keep = torch.arange(len(result) + len(record), device=self.device)
        keep_mask = torch.ones(len(idxs_keep), device=self.device).to(bool)
        keep_mask[idxs_input] = False
        idxs_keep = idxs_keep[keep_mask]
        return idxs_keep, idxs_input
 
    def _insert_sorted_results(
            self,
            record: torch.Tensor,
            record_idxs: torch.Tensor,
            result: torch.Tensor,
            result_idxs: torch.Tensor
        ) -> torch.Tensor:
        """
        Merge new results into an existing record at pre-computed sorted positions.

        Creates a new tensor large enough for both, places existing entries at
        record_idxs and new entries at result_idxs.

        Args:
            record: Existing recorded data. Shape: [N_record, ...].
            record_idxs: Positions for existing data in merged array. Shape: [N_record].
            result: New data to insert. Shape: [N_result, ...].
            result_idxs: Positions for new data in merged array. Shape: [N_result].

        Returns:
            Merged tensor with both record and result entries in sorted order.
            Shape: [N_record + N_result, ...].
        """
        add_shape = (len(record) + len(result), *record.shape[1:])
        old_record = record.clone()
        record = torch.nan*torch.ones(
            add_shape, device=self.device, dtype=self.dtype
        )
        record[record_idxs] = old_record
        record[result_idxs] = result
        #assert torch.sum(torch.isnan(record)) == 0
        return record
   
    def _record_results(
            self,
            record: Dict[str, torch.Tensor],
            take_gradient: bool,
            results: IntegralOutput
        ) -> Dict[str, torch.Tensor]:
        """
        Add a batch of accepted step results to the running record.

        On the first batch, initializes the record dict. On subsequent batches,
        inserts new results in time-sorted order and accumulates the integral
        and loss. When take_gradient is True, detaches results to prevent
        the computation graph from growing across batches.

        Args:
            record: Running record dict. Empty dict {} on the first call.
            take_gradient: Whether gradients are being computed. If True,
                detaches tensors before storing to keep graph manageable.
            results: IntegralOutput from the current accepted batch.

        Returns:
            Updated record dict with the new results merged in.
        """
        if len(record) == 0 and not take_gradient:
            record['integral'] = results.integral
            record['t'] = results.t
            record['h'] = results.h
            record['y'] = results.y
            record['sum_steps'] = results.sum_steps
            record['sum_step_errors'] = results.sum_step_errors
            record['integral_error'] = results.integral_error
            record['error_ratios'] = results.error_ratios
            record['loss'] = results.loss
            return record
        elif len(record) == 0 and take_gradient:
            record['integral'] = results.integral.detach()
            record['t'] = results.t.detach()
            record['h'] = results.h.detach()
            record['y'] = results.y.detach()
            record['sum_steps'] = results.sum_steps.detach()
            record['sum_step_errors'] = results.sum_step_errors.detach()
            record['integral_error'] = results.integral_error.detach()
            record['error_ratios'] = results.error_ratios.detach()
            record['loss'] = results.loss.detach()
            return record 

        idxs_keep, idxs_input = self._get_sorted_indices(
            record['t'][:,0,0].detach(), results.t[:,0,0].detach()
        )       
        for key in record.keys():
            is_number = len(record[key].shape) == 1 and record[key].shape[0] == 1 
            is_number = is_number or len(record[key].shape) == 0
            if 'integral' in key or 'loss' in key:
                record[key] = record[key] + getattr(results, key).detach()
            else:
                record[key] = self._insert_sorted_results(
                    record[key], idxs_keep, getattr(results, key), idxs_input
                )
        assert torch.all(record['t'][1:,0,0] - record['t'][:-1,0,0] > 0)

        return record

    def _sort_record(self, record: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Sort all per-step entries in the record by ascending time.

        The integration loop may process batches in any order, so the record
        needs to be sorted before final output. Scalar values (integral, loss)
        are not reordered since they are cumulative sums.

        Args:
            record: Record dict with per-step tensors.

        Returns:
            Record with per-step tensors sorted by start time of each step.
        """
        sorted_idxs = torch.argsort(record['t'][:,0,0], dim=0)
        for key in record.keys():
            if 'loss' not in key and 'integral' not in key:
                record[key] = record[key][sorted_idxs]
        all_ascending = torch.all(record['t'][1:,0,0] - record['t'][:-1,0,0] > 0)
        all_descending = torch.all(record['t'][1:,0,0] - record['t'][:-1,0,0] < 0)
        assert all_ascending or all_descending, "Times are required to be either in ascending or descending order"
        return record

    def _get_cpu_memory(self) -> Tuple[float, float]:
        """
        Query available and total CPU (system) memory in GB.

        Returns:
            Tuple of (free_gb, total_gb).
        """
        mem = psutil.virtual_memory()
        free = mem.available/1024**3
        total = mem.total/1024**3
        return free, total

    def _get_cuda_memory(self) -> Tuple[float, float]:
        """
        Query available and total CUDA (GPU) memory in GB.

        Accounts for both free system GPU memory and unused PyTorch cache
        memory (reserved but not allocated). This gives a more accurate
        picture of how much memory is truly available for new allocations.

        Returns:
            Tuple of (free_gb, total_gb).
        """
        mem_info = torch.cuda.mem_get_info(self.device)
        # Total memory on the GPU
        total_gpu = mem_info[1]/1024**3
        # Memory that is free outside of the PyTorch cache
        free_gpu = mem_info[0]/1024**3
        # Memory reserved for the PyTorch cache
        torch_cache = torch.cuda.memory_reserved(self.device)/1024**3
        # Cache memory being used by tensors
        torch_cache_used = torch.cuda.memory_allocated(self.device)/1024**3
        # Total free amount of memory that can be used
        total_free = free_gpu + (torch_cache - torch_cache_used)
        
        return total_free, total_gpu
    
    def _get_memory(self) -> Tuple[float, float]:
        """
        Query available and total memory in GB for the active device type.

        Dispatches to _get_cuda_memory() or _get_cpu_memory() based on
        self.device_type.

        Returns:
            Tuple of (free_gb, total_gb).
        """
        if self.device_type == 'cuda':
            return self._get_cuda_memory()
        else:
            return self._get_cpu_memory()

    def _setup_memory_checks(
            self,
            ode_fxn: Callable,
            t_test: torch.Tensor,
            ode_args: tuple = ()
        ) -> None:
        """
        Benchmark the integrand's memory footprint to determine batch sizes.

        Runs the integrand with increasing batch sizes (10, 100, 1000, ...)
        and measures the memory consumed per evaluation. This per-evaluation
        memory cost (ode_unit_mem_size) is then used throughout integration
        to dynamically compute how many steps can fit in one batch.

        A 2.1x safety factor is applied to the measured memory to account
        for intermediate allocations during integration (RK computation,
        error estimation, etc.).

        Args:
            ode_fxn: The integrand function to benchmark.
            t_test: A sample time point for benchmarking. Shape: [T] or [1, T].
            ode_args: Extra arguments passed to ode_fxn.
        """
        assert len(t_test.shape) <= 2
        if len(t_test.shape) == 2:
            t_test = t_test[0]
        t_test = t_test.unsqueeze(0)
        self.ode_unit_mem_size = None
        
        N = 10
        max_evals = 2*N
        eval_time = 0
        while eval_time < 0.1 and N < 1e9 and N < max_evals:
            t0 = time.time()
            t_input = torch.tile(t_test, (N, 1))
            mem_before = self._get_memory()
            if self.ode_unit_mem_size is not None:
                if self.ode_unit_mem_size*N > mem_before[0]:
                    return
            result = ode_fxn(t_input, *ode_args)
            mem_after = self._get_memory()
            del result
            self.ode_unit_mem_size = 2.1*max(0, (mem_before[0] - mem_after[0])/float(N))
            eval_time = time.time() - t0
            N = 10*N
            max_evals = self._get_max_ode_evals(0.8)
        print("Ending unit memory search")

    def _get_usable_memory(self, total_mem_usage: float) -> float:
        """
        Compute how much memory (in GB) can be used for integrand evaluations.

        Reserves a buffer of (1 - total_mem_usage) * total_memory to avoid
        out-of-memory errors from other system/PyTorch allocations.

        Args:
            total_mem_usage: Fraction of total memory allowed (0 < value <= 1).

        Returns:
            Usable memory in GB (non-negative).
        """
        free, total = self._get_memory()
        buffer = (1 - total_mem_usage)*total
        return max(0, free - buffer)
    
    def _get_max_ode_evals(self, total_mem_usage: float) -> int:
        """
        Compute the maximum number of integrand evaluations that fit in memory.

        Divides usable memory by the per-evaluation memory cost (measured by
        _setup_memory_checks). A small epsilon (1e-12) prevents division by zero.

        Args:
            total_mem_usage: Fraction of total memory allowed.

        Returns:
            Maximum number of evaluations (integer).
        """
        usable = self._get_usable_memory(total_mem_usage)
        return int(usable//(1e-12 + self.ode_unit_mem_size))

    def integrate(
            self,
            ode_fxn: Optional[Callable] = None,
            y0: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
            t_init: Optional[torch.Tensor] = None,
            t_final: Optional[torch.Tensor] = None,
            N_init_steps: int = 13,
            ode_args: tuple = (),
            take_gradient: bool = False,
            total_mem_usage: Optional[float] = None,
            loss_fxn: Optional[Callable] = None,
            max_batch: Optional[int] = None,
            verbose: bool = False,
            verbose_speed: bool = False,
        ) -> IntegralOutput:
        """
        Perform parallel adaptive numerical integration of ode_fxn.

        This is the main integration loop. It divides [t_init, t_final] into
        steps using barrier points, evaluates the integrand in parallel batches,
        and adaptively refines the mesh until all steps meet the error tolerance.

        The algorithm:
        1. Initialize barriers (random mesh or user-provided).
        2. While unevaluated steps remain:
           a. Select a batch of steps that fits in memory.
           b. Place C quadrature points in each step.
           c. Evaluate the integrand at all points in parallel.
           d. Compute integral contributions and error estimates via RK.
           e. Accept steps with error_ratio < 1; split the rest.
           f. Record accepted results.
        3. Optimize the final mesh (prune over-resolved + refine under-resolved).
        4. Return the integral, error, and diagnostics.

        Args:
            ode_fxn: The integrand f(t). Takes shape [N, T], returns [N, D].
                If None, uses the function from construction.
            y0: Initial integral accumulator value. Shape: [D].
            t: Optional initial step barriers. If provided, these are the
                starting mesh. If None, a random mesh is generated. Shape: [N, T].
            t_init: Lower integration bound. Shape: [T].
            t_final: Upper integration bound. Shape: [T].
            N_init_steps: Approximate number of initial steps when t is None.
                The actual count is ~sqrt(N_init_steps) segments with
                ~sqrt(N_init_steps)+1 random sub-barriers each.
            ode_args: Extra arguments passed to ode_fxn after the time tensor.
            take_gradient: If True, calls loss.backward() after each batch
                to compute gradients through the integration.
            total_mem_usage: Fraction of memory to use for batching. Overrides
                the value from construction if provided.
            loss_fxn: Custom loss function. Takes an IntegralOutput, returns a
                scalar tensor. If None, uses the integral value itself.
            max_batch: Maximum evaluations per batch. Overrides dynamic memory
                calculation if provided.
            verbose: Print error ratios and debug info during integration.
            verbose_speed: Print timing info for each sub-operation.

        Returns:
            IntegralOutput with the computed integral, error estimates, time
            mesh (t), optimized mesh (t_optimal), and diagnostics.

        Note:
            If t is provided, the solver uses these as initial barriers. Steps
            may be split or merged, but the bounds [t[0], t[-1]] are preserved.
            If t is None, a random initial mesh is generated in [t_init, t_final].
        """

        # Set dtype based on input
        self.set_dtype_by_input(t=t, t_init=t_init, t_final=t_init)

        # If t is given set t_init and t_final, else use input, else use save values
        if t is not None:
            assert len(t.shape) == 2
            t = t.to(self.dtype).to(self.device)
            if t_init is not None or t_final is not None:
                assert t_init is not None and t_final is not None,\
                    "Must provide both 't_init' and 't_final' or leave them both as None"
                t_init = t_init.to(self.dtype).to(self.device)
                t_final = t_final.to(self.dtype).to(self.device)
                assert torch.allclose(t[0], t_init, atol=self.atol_assert, rtol=self.rtol_assert)
                assert torch.allclose(t[-1], t_final, atol=self.atol_assert, rtol=self.rtol_assert)
            t_init = t[0]
            t_final = t[-1]
            assert t_init < t_final,\
                "Integrator requires t_init < t_final, consider switching them and multiplying the integral by -1. Please also consider effects to your ode_fxn."
        else:
            t_init = self.t_init if t_init is None else t_init
            t_final = self.t_final if t_final is None else t_final
        t_init = t_init.to(self.dtype).to(self.device)
        t_final = t_final.to(self.dtype).to(self.device)

        # Replace max_batch if default it given
        max_batch = self.max_batch if max_batch is None else max_batch

        
        # Get variables or populate with default values, send to correct device
        ode_fxn, t_init, t_final, y0 = self._check_variables(
            ode_fxn, t_init, t_final, y0
        )
        assert t_init < t_final, "Integrator requires t_init < t_final, consider switching them and multiplying the integral by -1. Please also consider the effects your loss function if one is provided."
        total_mem_usage = self.total_mem_usage if total_mem_usage is None\
            else total_mem_usage
        assert total_mem_usage <= 1.0 and total_mem_usage > 0,\
            "total_mem_usage is a ratio and must be 0 < total_mem_usage <= 1"
        # Check if this is the same integrand as the previous call (for warm-starting)
        same_ode_fxn = ode_fxn.__name__ == self.previous_ode_fxn
        # Benchmark memory footprint on first call with a new integrand
        if not same_ode_fxn and max_batch is None:
            self._setup_memory_checks(ode_fxn, t_init, ode_args=ode_args)
        assert self._get_max_ode_evals(total_mem_usage) > (2*self.Cm1 + 1),\
            "Not enough free memory to run 2 integration steps, consider increasing total_mem_usage"
        loss_fxn = loss_fxn if loss_fxn is not None else self._integral_loss

        # Make sure ode_fxn exists and provides the correct output
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        test_output = ode_fxn(
            torch.tensor([[t_init]], dtype=self.dtype, device=self.device),
            *ode_args
        ) 
        assert len(test_output.shape) >= 2
        del test_output
        
        # Load latest evaluation if it exists and values are unspecified
        if t is None and same_ode_fxn:
            #TODO: CHECK THIS PART WITH MULTI DIM T
            assert self.t_step_barriers_previous is not None
            mask = (self.t_step_barriers_previous[:,0] <= t_final[0])\
                + (self.t_step_barriers_previous[:,0] >= t_init[0])
            t_step_barriers = self.t_step_barriers_previous[mask]
            if not torch.all(t_step_barriers[0] == t_init):
                t_step_barriers = torch.concatenate(
                    [t_init.unsqueeze(0), t_step_barriers], dim=0
                )
            if not torch.all(t_step_barriers[-1] == t_final):
                t_step_barriers = torch.concatenate(
                    [t_step_barriers, t_init.unsqueeze(0)], dim=0
                )
            
        if t is None:
            # Generate a random initial mesh of barriers across [t_init, t_final].
            # Uses sqrt(N_init_steps) evenly-spaced segments, each subdivided by
            # random sub-barriers. Randomization avoids edge issues that occur when
            # barriers align with features of the integrand.
            t_is_given = False
            N_even_t = torch.sqrt(torch.tensor(N_init_steps, dtype=torch.float)).to(torch.int)
            dt = (t_final - t_init)/N_even_t
            t_step_barriers = t_init + dt * torch.arange(N_even_t, device=self.device)[:,None,None] #TODO: this assumes time is 1d

            # Create random sub-barriers within each segment and sort them
            random_ts = dt*torch.rand(
                (N_even_t, N_even_t + 1, 1), device=self.device
            )
            random_ts = torch.sort(random_ts, dim=1)[0]
            t_step_barriers = t_step_barriers + random_ts
            # Enforce exact start and end points
            t_step_barriers[0] += t_init - t_step_barriers[0,0]
            t_step_barriers[-1] += t_final - t_step_barriers[-1,-1]
            # Flatten segments into a single sorted barrier array
            t_step_barriers = torch.flatten(t_step_barriers, start_dim=0, end_dim=1)
            t_step_barriers[0] = t_init
            t_step_barriers[-1] = t_final
            assert torch.all(t_step_barriers[1:] - t_step_barriers[:-1] > 0)
        else:
            t_is_given = True
            t_step_barriers = t
        t_step_trackers = torch.ones(len(t_step_barriers), device=self.device).to(bool)
        t_step_trackers[-1] = False # t_final cannot be a step starting point
        """
        t_steps_init = self._get_initial_t_steps(
            t, t_init, t_final, inforce_endpoints=True
        )
        """
        t, y = None, None

        record = {}
        # === Main integration loop ===
        # Continues until all steps have been evaluated and accepted
        # (t_step_trackers[i] == False for all i)
        while torch.any(t_step_trackers):
            # Determine how many steps fit in one batch based on memory
            if max_batch is not None:
                max_steps = max_batch//self.C
            else:
                max_steps = int(self._get_max_ode_evals(total_mem_usage)//self.C)

            if y is not None:
                assert max_steps >= len(y), f"{max_steps}  {len(y)}"

            # --- Step 1: Select a batch of unevaluated steps ---
            # Find barrier indices where t_step_trackers is True, take up to max_steps
            step_idxs = torch.arange(len(t_step_barriers), device=self.device)
            step_idxs = step_idxs[t_step_trackers]
            step_idxs = step_idxs[:max_steps]
            # Place C quadrature points within each selected step
            t_step_eval = self._t_step_interpolate(
                t_step_barriers[step_idxs], t_step_barriers[step_idxs+1]
            )
            t_flat = torch.flatten(t_step_eval, start_dim=0, end_dim=1)
            assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert >= 0)
            error_ratios=None

            # --- Step 2: Evaluate the integrand at all quadrature points ---
            # Flatten [N, C, T] -> [N*C, T] for batch evaluation, then reshape back
            N, C, T = t_step_eval.shape
            y_step_eval = ode_fxn(
                torch.flatten(t_step_eval, start_dim=0, end_dim=-2),
                *ode_args
            )
            y_step_eval = torch.reshape(y_step_eval, (N, C, -1))

            # --- Step 3: Compute integral contributions via RK formula ---
            t0 = time.time()
            method_output = self._calculate_integral(
                t_step_eval,
                y_step_eval,
                y0=torch.zeros(1, device=self.device, dtype=self.dtype)
            )
            if len(record) == 0:
                # First batch: integral is just this batch's contribution
                current_integral = method_output.integral.detach()
                all_sum_steps = method_output.sum_steps.detach()
                cum_sum_steps = torch.cumsum(all_sum_steps, 0)
            else:
                # Subsequent batches: add to previously recorded integral
                current_integral = record['integral'] + method_output.integral.detach()
                # Merge new steps into the sorted record to compute cumulative sums
                idxs_keep, idxs_input = self._get_sorted_indices(
                    record['t'][:,0,0], t_step_eval[:,0,0]
                )
                all_sum_steps = self._insert_sorted_results(
                    record['sum_steps'], idxs_keep, method_output.sum_steps, idxs_input
                )
                cum_sum_steps = torch.cumsum(all_sum_steps, 0)[idxs_input]
            if verbose_speed: print("\t calc integrals 1", time.time() - t0)

            # --- Step 4: Compute error ratios for each step ---
            t0 = time.time()
            error_ratios, error_ratios_2steps = self._compute_error_ratios(
                sum_step_errors=method_output.sum_step_errors,
                cum_sum_steps=cum_sum_steps,
                integral=current_integral,
            )
            if verbose_speed: print("\t calculate errors", time.time() - t0)
            assert len(y_step_eval) == len(error_ratios)
            assert len(y_step_eval) - 1 == len(error_ratios_2steps), f" y: {y_step_eval.shape} | ratios: {error_ratios_2steps.shape} | t: {t.shape}"
            if verbose:
                print("ERROR1", error_ratios)
                print("ERROR2", error_ratios_2steps)

            # Early exit if too many steps fail and user-provided mesh is given
            if t_is_given and self.max_path_change is not None:
                fail_ratio = torch.sum(error_ratios > 1.).to(float)/len(error_ratios)
                if fail_ratio >= self.max_path_change:
                    print(f"WARNING: {fail_ratio*100}% of integration steps failed error requirements, which is greater than max_path_change ({self.max_path_change}), now exiting.")
                    return None

            # --- Step 5: Adaptive refinement ---
            # Split steps with error_ratio >= 1, keep steps with error_ratio < 1,
            # and update barriers/trackers accordingly
            method_output, y_step_eval, t_step_eval,\
            t_step_barriers, t_step_trackers, error_ratios = self._adaptively_add_steps(
                method_output=method_output,
                error_ratios=error_ratios,
                y_step_eval=y_step_eval,
                t_step_eval=t_step_eval,
                t_step_barriers=t_step_barriers,
                t_step_barrier_idxs=step_idxs,
                t_step_trackers=t_step_trackers
            )
            # Verify barrier ordering after adaptive refinement
            t_step_barriers_diff = t_step_barriers[1:,0] - t_step_barriers[:-1,0]
            assert (
                torch.all(t_step_barriers_diff + self.atol_assert > 0)\
                or torch.all(t_step_barriers_diff - self.atol_assert < 0)
            )

            # --- Step 6: Record accepted results and handle gradients ---
            if t_step_eval.shape[0] > 0:
                take_gradient = torch.any(t_step_trackers) or take_gradient
                intermediate_results = IntegralOutput(
                    integral=method_output.integral,
                    loss=None,
                    gradient_taken=take_gradient,
                    t=t_step_eval,
                    h=method_output.h,
                    y=y_step_eval,
                    sum_steps=method_output.sum_steps,
                    integral_error=method_output.integral_error,
                    sum_step_errors=torch.abs(method_output.sum_step_errors),
                    error_ratios=error_ratios,
                    t_init=t_init,
                    t_final=t_final,
                    y0=0
                )

                #TODO make sure growing string loss center is a time not the number of evals because eval number is meaningless here.
                # Compute loss and accumulate into the record
                loss = loss_fxn(intermediate_results)
                intermediate_results.loss = loss
                record = self._record_results(
                    record=record,
                    take_gradient=take_gradient,
                    results=intermediate_results
                )

                # Backpropagate gradients through the integration if requested
                if take_gradient:
                    if loss.requires_grad:
                        loss.backward()
            del y_step_eval

        # === Post-convergence: sort results and optimize the mesh ===
        record = self._sort_record(record)
        # Prune over-resolved steps and refine under-resolved ones
        t_step_barriers_optimal = self._get_optimal_t_step_barriers(
            record, t_step_barriers
        )
        # Cache results for warm-starting subsequent calls with the same integrand
        self.t_step_barriers_previous = t_step_barriers_optimal
        self.previous_ode_fxn = ode_fxn.__name__

        return IntegralOutput(
            integral=record['integral'],
            loss=record['loss'],
            gradient_taken=take_gradient,
            t_optimal=t_step_barriers_optimal,
            t=record['t'],
            h=record['h'],
            y=record['y'],
            sum_steps=record['sum_steps'],
            integral_error=record['integral_error'],
            sum_step_errors=torch.abs(record['sum_step_errors']),
            error_ratios=record['error_ratios'],
        )


class ParallelUniformAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
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
            t_init=None,
            t_final=None
        ):
        Creates an initial time sampling tensor either from scratch or from a
        tensor of time points with dimension d.

        Args:
            t (Tensor): Input time, either None or tensor starting and
                ending at the integration bounds
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
        
        Shapes:
            t : [N, T] will populate intermediate evaluations according to
                integration method; [N, C, T] will return t if C is the same
                as the number of evaluations per step, otherwise it will create
                C steps between the first and last values in the second dim 
            t_init: [T]
            t_final: [T]
        
        # Get variables or populate with default values, send to correct device
        _, t_init, t_final, _ = self._check_variables(
            None, t_init, t_final, None
        )
        if t is None:
            t = torch.linspace(0, 1., 7*self.Cm1 + 1, device=self.device).unsqueeze(-1)
            t = t_init + t*(t_final - t_init)
        elif len(t.shape) == 3:
            if t.shape[1] == self.C:
                return t
            else:
                if len(t) > 1:
                    print(t[:,:,0])
                    assert torch.allclose(t[:-1,-1], t[1:,0], atol=self.atol_assert, rtol=self.rtol_assert)
                t = t[:,torch.tensor([0,-1], dtype=torch.int, device=self.device),:]
                t = torch.flatten(t, start_dim=0, end_dim=1)
        return self._t_step_interpolate(t[:-1], t[1:])
    """
 

    def _t_step_interpolate(self, t_left: torch.Tensor, t_right: torch.Tensor) -> torch.Tensor:
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
        steps = self.method.tableau.c.unsqueeze(-1)*dt
        return t_left.unsqueeze(1) + steps


    def _evaluate_adaptive_y(
            self,
            ode_fxn: Callable,
            idxs_add: torch.Tensor,
            y: torch.Tensor,
            t: torch.Tensor,
            ode_args: tuple = ()
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split failed steps at their midpoints and evaluate the integrand.

        Each step that failed the error check is divided into two sub-steps
        at its midpoint. Fresh quadrature points are placed in each sub-step
        using the tableau's c values, and the integrand is evaluated at all
        new points.

        Args:
            ode_fxn: The integrand function f(t). Takes [N, T], returns [N, D].
            idxs_add: Indices of steps that need splitting. Shape: [R].
            y: Current integrand evaluations (unused here, present for
                interface compatibility with the variable solver). Shape: [N, C, D].
            t: Current quadrature point positions. Shape: [N, C, T].
            ode_args: Extra arguments passed to ode_fxn.

        Returns:
            Tuple of (y_add, t_eval_steps) where:
                - y_add: Integrand values at new quadrature points.
                    Shape: [2*R, C, D] (two sub-steps per split step).
                - t_eval_steps: New quadrature point positions.
                    Shape: [2*R, C, T].
        """
        D = t.shape[-1]
        # Compute the midpoint of each failed step
        t_mid = (t[idxs_add,-1] + t[idxs_add,0])/2.
        # Build left and right boundaries for the two new sub-steps
        t_left = torch.concatenate(
            [t[idxs_add,None,0], t_mid[:,None]], dim=1
        )
        t_right = torch.concatenate(
            [t_mid[:,None], t[idxs_add,None,-1]], dim=1
        )
        # Place quadrature points in each sub-step and evaluate
        t_eval_steps = self._t_step_interpolate(
            t_left.view(-1,D), t_right.view(-1, D)
        )
        y_add = ode_fxn(t_eval_steps.view(-1, D), *ode_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", C=self.C)
        return y_add, t_eval_steps
        
    
    def _merge_excess_t(
            self,
            t: torch.Tensor,
            sum_steps: torch.Tensor,
            sum_step_errors: torch.Tensor,
            remove_idxs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of consecutive over-resolved steps into single larger steps.

        For uniform sampling, merging replaces two consecutive steps [A, B]
        with a single step spanning [A.start, B.end]. New quadrature points
        are placed at the standard tableau c positions within the merged step.
        The integral contributions and errors of both steps are summed.

        Args:
            t: Quadrature point positions for all steps. Shape: [N, C, T].
            sum_steps: Integral contribution of each step. Shape: [N, D].
            sum_step_errors: Error estimate of each step. Shape: [N, D].
            remove_idxs: Indices of the first step in each pair to merge.
                The pair (remove_idxs[i], remove_idxs[i]+1) is merged.
                Shape: [R].

        Returns:
            Tuple of (t_pruned, sum_steps_pruned, sum_step_errors_pruned)
            with the merged steps replacing the original pairs. Each has
            N-R entries along the first dimension.
        """
        if len(remove_idxs) == 0 or len(t) == 1:
            return t, sum_steps, sum_step_errors

        # Create merged steps spanning from the left edge of the first step
        # to the right edge of the second step, with new c-based quadrature points
        t_replace = self._t_step_interpolate(
            t[remove_idxs,0], t[remove_idxs+1,-1]
        )

        # Sum integral contributions and errors of the merged pair
        sum_steps_replace = sum_steps[remove_idxs] + sum_steps[remove_idxs+1]
        sum_step_errors_replace = sum_step_errors[remove_idxs] + sum_step_errors[remove_idxs+1]

        # Remove the first step of each pair from the arrays
        remove_mask = torch.ones(len(t), device=self.device, dtype=torch.bool)
        remove_mask[remove_idxs] = False
        t_pruned = t[remove_mask]
        sum_steps_pruned = sum_steps[remove_mask]
        sum_step_errors_pruned = sum_step_errors[remove_mask]

        # Place the merged step data at the position of the second (kept) step,
        # adjusting indices to account for earlier removals shifting positions
        remove_idxs_shifted = remove_idxs - torch.arange(
            len(remove_idxs), device=self.device
        )
        t_pruned[remove_idxs_shifted] = t_replace
        sum_steps_pruned[remove_idxs_shifted] = sum_steps_replace
        sum_step_errors_pruned[remove_idxs_shifted] = sum_step_errors_replace

        # Verify time ordering is preserved after merging
        t_pruned_flat = torch.flatten(t_pruned, start_dim=0, end_dim=1)
        assert torch.all(t_pruned_flat[1:] - t_pruned_flat[:-1] + self.atol_assert >= 0)
        t_flat = torch.flatten(t, start_dim=0, end_dim=1)
        assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert>= 0)

        return t_pruned, sum_steps_pruned, sum_step_errors_pruned
    


class ParallelVariableAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    """
    Parallel solver using variable-sampling Runge-Kutta methods.

    In variable sampling, quadrature points within each step can be at
    arbitrary positions (not fixed by the tableau). The b weights are
    computed dynamically based on the actual positions of the points via
    the method's ``tableau_b(c)`` function.

    This flexibility is especially useful during adaptive refinement: when
    a step is split, the existing evaluation points from the original step
    can be reused in the sub-steps (they just end up at different fractional
    positions within the new, smaller steps). This avoids redundant function
    evaluations.

    When merging two steps, the combined points are subsampled at evenly
    spaced indices to fit C points, and the b weights are recomputed for
    their new fractional positions.

    Supported methods: 'adaptive_heun', 'generic3'.

    Attributes:
        method: The variable method instance with a ``tableau_b(c)`` method.
        order: Convergence order of the RK method.
        C: Number of quadrature points per step.
        Cm1: C - 1, used for indexing.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.method_name in VARIABLE_METHODS, \
            f"Cannot find method '{self.method_name}' in supported methods: {list(VARIABLE_METHODS.keys())}"
        self.method = VARIABLE_METHODS[self.method_name]()
        self.order = self.method.order
        self.C = self.method.n_tableau_c
        self.Cm1 = self.C - 1
    
    """
    def _initial_t_steps(
            self,
            t,             
            t_init=None,
            t_final=None
        ):
        Creates an initial time sampling tensor either from scratch or from a
        tensor of time points with dimension d.

        Args:
            t (Tensor): Input time, either None or tensor starting and
                ending at the integration bounds
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
        
        Shapes:
            t : [N, T] will populate intermediate evaluations according to
                integration method; [N, C, T] will return t if C is the same
                as the number of evaluations per step, otherwise it will create
                C steps between the first and last values in the second dim 
            t_init: [T]
            t_final: [T]
 
        # Get variables or populate with default values, send to correct device
        _, t_init, t_final, _ = self._check_variables(
            t_init=t_init, t_final=t_final
        )
        if t is None:
            t = torch.linspace(0, 1., 7, device=self.device).unsqueeze(-1)
            t = t_init + t*(t_final - t_init)
            t_left = t[:-1]
            t_right = t[1:]
        elif len(t.shape) == 2:
            t_left = t[:-1]
            t_right = t[1:]
        elif t.shape[1] == self.C:
            return t
        else:
            if len(t) > 1:
                assert torch.allclose(t[:-1,-1], t[1:,0], atol=self.atol_assert, rtol=self.rtol_assert)
            t_left = t[:,0] 
            t_right = t[:,-1] 
            #steps = torch.tile(
            #    torch.arange(self.C)[None,:,None], (len(t), 1, t.shape[-1])
            #)
            #return t[:,0,:] + steps*(t[:,-1,:] - t[:,0,:])/self.Cm1
        steps = torch.arange(self.C, device=self.device)[None,:,None]/self.Cm1
        t_left = t_left.unsqueeze(1)
        t_right = t_right.unsqueeze(1)
        return t_left + steps*(t_right - t_left)
        print("here", t.shape, t)
        _t = torch.reshape(t[:-1], (-1, self.Cm1, 1))
        _t_ends = torch.concatenate([_t[1:,0], t[None,-1]]).unsqueeze(1)
        return torch.concatenate([_t, _t_ends], dim=1)
    """


    def _evaluate_adaptive_y(
            self,
            ode_fxn: Callable,
            idxs_add: torch.Tensor,
            y: torch.Tensor,
            t: torch.Tensor,
            ode_args: tuple = ()
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split failed steps and reuse existing evaluations where possible.

        Unlike the uniform solver which discards old evaluations, the variable
        solver can reuse them. New evaluation points are placed at the midpoints
        between each pair of consecutive existing points. The original and new
        points are then interleaved and reshaped into two sub-steps of C points
        each.

        For a step with C=3 points [p0, p1, p2], this creates midpoints
        [m01, m12] and interleaves them: [p0, m01, p1, m12, p2, p2]. The
        duplicate at position C acts as the shared boundary point. This is
        then reshaped into two sub-steps: [p0, m01, p1] and [m12, p2, p2].

        Args:
            ode_fxn: The integrand function f(t). Takes [N, T], returns [N, D].
            idxs_add: Indices of steps that need splitting. Shape: [R].
            y: Current integrand evaluations, reused in sub-steps. Shape: [N, C, D].
            t: Current quadrature point positions. Shape: [N, C, T].
            ode_args: Extra arguments passed to ode_fxn.

        Returns:
            Tuple of (y_add_combined, t_add_combined) where:
                - y_add_combined: Integrand values for the new sub-steps,
                    interleaving old and new evaluations. Shape: [2*R, C, D].
                - t_add_combined: Time positions for the new sub-steps.
                    Shape: [2*R, C, T].
        """
        # Compute midpoints between each pair of consecutive quadrature points
        t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
        # Evaluate the integrand at the new midpoints
        y_add = ode_fxn(t_steps_add.view(-1, t.shape[-1]), *ode_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", N=len(idxs_add))
        D = y_add.shape[-1]

        # Build index arrays to interleave old points (even positions) and
        # new midpoints (odd positions) into a 2*C-length array
        select_prev_idxs = torch.arange(self.C, device=self.device)*2
        select_prev_idxs[select_prev_idxs>=self.C] += 1
        select_add_idxs = torch.arange(self.Cm1, device=self.device)*2 + 1
        select_add_idxs[select_add_idxs>=self.C] += 1

        # Interleave old and new time points into a combined array
        t_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (self.C)*2, D),
            dtype=self.dtype,
            device=self.device
        )
        t_add_combined[:,select_prev_idxs] = t[idxs_add]
        t_add_combined[:,select_add_idxs] = t_steps_add
        # Duplicate the boundary point so both sub-steps share it
        t_add_combined[:,self.C] = t_add_combined[:,self.C-1]
        # Reshape from [R, 2*C] into [2*R, C] (two sub-steps per original step)
        t_add_combined = torch.reshape(
            t_add_combined, (len(idxs_add)*2, self.C, D)
        )

        # Interleave old and new integrand values the same way
        y_add_combined = torch.nan*torch.ones(
            (len(idxs_add), self.C*2, D),
            dtype=self.dtype,
            device=self.device
        )
        y_add_combined[:,select_prev_idxs] = y[idxs_add]
        y_add_combined[:,select_add_idxs] = y_add
        y_add_combined[:,self.C] = y_add_combined[:,self.C-1]
        y_add_combined = torch.reshape(
            y_add_combined, (len(idxs_add)*2, self.C, D)
        )

        return y_add_combined, t_add_combined


    def _merge_excess_t(
            self,
            t: torch.Tensor,
            sum_steps: torch.Tensor,
            sum_step_errors: torch.Tensor,
            remove_idxs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge pairs of consecutive over-resolved steps into single larger steps.

        For variable sampling, merging concatenates the quadrature points from
        both steps (C + C-1 = 2C-1 points, since the shared boundary is not
        duplicated from the second step), then subsamples every other point to
        get back to C points. Because the points are now at non-standard
        fractional positions within the merged step, the variable method's
        ``tableau_b(c)`` will recompute appropriate weights on the next evaluation.

        Args:
            t: Quadrature point positions for all steps. Shape: [N, C, T].
            sum_steps: Integral contribution of each step. Shape: [N, D].
            sum_step_errors: Error estimate of each step. Shape: [N, D].
            remove_idxs: Indices of the first step in each pair to merge.
                The pair (remove_idxs[i], remove_idxs[i]+1) is merged.
                Shape: [R].

        Returns:
            Tuple of (t_pruned, sum_steps_pruned, sum_step_errors_pruned)
            with the merged steps replacing the original pairs. Each has
            N-R entries along the first dimension.
        """
        if len(remove_idxs) == 0 or len(t) == 1:
            return t, sum_steps, sum_step_errors
        t_flat = torch.flatten(t, start_dim=0, end_dim=1)
        assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert >= 0)

        # Concatenate points from both steps (skip first point of second step
        # since it equals the last point of the first step), giving 2C-1 points
        combined_steps = torch.concatenate(
            [t[remove_idxs,:], t[remove_idxs+1,1:]], dim=1
        )
        sum_steps_replace = sum_steps[remove_idxs] + sum_steps[remove_idxs+1]
        sum_step_errors_replace = sum_step_errors[remove_idxs] + sum_step_errors[remove_idxs+1]
        # Subsample every other point to reduce 2C-1 back to C points
        keep_idxs = torch.arange(self.C, dtype=torch.long, device=self.device)*2

        # Remove the first step of each pair from the arrays
        remove_mask = torch.ones(len(t), dtype=torch.bool, device=self.device)
        remove_mask[remove_idxs] = False
        t_pruned = t[remove_mask]
        sum_steps_pruned = sum_steps[remove_mask]
        sum_step_errors_pruned = sum_step_errors[remove_mask]

        # Place the merged step data at the position of the second (kept) step,
        # adjusting indices to account for earlier removals shifting positions
        update_idxs = remove_idxs - torch.arange(
            len(remove_idxs), device=self.device
        )
        t_pruned[update_idxs] = combined_steps[:,keep_idxs]
        sum_steps_pruned[update_idxs] = sum_steps_replace
        sum_step_errors_pruned[update_idxs] = sum_step_errors_replace

        # Verify time ordering and step continuity after merging
        t_pruned_flat = torch.flatten(t_pruned, start_dim=0, end_dim=1)
        assert torch.all(t_pruned_flat[1:] - t_pruned_flat[:-1] + self.atol_assert >= 0)
        assert np.allclose(t_pruned[:-1,-1,:], t_pruned[1:,0,:], atol=self.atol_assert, rtol=self.rtol_assert)

        return t_pruned, sum_steps_pruned, sum_step_errors_pruned