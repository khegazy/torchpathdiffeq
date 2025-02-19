import psutil
import time
import torch
import numpy as np
from einops import rearrange
from dataclasses import asdict as dataclass_asdict

from .methods import _get_method, UNIFORM_METHODS, VARIABLE_METHODS
from .base import SolverBase, IntegralOutput, steps

class ParallelAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, remove_cut=0.1, use_absolute_error_ratio=True, error_calc_idx=None, *args, **kwargs):
        """
        Args:
        remove_cut (float): Cut to remove integration steps with error ratios
            less than this value, must be < 1
        use_absolute_error_ratio (bool): Use the total integration value when
            calulating the error ratio for adding or removing points, otherwise
            use integral value up to the time step being evaluated
        """
        
        super().__init__(*args, **kwargs)
        assert remove_cut < 1.
        self.remove_cut = remove_cut
        self.use_absolute_error_ratio = use_absolute_error_ratio

        self.method = None
        self.order = None
        self.C = None
        self.Cm1 = None
        self.t_step_barriers_previous = None
        self.previous_ode_fxn = None
        self.ode_eval_unit_size = None
        self.error_calc_idx = error_calc_idx


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
                integration method, [N, C, T] will return t
            t_init: [T]
            t_final: [T]
        raise NotImplementedError
    """
    

    def _evaluate_adaptive_y(self, ode_fxn, idxs_add, y, t, ode_args=()):
        raise NotImplementedError
    

    def _merge_excess_t(self, t, remove_idxs):
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
    

    def _error_norm(self, error):
        """
        Normalize multivariate errors to determine the step's total error
        """
        return torch.sqrt(torch.mean(error**2, -1))
    

    def _get_new_eval_times(self, t, error_ratios=None, t_init=None, t_final=None):

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
        
        print("new times", t_add.shape)
        print("\t", t_add)
        return idxs_add, t_add
        

    def _get_initial_t_steps(self, t, t_init, t_final, inforce_endpoints=False):
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


    def _add_initial_y(
            self,
            ode_fxn,
            t_steps_add,
            ode_args=(),
        ):
        """
        Initial evaluation of ode_fxn for the given t points, if t is None it
        will be initialized within the range given by t_init and t_final

        Args:
            ode_fxn (Callable): The ode function to be integrated
            t (Tensor): Initial time evaluations in the path integral, either
                None or tensor starting and ending at the integration bounds
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
            ode_args (Tuple): Arguments for ode_fxn
        
        Shapes:
            t : [N, C, T]
            t_init: [T]
            t_final: [T]
        
        Notes:
            ode_fxn takes as input (t, *args)
        """
        """
        if t is not None and len(t.shape) == 1:
            t = t.unsqueeze(-1)
        t_steps = self._initial_t_steps(
            t, t_init=t_init, t_final=t_final
        ).to(torch.float64)
        N, C, _ = t_steps.shape

        """
        N, C, _ = t_steps_add.shape 
        # Time points to evaluate, remove repetitive time points at the end
        # of each step to minimize evaluations
        t_add = torch.concatenate(
            [
                t_steps_add[0],
                t_steps_add[1:,1:].reshape((-1, *(t_steps_add.shape[2:])))
            ],
            dim=0
        )
        # Calculate new geometries
        y_add = ode_fxn(t_add, *ode_args).to(self.dtype)
        _, D = y_add.shape

        # Add repetitive y values to the end of each integration step
        y_steps = torch.reshape(y_add[1:], (N, C-1, D))
        y = torch.concatenate(
            [
                torch.concatenate(
                    [y_add[None,None,0,...], y_steps[:-1,None,-1]], dim=0
                ),
                y_steps
            ],
            dim=1
        )
        return y, t_steps_add
        """
        else:
            assert len(t_add) % self.Cm1 == 0
            y = torch.concatenate(
                [
                    y,
                    torch.concatenate(
                        [
                            torch.concatenate(
                                [y[None,None,-1,-1], y_add[:-1,-1,None]], dim=0
                            ),
                            y_add
                        ],
                        dim=1
                    )
                ],
                dim=0
            )
        """

    def _adaptively_add_y(
            self,
            method_output,
            error_ratios,
            y_step_eval,
            t_step_eval,
            t_step_barriers,
            t_step_barrier_idxs,
            t_step_trackers,
        ):
        keep_mask = error_ratios < 1.0
        t_step_trackers[t_step_barrier_idxs[keep_mask]] = False
        method_output.sum_steps = method_output.sum_steps[keep_mask]
        method_output.sum_step_errors = method_output.sum_step_errors[keep_mask]
        method_output.h = method_output.h[keep_mask]
        method_output.integral = torch.sum(method_output.sum_steps, 0)
        method_output.integral_error = torch.sum(method_output.sum_step_errors, 0)

        remove_mask = error_ratios >= 1.0
        N_t_add = torch.sum(remove_mask)
        t_step_barriers_new = torch.nan*torch.ones(
            (N_t_add + len(t_step_barriers), t_step_barriers.shape[-1]),
            dtype=t_step_barriers.dtype,
            device=self.device
        )
        t_step_trackers_new = torch.ones(
            N_t_add + len(t_step_barriers),
            dtype=bool,
            device=self.device
        )


        # Transfer over previous time points
        idx_offset = torch.zeros(
            len(t_step_barriers), dtype=torch.long, device=self.device
        )
        idx_offset[t_step_barrier_idxs[remove_mask]+1] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idxs_transfer = idx_offset\
            + torch.arange(len(t_step_barriers), device=self.device)
        t_step_barriers_new[idxs_transfer] = t_step_barriers.clone()
        t_step_trackers_new[idxs_transfer] = t_step_trackers.clone()
        
        # Add new time points between old ones
        idxs_new = t_step_barrier_idxs[remove_mask]\
            + torch.arange(N_t_add, device=self.device) + 1
        t_add_barriers = 0.5*(
            t_step_barriers_new[idxs_new-1] + t_step_barriers_new[idxs_new+1] 
        )
        t_step_barriers_new[idxs_new] = t_add_barriers
        assert torch.sum(torch.isnan(t_step_barriers_new)) == 0
        assert len(idxs_new) + len(idxs_transfer) == len(t_step_barriers_new)

        return method_output, y_step_eval[keep_mask], t_step_eval[keep_mask], t_step_barriers_new, t_step_trackers_new, error_ratios[keep_mask]


    def _adaptively_add_y_orig(
            self,
            ode_fxn,
            y,
            t,
            idxs_add,
            ode_args=()
        ):
        """
        Adds new time points between current time points and splits these
        poinsts into two steps where error_ratio < 1. ode_fxn is evaluated at
        these new time points, both the new time points and y points are merged
        with the original values in order of time.

        Args:
            ode_fxn (Callable): The ode function to be integrated
            y (Tensor): Evaluations of ode_fxn at time points t
            t (Tensor): Current time evaluations in the path integral
            error_ratios (Tensor): Numerical integration error ratios for each
                integration step
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
            ode_args (Tuple): Arguments for ode_fxn
        
        SHAPES:
            y: [N, C, D]
            t : [N, C, T]
            error_ratios : [N]
            t_init: [T]
            t_final: [T]
        
        Notes:
            ode_fxn takes as input (t, *args)
        """

        if idxs_add is None or len(idxs_add) == 0:
            return y, t

   
        """
        # Get variables or populate with default values, send to correct device
        _, t_init, t_final, _ = self._check_variables(t_init=t_init, t_final=t_final)

        if y is None or t is None or error_ratios is None:
            return self._add_initial_y(ode_fxn=ode_fxn, ode_args=ode_args, t=t, t_init=t_init, t_final=t_final)

        if torch.all(error_ratios <= 1.):
            return y, t
        """
        # Get new time steps for merged steps
        #idxs_add = torch.where(error_ratios > 1.)[0]
        #t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
        # Calculate new geometries
        n_add = len(idxs_add)
        # ode_fxn input is 2 dims, t_steps_add has 3 dims, combine first two
        y_add, t_steps_add = self._evaluate_adaptive_y(ode_fxn, idxs_add, y, t, ode_args)
        D = y_add.shape[-1]
        #print("Y T OUT", y_add.shape, t_steps_add.shape)
        #print("T ADD", t_steps_add[:,:,0])
        #print("Y ADD", y_add[:,:,0])
        """
        t_steps_add = rearrange(t_steps_add, 'n c d -> (n c) d')
        y_add = ode_fxn(t_steps_add, *ode_args).to(torch.float64)
        t_steps_add = rearrange(t_steps_add, '(n c) d -> n c d', n=n_add, c=self.Cm1) 
        y_add = rearrange(y_add, '(n c) d -> n c d', n=n_add, c=self.Cm1) 
        """

        # Create new vector to fill with old and new values
        y_combined = torch.zeros(
            (len(y)+n_add, self.Cm1+1, D),
            dtype=self.dtype,
            requires_grad=False,
            device=self.device
        ).detach()
        t_combined = torch.zeros(
            (len(y)+n_add, self.Cm1+1, t.shape[-1]),
            dtype=self.dtype,
            requires_grad=False,
            device=self.device
        ).detach()
        
        # Add old t and y values, skipping regions with new points
        idx_offset = torch.zeros(len(y), dtype=torch.long, device=self.device)
        idx_offset[idxs_add] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idx_input = torch.arange(len(y), device=self.device) + idx_offset
        y_combined[idx_input,:] = y
        t_combined[idx_input,:] = t

        # Add new t and y values to added rows
        idxs_add_offset = idxs_add + torch.arange(
            len(idxs_add), device=self.device
        )
        select_idxs = torch.arange(len(idxs_add), device=self.device)*2
        #t_add_combined = torch.nan*torch.ones(
        #    (len(idxs_add), (self.Cm1+1)*2-1, D),
        #    dtype=torch.float64,
        #    device=self.device
        #)
        #t_add_combined[:,torch.arange(self.Cm1+1)*2] = t[idxs_add]
        #t_add_combined[:,torch.arange(self.Cm1)*2+1] = t_steps_add
        #t_combined[idxs_add_offset,:,:] = t_add_combined[:,:self.Cm1+1]
        #t_combined[idxs_add_offset+1,:,:] = t_add_combined[:,self.Cm1:]
        t_combined[idxs_add_offset,:,:] = t_steps_add[select_idxs]
        t_combined[idxs_add_offset+1,:,:] = t_steps_add[select_idxs+1]

        #y_add_combined = torch.nan*torch.ones(
        #    (len(idxs_add), (self.Cm1+1)*2-1, D),
        #    dtype=torch.float64,
        #    device=self.device
        #)
        #y_add_combined[:,torch.arange(self.Cm1+1)*2] = y[idxs_add]
        #y_add_combined[:,torch.arange(self.Cm1)*2+1] = y_add
        #y_combined[idxs_add_offset,:,:] = y_add_combined[:,:self.Cm1+1]
        #y_combined[idxs_add_offset+1,:,:] = y_add_combined[:,self.Cm1:]
        y_combined[idxs_add_offset,:,:] = y_add[select_idxs]
        y_combined[idxs_add_offset+1,:,:] = y_add[select_idxs+1]

        #print("TIMES", t_combined[:,:,0])
        #print("Y", y_combined[:,:,0])
        #print("DT", t_combined[1:,0,0]-t_combined[:-1,-1,0])
        assert torch.allclose(
            t_combined[:-1,-1], t_combined[1:,0],
            atol=self.atol_assert, rtol=self.rtol_assert
        )
        return y_combined, t_combined


        if y is None:
            return self._add_initial_y(
                ode_fxn=ode_fxn,
                t_steps_add=t_add,
                ode_args=ode_args
            )
        else:
            return self._interweave_evaluations(
                ode_fxn=ode_fxn,
                y=y,
                t=t,
                idxs_add=idxs_add,
                t_steps_add=t_add,
                ode_args=ode_args
            )
 
    def prune_excess_t(self, t, error_ratios_2steps):
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
            return t# torch.empty(0, dtype=torch.long)
        # Since error ratios encompasses 2 RK steps each neighboring element shares
        # a step, we cannot remove that same step twice and therefore remove the 
        # first in pair of steps that it appears in
        ratio_idxs_cut = torch.where(
            self._rec_remove(error_ratios_2steps < self.remove_cut)
        )[0] # Index for first interval of 2
        assert not torch.any(ratio_idxs_cut[:-1] + 1 == ratio_idxs_cut[1:])

        if len(ratio_idxs_cut) == 0:
            return t
        
        t_pruned = self._merge_excess_t(t, ratio_idxs_cut)
        return t_pruned


    def _rec_remove(self, mask):
        """
        Make no neighboring values are both True by setting the second value
        to False, this is done recursively.
        """
        mask2 = mask[:-1]*mask[1:]
        if torch.any(mask2):
            if mask2[0]:
                mask[1] = False
            if len(mask) > 2:
                return self._rec_remove(torch.concatenate(
                    [
                        mask[:2],
                        mask2[1:]*mask[:-2] + (~mask2[1:])*mask[2:]
                    ]
                ))
            else:
                return mask
        else:
            return mask


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
        
        print("IN CUMSUM ERORR")
        return error_ratio, error_ratio_2steps
 
    def _get_sorted_indices(self, record, result):
        idxs_sorted = torch.searchsorted(record, result)
        idxs_input = idxs_sorted + torch.arange(len(result), device=self.device)
        idxs_keep = torch.arange(len(result) + len(record))
        keep_mask = torch.ones(len(idxs_keep), device=self.device).to(bool)
        keep_mask[idxs_input] = False
        idxs_keep = idxs_keep[keep_mask]
        return idxs_keep, idxs_input
 
    def _insert_sorted_results(self, record, record_idxs, result, result_idxs):
        add_shape = (len(record) + len(result), *record.shape[1:])
        old_record = record.clone()
        record = torch.nan*torch.ones(
            add_shape, device=self.device, dtype=old_record.dtype
        )
        record[record_idxs] = old_record
        record[result_idxs] = result
        assert torch.sum(torch.isnan(record)) == 0
        return record
   
    def _record_results(self, record, take_gradient, results):
        if len(record) == 0 and not take_gradient:
            record['integral'] = results.integral
            #record['t_pruned'] = results.t_pruned
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
            #record['t_pruned'] = results.t_pruned.detach()
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
        result_dict = dataclass_asdict(results)
        for key in record.keys():
            is_number = len(record[key].shape) == 1 and record[key].shape[0] == 1 
            is_number = is_number or len(record[key].shape) == 0
            if 'integral' in key or 'loss' in key:
                record[key] = record[key] + result_dict[key].detach()
            else:
                record[key] = self._insert_sorted_results(
                    record[key], idxs_keep, result_dict[key], idxs_input
                )
        assert torch.all(record['t'][1:,0,0] - record['t'][:-1,0,0] > 0)
        """
        record['integral'] = record['integral']\
            + results.integral.detach()
        #record['t_pruned'] = torch.concatenate(
        #    [record['t_pruned'], results.t_pruned.detach()], dim=0
        #)
        record['t'] = torch.concatenate(
            [record['t'], results.t.detach()], dim=0
        )
        record['h'] = torch.concatenate(
            [record['h'], results.h.detach()], dim=0
        )
        record['y'] = torch.concatenate(
            [record['y'], results.y.detach()], dim=0
        )
        record['sum_steps'] = torch.concatenate(
            [record['sum_steps'], results.sum_steps.detach()], dim=0
        )
        record['sum_step_errors'] = torch.concatenate(
            [record['sum_step_errors'], results.sum_step_errors.detach()],
            dim=0
        )
        record['integral_error'] = record['integral_error']\
            + results.integral_error.detach()
        record['error_ratios'] = torch.concatenate(
            [record['error_ratios'], results.error_ratios.detach()], dim=0
        )
        record['loss'] = record['loss'] + results.loss.detach()
        """

        return record

    def _sort_record(self, record):
        sorted_idxs = torch.argsort(record['t'][:,0,0], dim=0)
        for key in record.keys():
            if len(record[key].shape) > 0 and len(record[key]) > 1:
                record[key] = record[key][sorted_idxs]
        all_ascending = torch.all(record['t'][1:,0,0] - record['t'][:-1,0,0] > 0)
        all_descending = torch.all(record['t'][1:,0,0] - record['t'][:-1,0,0] < 0)
        assert all_ascending or all_descending
        return record

    def _get_cpu_memory(self):
        mem = psutil.virtual_memory()
        free = mem.available/1024**3
        total = mem.total/1024**3
        return free, total

    def _get_cuda_memory(self):
        free = torch.cuda.mem_get_info()[0]/1024**3
        total = torch.cuda.mem_get_info()[1]/1024**3
        return free, total
    
    def _get_memory(self):
        if self.device_type == 'cuda':
            return self._get_cuda_memory()
        else:
            return self._get_cpu_memory()

    def _setup_memory_checks(self, ode_fxn, t_test, ode_args=()):
        assert len(t_test.shape) <= 2
        if len(t_test.shape) == 2:
            t_test = t_test[0]
        t_test = t_test.unsqueeze(0)
        self.ode_unit_mem_size = None
        
        N = 1
        eval_time = 0
        while eval_time < 0.1 and N < 1e9:
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

    def _get_usable_memory(self, total_mem_usage): 
        free, total = self._get_memory()
        buffer = (1 - total_mem_usage)*total
        return max(0, free - buffer)
    
    def _get_max_ode_evals(self, total_mem_usage):
        usable = self._get_usable_memory(total_mem_usage)
        #print("Usable", usable, int(usable//(1e-12 + self.ode_unit_mem_size)))
        return int(usable//(1e-12 + self.ode_unit_mem_size))

    def integrate(
            self,
            ode_fxn=None,
            y0=None,
            t=None,
            t_init=None,
            t_final=None,
            N_init_steps=13,
            ode_args=(),
            take_gradient=None,
            total_mem_usage=0.9,
            loss_fxn=None,
            verbose=True,
            verbose_speed=False,
        ):
        """
        Perform the parallel numerical path integral on ode_fxn over a path
        parameterized by time (t), which ranges from t_init to t_final.

        Args:
            ode_fxn (Callable): The function to integrate over along the path
                parameterized by t
            y0 (Tensor): Initial value of the integral
            t (Tensor): Initial time points to evaluate ode_fxn and perform the
                numerical integration over
            t_init (Tensor): Initial integration time points
            t_final (Tensor): Final integration time points
            ode_args (Tuple): Extra arguments provided to ode_fxn
            verbose (bool): Print derscriptive messages about the evaluation
            verbose_speed (bool): Time integration subprocesses and print
        
        Shapes:
            y0: [D]
            t: [N, C, T] or for [N, T] the intermediate time points will be 
                calculated
            t_init: [T]
            t_final: [T]
        
        Note:
            If t is None it will be initialized with a few integration steps
            in the range [t_init, t_final]. If t is specified integration steps
            between t[0] and t[-1] may be removed and/or added, but the bounds
            of integration will remain [t[0], t[-1]]. If t is 2 dimensional the 
            intermediate time points will be calculated.
        """
        # If t is given it must be consistent with given t_init and t_final
        t_init = t_init if t_init is None else t_init.to(self.dtype)
        t_final = t_final if t_final is None else t_final.to(self.dtype)

        if t is not None:
            if len(t.shape) == 2:
                assert torch.all(t[1:] + self.atol_assert > t[:-1])
                if self.Cm1 > 1:
                    error_message = f"When giving t (with 2 dims) the first dimension (N) must satisfy N % {self.Cm1} = 1 in order to be properly divided into integration steps for the {self.method_name} method."
                    assert len(t) % self.Cm1 == 1, error_message
            if len(t.shape) == 3:
                error_message = f"When giving t (with 3 dims) the second dimension must match the integration method's number of samples per step. For {self.method_name} it's {self.C}"
                assert t.shape[1] == self.C, error_message
                assert torch.all(
                    torch.flatten(t, 0, 1)[1:] + self.atol_assert >= torch.flatten(t, 0, 1)[:-1]
                )
            if t_init is None:
                t_init = t[0,0]
            else:
                assert torch.allclose(t[0,0], t_init, atol=self.atol_assert, rtol=self.rtol_assert)
            if t_final is None:
                t_final = t[-1,-1]
            else:
                assert torch.allclose(t[-1,-1], t_final, atol=self.atol_assert, rtol=self.rtol_assert)
        #assert t_init < t_final, "Integrator requires t_init < t_final, consider switching them and multiplying the integral by -1. Please also consider the effects your loss function if one is provided."
        
        # Get variables or populate with default values, send to correct device
        ode_fxn, t_init, t_final, y0 = self._check_variables(
            ode_fxn, t_init, t_final, y0
        )
        assert t_init < t_final, "Integrator requires t_init < t_final, consider switching them and multiplying the integral by -1. Please also consider the effects your loss function if one is provided."
        assert total_mem_usage <= 1.0 and total_mem_usage > 0,\
            "total_mem_usage is a ratio and must be 0 < total_mem_usage <= 1"
        same_ode_fxn = ode_fxn.__name__ == self.previous_ode_fxn
        if not same_ode_fxn:
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
            if not np.all(t_step_barriers[0] == t_init):
                t_step_barriers = torch.concatenate(
                    [t_init.unsqueeze(0), t_step_barriers], dim=0
                )
            if not np.all(t_step_barriers[-1] == t_final):
                t_step_barriers = torch.concatenate(
                    [t_step_barriers, t_init.unsqueeze(0)], dim=0
                )
            
        if t is None:
            t_step_barriers = t_init\
                + torch.arange(N_init_steps+1).unsqueeze(-1)*(t_final - t_init)/N_init_steps
        else:
            t_step_barriers = t[:,0]
        t_step_trackers = torch.ones(len(t_step_barriers), device=self.device).to(bool)
        t_step_trackers[-1] = False # t_final cannot be a step starting point
        """
        t_steps_init = self._get_initial_t_steps(
            t, t_init, t_final, inforce_endpoints=True
        )
        """
        t, y = None, None

        #print("T STEPS INIT", t_init.shape, t_init, t_step_barriers.shape, t_step_barriers)
        record = {}
        while torch.any(t_step_trackers):
            """
            # Initial time should start at the end of the previous batch
            _t_init = t_init if t is None else t[-1,-1]
            #if t is not None:
            #    print("PREV T", t[:,:,0])
            #print("T INIT", _t_init)
            t_steps = self._get_initial_t_steps(
                t_steps_init, _t_init, t_final, inforce_endpoints=True
            )
            max_steps = int(0.8*self._get_max_ode_evals(total_mem_usage)//self.C)
            t_steps = t_steps[:min(len(t_steps), max_steps)]
            #print("INIT TADD PORTION", t_add.shape, t_steps.shape)
            #print("TADD INIT", t_add[:,:,0])
            #t = None
            y, t = self._add_initial_y(
                ode_fxn=ode_fxn, t_steps_add=t_steps, ode_args=ode_args
            )
            """
            max_steps = int(self._get_max_ode_evals(total_mem_usage)//self.C)
            if y is not None:
                assert max_steps >= len(y), f"{max_steps}  {len(y)}"
            
            # Gather integration steps
            step_idxs = torch.arange(len(t_step_barriers), device=self.device)
            step_idxs = step_idxs[t_step_trackers]
            step_idxs = step_idxs[:max_steps]
            t_step_eval = self._t_step_interpolate(
                t_step_barriers[step_idxs], t_step_barriers[step_idxs+1]
            )
            t_flat = torch.flatten(t_step_eval, start_dim=0, end_dim=1)
            assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert >= 0)
            error_ratios=None

            N, C, D = t_step_eval.shape
            y_step_eval = ode_fxn(
                torch.flatten(t_step_eval, start_dim=0, end_dim=-2),
                *ode_args
            )
            y_step_eval = torch.reshape(y_step_eval, (N, C, D))
            
            # Evaluate integral
            t0 = time.time()
            method_output = self._calculate_integral(
                t_step_eval, y_step_eval, y0=torch.zeros(1, device=self.device)
            )
            if len(record) == 0:
                current_integral = method_output.integral.detach()
                all_sum_steps = method_output.sum_steps.detach()
                cum_sum_steps = torch.cumsum(all_sum_steps, 0)
            else:
                current_integral = record['integral'] + method_output.integral.detach() 
                idxs_keep, idxs_input = self._get_sorted_indices(
                    record['t'][:,0,0], t_step_eval[:,0,0]
                )
                all_sum_steps = self._insert_sorted_results(
                    record['sum_steps'], idxs_keep, method_output.sum_steps, idxs_input
                )
                cum_sum_steps = torch.cumsum(all_sum_steps, 0)[idxs_input]
            if verbose_speed: print("\t calc integrals 1", time.time() - t0)
            #print("METHOD OUTPUT", t_step_eval.shape, method_output.sum_steps.shape)
            
            # Calculate error
            t0 = time.time()
            if self.error_calc_idx is None:
                sum_step_errors = method_output.sum_step_errors
            else:
                sum_step_errors =\
                    method_output.sum_step_errors[:,self.error_calc_idx, None]
            error_ratios, error_ratios_2steps = self._compute_error_ratios(
                sum_step_errors=sum_step_errors,
                cum_sum_steps=cum_sum_steps,
                integral=current_integral,
            )
            if verbose_speed: print("\t calculate errors", time.time() - t0)
            assert len(y_step_eval) == len(error_ratios)
            assert len(y_step_eval) - 1 == len(error_ratios_2steps), f" y: {y_step_eval.shape} | ratios: {error_ratios_2steps.shape} | t: {t.shape}"
            if verbose:
                print("ERROR1", error_ratios)
                print("ERROR2", error_ratios_2steps)
            
            # Remove components with large errors
                
            #print("BEFORE ADAPTIVE", t_step_eval.shape, t_step_barriers.shape, t_step_trackers.shape, y_step_eval.shape, error_ratios.shape)
            #print(t_step_barriers[:,0])
            method_output, y_step_eval, t_step_eval,\
            t_step_barriers, t_step_trackers, error_ratios = self._adaptively_add_y(
                method_output=method_output,
                error_ratios=error_ratios,
                y_step_eval=y_step_eval,
                t_step_eval=t_step_eval,
                t_step_barriers=t_step_barriers,
                t_step_barrier_idxs=step_idxs,
                t_step_trackers=t_step_trackers
            )
            t_step_barriers_diff = t_step_barriers[1:,0] - t_step_barriers[:-1,0]
            assert (
                torch.all(t_step_barriers_diff + self.atol_assert > 0)\
                or torch.all(t_step_barriers_diff - self.atol_assert < 0) 
            )
            #print("AFTER ADAPTIVE", t_step_eval.shape, t_step_barriers.shape, t_step_trackers.shape, y_step_eval.shape, error_ratios.shape)
            #print(t_step_barriers[:,0])
            if t_step_eval.shape[0] > 0:
                intermediate_results = IntegralOutput(
                    integral=method_output.integral,
                    loss=None,
                    gradient_taken=take_gradient,
                    #t_pruned=t_pruned,
                    t=t_step_eval,
                    h=method_output.h,
                    y=y_step_eval,
                    sum_steps=method_output.sum_steps,
                    integral_error=method_output.integral_error,
                    sum_step_errors=torch.abs(method_output.sum_step_errors),
                    error_ratios=error_ratios,
                    #t_init=t_init,
                    #t_final=t_final,
                    y0=0
                )

                #TODO make sure growing string loss center is a time not the number of evals because eval number is meaningless here.
                # Calculate loss
                loss = loss_fxn(intermediate_results)
                intermediate_results.loss = loss
                """
                    integral=method_output.integral,
                    t=t,
                    h=method_output.h,
                    sum_steps=method_output.sum_steps,
                    integral_error=method_output.integral_error,
                    errors=torch.abs(method_output.sum_step_errors),
                    error_ratios=error_ratios
                )
                """
                # Add results to record
                record = self._record_results(
                    record=record,
                    take_gradient=take_gradient,
                    results=intermediate_results
                )
    
                if take_gradient:
                    if loss.requires_grad:
                        loss.backward()
                    #del y
                    #y = None
                #print(t[:,:,0])
                #print(y[:,:,0])
                #print("LOSS!!!!!", loss)
        
        # Prune t of excess sampling
        record = self._sort_record(record)
        # TODO: Change to absolute and add new t points as errors have changed with full integral
        _, error_ratios_2steps = self._compute_error_ratios(
            sum_step_errors=record['sum_step_errors'],
            sum_steps=record['sum_steps'],
            integral=record['integral'].detach()
        )
        t_pruned = self.prune_excess_t(record['t'], error_ratios_2steps)
        #print("REMOVING T", record['t'].shape, error_ratios_2steps.shape, t_pruned.shape)

        self.previous_ode_fxn = ode_fxn.__name__
        t_step_barriers_pruned = torch.concatenate(
            [t_pruned[:,0,:], t_step_barriers[-1].unsqueeze(0)],
            dim=0
        )
        self.t_step_barriers_previous = t_step_barriers_pruned

        return IntegralOutput(
            integral=record['integral'],
            loss=record['loss'],
            gradient_taken=take_gradient,
            t_pruned=t_step_barriers_pruned,
            t=record['t'],
            h=record['h'],
            y=record['y'],
            sum_steps=record['sum_steps'],
            integral_error=record['integral_error'],
            sum_step_errors=torch.abs(record['sum_step_errors']),
            error_ratios=record['error_ratios'],
        )

        while False: 
            adsf

            while error_ratios is None or torch.any(error_ratios > 1.):
                tl0 = time.time()
                if verbose:
                    print("BEGINNING LOOP")

                # Evaluate new points and add new evals and points to arrays
                t0 = time.time()
                #t_add'], idxs_add'] = self._get_new_eval_times(
                #    t, error_ratios=error_ratios, t_init=t_init, t_final=t_final
                #)

                # Determine new points to add
                if error_ratios is None:
                    idxs_add= None
                else:
                    idxs_add = torch.where(error_ratios > 1.)[0]
                    #print("START ADDING INDICES", t.shape, len(idxs_add), "  |  ", self._get_max_ode_evals(total_mem_usage), len(idxs_add)*self.C)
                    #print("MEMORY", self._get_cuda_memory(), len(idxs_add)*self.C*self.ode_unit_mem_size)
                    if self._get_max_ode_evals(total_mem_usage) < len(idxs_add)*self.C:
                        del y
                        eval_counts = torch.ones(len(t), device=self.device)
                        eval_counts[idxs_add] = 2
                        eval_counts = self.C*torch.cumsum(eval_counts, dim=0).to(torch.int)
                        
                        max_steps = int(0.48*self._get_max_ode_evals(total_mem_usage))
                        assert max_steps > 2*self.Cm1 + 1
                        cut_idx = torch.argmin(torch.abs(eval_counts - max_steps))
                        if eval_counts[cut_idx] > max_steps:
                            cut_idx -= 1
                        t = t[:cut_idx]
                        y, t = self._add_initial_y(
                            ode_fxn=ode_fxn, t_steps_add=t, ode_args=ode_args
                        )
                        idxs_add = None
                        take_gradient = take_gradient is None or take_gradient

                """
                else:
                    idxs_add = torch.where(error_ratios > 1.)[0]
                    #t_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
                
                    # If unique evaluations in y exceeds memory after adding
                    # t_add points, remove latest time evaluations
                    max_steps = self._get_max_ode_evals(total_mem_usage)
                    if len(idxs_add) > max_steps:
                        #print("Removing", n_next_steps, max_steps, t.shape, t_add.shape)
                        eval_counts = torch.ones(len(y), device=self.device)
                        eval_counts[idxs_add] = 2
                        eval_mask = torch.cumsum(eval_counts, dim=0).to(torch.int) <= max_steps
                        y = y[eval_mask]
                        t = t[eval_mask]
                        #t_add = t_add[eval_mask[idxs_add]]
                        idxs_add = idxs_add[eval_mask[idxs_add]]
                """

                #print("B4 Adding Y", t_add.shape, len(t_add))
                #if y is not None:
                #    print(y.shape, t.shape)
                y, t = self._adaptively_add_y(
                    ode_fxn, y, t, idxs_add, ode_args
                )
                #print("SECOND MEMORY", self._get_cuda_memory())
                #print("ADDED Y (t)", y.shape)#, t[:,:,0])
                t_flat = torch.flatten(t, start_dim=0, end_dim=1)
                assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert >= 0)
                #y, t = self.adaptively_add_y(
                #    ode_fxn, y, t, error_ratios, t_init, t_final, ode_args
                #)
                if verbose_speed: print("\t add time", time.time() - t0)
                if verbose:
                    print("NEW T", t.shape, t[:,:,0])
                    print("NEW Y", y.shape, y[:,:,0])

                # Evaluate integral
                t0 = time.time()
                method_output = self._calculate_integral(t, y, y0=y0)
                if verbose_speed: print("\t calc integrals 1", time.time() - t0)
                
                # Calculate error
                t0 = time.time()
                if self.error_calc_idx is None:
                    sum_step_errors = method_output.sum_step_errors
                else:
                    sum_step_errors =\
                        method_output.sum_step_errors[:,self.error_calc_idx]
                    sum_step_errors = sum_step_errors.unsqueeze(-1)
                error_ratios, error_ratios_2steps = self._compute_error_ratios(
                    sum_step_errors=sum_step_errors,
                    sum_steps=method_output.sum_steps,
                    integral=method_output.integral.detach()
                )
                if verbose_speed: print("\t calculate errors", time.time() - t0)
                assert len(y) == len(error_ratios)
                assert len(y) - 1 == len(error_ratios_2steps), f" y: {y.shape} | ratios: {error_ratios_2steps.shape} | t: {t.shape}"
                if verbose:
                    print("ERROR1", error_ratios)
                    print("ERROR2", error_ratios_2steps)
                
                t0 = time.time()
                #print("end inner while", y.shape, t.shape)
                if verbose_speed: print("\t removal mask", time.time() - t0)
                if verbose_speed: print("\tLOOP TIME", time.time() - tl0)
            
            # Create mask for remove points that are too close
            t_pruned = self.prune_excess_t(t, error_ratios_2steps)
            t_flat = torch.flatten(t, start_dim=0, end_dim=1)
            assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert >= 0)
            intermediate_results = IntegralOutput(
                integral=method_output.integral,
                loss=None,
                gradient_taken=take_gradient,
                t_pruned=t_pruned,
                t=t,
                h=method_output.h,
                y=y,
                sum_steps=method_output.sum_steps,
                integral_error=method_output.integral_error,
                sum_step_errors=torch.abs(sum_step_errors),
                error_ratios=error_ratios,
                t_init=t_init,
                t_final=t_final,
                y0=y0
            )


            # Calculate loss
            loss = loss_fxn(intermediate_results)
            intermediate_results.loss = loss
            """
                integral=method_output.integral,
                t=t,
                h=method_output.h,
                sum_steps=method_output.sum_steps,
                integral_error=method_output.integral_error,
                errors=torch.abs(method_output.sum_step_errors),
                error_ratios=error_ratios
            )
            """
            # Add results to record
            record = self._record_results(
                record=record,
                take_gradient=take_gradient,
                results=intermediate_results
            )
            """
                method_output=method_output,
                t=t,
                t_pruned=t_pruned,
                y=y,
                error_ratios=error_ratios,
                loss=loss
            )
            """
            # If batching, take the gradient and free memory
            if take_gradient:
                if loss.requires_grad:
                    loss.backward()
                del y
                y = None
            #print(t[:,:,0])
            #print(y[:,:,0])
            #print("LOSS!!!!!", loss)

        self.previous_ode_fxn = ode_fxn.__name__
        self.t_previous = t
        return IntegralOutput(
            integral=record['integral'],
            loss=record['loss'],
            gradient_taken=take_gradient,
            t_pruned=record['t_pruned'],
            t=record['t'],
            h=record['h'],
            y=record['y'],
            sum_steps=record['sum_steps'],
            integral_error=record['integral_error'],
            sum_step_errors=torch.abs(record['sum_step_errors']),
            error_ratios=record['error_ratios'],
        )


class ParallelUniformAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        error_message = f"Cannot find method '{self.method_name}' in supported method: {list(UNIFORM_METHODS.keys())}"
        assert self.method_name in UNIFORM_METHODS, error_message
        self.method = _get_method(steps.ADAPTIVE_UNIFORM, self.method_name, self.device)# UNIFORM_METHODS[self.method_name].to(self.device)
        self.order = self.method.order
        self.C = len(self.method.tableau.c)
        self.Cm1 = self.C - 1
 

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
 

    def _t_step_interpolate(self, t_left, t_right):
        """
        Determine the time points to evaluate within the integration step that
        spans [t_left, t_right] using the method's tableau c values

        Args:
            t_left (Tensor): Beginning times of all the integration steps
            t_left (Tensor): End times of all the integration steps
        
        Shapes:
            t_left: [N, T]
            t_right: [N, T]
        """
        dt = (t_right - t_left).unsqueeze(1)
        steps = self.method.tableau.c.unsqueeze(-1)*dt
        return t_left.unsqueeze(1) + steps


    def _evaluate_adaptive_y(self, ode_fxn, idxs_add, y, t, ode_args=()):
        D = t.shape[-1]
        t_mid = (t[idxs_add,-1] + t[idxs_add,0])/2.
        t_left = torch.concatenate(
            [t[idxs_add,None,0], t_mid[:,None]], dim=1
        )
        t_right = torch.concatenate(
            [t_mid[:,None], t[idxs_add,None,-1]], dim=1
        )
        #print("ORIG T", t[idxs_add,:,0])
        #print("T LEFT", t_left[:,:,0])
        #print("T RIGHT", t_right[:,:,0])
        #print("VIEW", t_left.view(-1,D))
        t_eval_steps = self._t_step_interpolate(
            t_left.view(-1,D), t_right.view(-1, D)
        )
        y_add = ode_fxn(t_eval_steps.view(-1, D), *ode_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", C=self.C)
        #print("SHAPES", t_eval_steps.shape, y_add.shape)
        return y_add, t_eval_steps
        
    
    def _merge_excess_t(self, t, remove_idxs):
        """
        Merge two integration steps together through the time tensor

        Args:
            t (Tensor): Time points previously evaluated and will be pruned
            remove_idxs (Tensor): Index corresponding to the first time step
                in the contiguous pair to remove
        
        Shapes:
            t: [N, C, T]
            remove_idxs: [R]
        """
        if len(remove_idxs) == 0 or len(t) == 1:
            return t
        t_replace = self._t_step_interpolate(
            t[remove_idxs,0], t[remove_idxs+1,-1]
        )

        remove_mask = torch.ones(len(t), device=self.device, dtype=torch.bool)
        remove_mask[remove_idxs] = False
        #print(remove_idxs[:5])
        #print(remove_mask[:5])
        t_pruned = t[remove_mask]
        remove_idxs_shifted = remove_idxs - torch.arange(
            len(remove_idxs), device=self.device
        ) 
        t_pruned[remove_idxs_shifted] = t_replace
        #print(t_replace[:5,:,0])
        #print(t_pruned[:5,:,0])
        t_pruned_flat = torch.flatten(t_pruned, start_dim=0, end_dim=1)
        assert torch.all(t_pruned_flat[1:] - t_pruned_flat[:-1] + self.atol_assert >= 0)
        t_flat = torch.flatten(t, start_dim=0, end_dim=1)
        assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert>= 0)
        #print("t_replace", t_replace)
        #t[remove_idxs+1] = t_replace
        #print("filled t", t[:,:,0])
        #print("REMOVE MASK", remove_mask)
        #print("END REMOVE EXC", t[remove_mask,:,0])
        #print("T PRUNED", t_pruned[:,:,0])
        return t_pruned
    

class ParallelVariableAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.method_name in VARIABLE_METHODS
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


    def _evaluate_adaptive_y(self, ode_fxn, idxs_add, y, t, ode_args=()):
        t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
        #print("ORIG T", t[idxs_add,:,0])
        y_add = ode_fxn(t_steps_add.view(-1, t.shape[-1]), *ode_args)
        y_add = rearrange(y_add, "(N C) D -> N C D", N=len(idxs_add))
        D = y_add.shape[-1]
        
        select_prev_idxs = torch.arange(self.C, device=self.device)*2
        select_prev_idxs[select_prev_idxs>=self.C] += 1
        select_add_idxs = torch.arange(self.Cm1, device=self.device)*2 + 1
        select_add_idxs[select_add_idxs>=self.C] += 1
        t_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (self.C)*2, D),
            dtype=self.dtype,
            device=self.device
        )
        t_add_combined[:,select_prev_idxs] = t[idxs_add]
        t_add_combined[:,select_add_idxs] = t_steps_add
        t_add_combined[:,self.C] = t_add_combined[:,self.C-1]
        t_add_combined = torch.reshape(
            t_add_combined, (len(idxs_add)*2, self.C, D)
        )

        #print("IDXS", select_prev_idxs, select_add_idxs)
        y_add_combined = torch.nan*torch.ones(
            (len(idxs_add), self.C*2, D),
            dtype=self.dtype,
            device=self.device
        )
        #print("SHAPES",y.shape, y_add.shape)
        y_add_combined[:,select_prev_idxs] = y[idxs_add]
        y_add_combined[:,select_add_idxs] = y_add
        y_add_combined[:,self.C] = y_add_combined[:,self.C-1]
        y_add_combined = torch.reshape(
            y_add_combined, (len(idxs_add)*2, self.C, D)
        )

        #print("SHAPES", t_add_combined.shape, y_add_combined.shape)
        #print(t[:,:,0], y[:,:,0])
        #print(t_add_combined[:,:,0], y_add_combined[:,:,0])
        return y_add_combined, t_add_combined


    def _merge_excess_t(self, t, remove_idxs):
        """
        Merge two integration steps together through the time tensor

        Args:
            t (Tensor): Time points previously evaluated and will be pruned
            remove_idxs (Tensor): Index corresponding to the first time step
                in the contiguous pair to remove
        
        Shapes:
            t: [N, C, T]
            remove_idxs: [R]
        """
        if len(remove_idxs) == 0 or len(t) == 1:
            return t
        t_flat = torch.flatten(t, start_dim=0, end_dim=1)
        assert torch.all(t_flat[1:] - t_flat[:-1] + self.atol_assert >= 0)
        combined_steps = torch.concatenate(
            [t[remove_idxs,:], t[remove_idxs+1,1:]], dim=1
        )
        keep_idxs = torch.arange(self.C, dtype=torch.long, device=self.device)*2
        #print(combined_steps[:5,:,0])
        #print(keep_idxs)
        #print(combined_steps[:5,keep_idxs,0])
        
        remove_mask = torch.ones(len(t), dtype=torch.bool, device=self.device)
        remove_mask[remove_idxs] = False
        #print(remove_idxs[:5])
        #print(remove_mask[:5])
        t_pruned = t[remove_mask]
        update_idxs = remove_idxs-torch.arange(len(remove_idxs), device=self.device)
        t_pruned[update_idxs] = combined_steps[:,keep_idxs]
        t_pruned_flat = torch.flatten(t_pruned, start_dim=0, end_dim=1)
        assert torch.all(t_pruned_flat[1:] - t_pruned_flat[:-1] + self.atol_assert >= 0)
        assert np.allclose(t_pruned[:-1,-1,:], t_pruned[1:,0,:], atol=self.atol_assert, rtol=self.rtol_assert)

        #t[remove_idxs+1] = combined_steps[:,keep_idxs]
        #remove_mask = torch.ones(len(t), dtype=torch.bool)
        #remove_mask[remove_idxs] = False
        return t_pruned
        return t[remove_mask]
        
    def __depricated_adaptively_add_y(
            self,
            ode_fxn,
            y,
            t,
            idxs_add,
            ode_args=()
        ):
        """
        Adds new time points between current time points and splits these
        poinsts into two steps where error_ratio < 1. ode_fxn is evaluated at
        these new time points, both the new time points and y points are merged
        with the original values in order of time.

        Args:
            ode_fxn (Callable): The ode function to be integrated
            y (Tensor): Evaluations of ode_fxn at time points t
            t (Tensor): Current time evaluations in the path integral
            error_ratios (Tensor): Numerical integration error ratios for each
                integration step
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
            ode_args (Tuple): Arguments for ode_fxn
        
        SHAPES:
            y: [N, C, D]
            t : [N, C, T]
            error_ratios : [N]
            t_init: [T]
            t_final: [T]
        
        Notes:
            ode_fxn takes as input (t, *args)
        """

        if idxs_add is None or len(idxs_add) == 0:
            return y, t

   
        """
        # Get variables or populate with default values, send to correct device
        _, t_init, t_final, _ = self._check_variables(t_init=t_init, t_final=t_final)

        if y is None or t is None or error_ratios is None:
            return self._add_initial_y(ode_fxn=ode_fxn, ode_args=ode_args, t=t, t_init=t_init, t_final=t_final)

        if torch.all(error_ratios <= 1.):
            return y, t
        """
        # Get new time steps for merged steps
        #idxs_add = torch.where(error_ratios > 1.)[0]
        #t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
        # Calculate new geometries
        n_add = idxs_add.shape
        # ode_fxn input is 2 dims, t_steps_add has 3 dims, combine first two
        y_add, t_steps_add = self._evaluate_adaptive_y(ode_fxn, idxs_add, t, ode_args)
        print("Y OUT", y_add.shape)
        print("T STEP SHAPE", t_steps_add.shape)
        """
        t_steps_add = rearrange(t_steps_add, 'n c d -> (n c) d')
        y_add = ode_fxn(t_steps_add, *ode_args).to(torch.float64)
        t_steps_add = rearrange(t_steps_add, '(n c) d -> n c d', n=n_add, c=self.Cm1) 
        y_add = rearrange(y_add, '(n c) d -> n c d', n=n_add, c=self.Cm1) 
        """

        # Create new vector to fill with old and new values
        y_combined = torch.zeros(
            (len(y)+len(y_add), self.Cm1+1, y_add.shape[-1]),
            dtype=self.dtype,
            requires_grad=False,
            device=self.device
        ).detach()
        t_combined = torch.zeros_like(
            y_combined,
            dtype=self.dtype,
            requires_grad=False,
            device=self.device
        ).detach()
        
        # Add old t and y values, skipping regions with new points
        idx_offset = torch.zeros(len(y), dtype=torch.long, device=self.device)
        idx_offset[idxs_add] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idx_input = torch.arange(len(y), device=self.device) + idx_offset
        y_combined[idx_input,:] = y
        t_combined[idx_input,:] = t

        # Add new t and y values to added rows
        idxs_add_offset = idxs_add + torch.arange(
            len(idxs_add), device=self.device
        )
        t_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (self.Cm1+1)*2-1, d),
            dtype=self.dtype,
            device=self.device
        )
        t_add_combined[:,torch.arange(self.Cm1+1, device=self.device)*2] = t[idxs_add]
        t_add_combined[:,torch.arange(self.Cm1, device=self.device)*2+1] = t_steps_add
        t_combined[idxs_add_offset,:,:] = t_add_combined[:,:self.Cm1+1]
        t_combined[idxs_add_offset+1,:,:] = t_add_combined[:,self.Cm1:]

        y_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (self.Cm1+1)*2-1, d),
            dtype=torch.self.dtype,
            device=self.device
        )
        y_add_combined[:,torch.arange(self.Cm1+1, device=self.device)*2] = y[idxs_add]
        y_add_combined[:,torch.arange(self.Cm1, device=self.device)*2+1] = y_add
        y_combined[idxs_add_offset,:,:] = y_add_combined[:,:self.Cm1+1]
        y_combined[idxs_add_offset+1,:,:] = y_add_combined[:,self.Cm1:]

        assert torch.allclose(t_combined[:-1,-1], t_combined[1:,0], atol=self.atol_assert, rtol=self.rtol_assert)
        return y_combined, t_combined

