import time
import torch
import numpy as np
from einops import rearrange

from .methods import _get_method, UNIFORM_METHODS, VARIABLE_METHODS
from .base import SolverBase, IntegralOutput, steps

class ParallelAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, remove_cut=0.1, use_absolute_error_ratio=True, *args, **kwargs):
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
        self.previous_t = None
        self.previous_ode_fxn = None


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
                integration method, [N, C, T] will retun t
            t_init: [T]
            t_final: [T]
        """
        raise NotImplementedError
    

    def _remove_excess_t(self, t, remove_idxs):
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
            ).to(torch.float64)
            N, C, _ = t_steps.shape

            # Time points to evaluate, remove repetitive time points at the end
            # of each step to minimize evaluations
            t_add = torch.concatenate(
                [t_steps[0], t_steps[1:,1:].reshape((-1, *(t_steps.shape[2:])))],
                dim=0
            )
            idxs_add = torch.arange(N)
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
            ).to(torch.float64)

        if inforce_endpoints:
            if t_init != t[0,0]:
                # Remove time steps where first point is less than t_init
                t = t[t[:,-1,0] > t_init[0]]
                # First step should start at t_init
                inp = torch.tensor([t_init.unsqueeze(0), t[0,-1].unsqueeze(0)])
                if t.shape[-1] == 1:
                    inp = inp.unsqueeze(-1)
                t[0] = self._initial_t_steps(
                    inp, t_init=t_init, t_final=t_final
                ).to(torch.float64)
            if t_final != t[-1,-1]:
                # Remove time steps where last point is greater than t_final
                t = t[t[:,0] < t_final]
                # Last step should end at t_final
                inp = torch.tensor([t[-1,0].unsqueeze(0), t_final.unsqueeze(0)]),
                if t.shape[-1] == 1:
                    inp = inp.unsqueeze(-1)
                t[-1] = self._initial_t_steps(
                    inp, t_init=t_init, t_final=t_final
                ).to(torch.float64)
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
        N, C, T = t_steps_add.shape 
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
        y_add = ode_fxn(t_add, *ode_args).to(torch.float64)

        # Add repetitive y values to the end of each integration step
        y_steps = torch.reshape(y_add[1:], (N, C-1, T))
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


   
    def _interweave_evaluations(
            self,
            ode_fxn,
            y,
            t,
            idxs_add,
            t_steps_add,
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
        n_add, nm1_c, d = t_steps_add.shape
        # ode_fxn input is 2 dims, t_steps_add has 3 dims, combine first two
        t_steps_add = rearrange(t_steps_add, 'n c d -> (n c) d')
        y_add = ode_fxn(t_steps_add, *ode_args).to(torch.float64)
        t_steps_add = rearrange(t_steps_add, '(n c) d -> n c d', n=n_add, c=nm1_c) 
        y_add = rearrange(y_add, '(n c) d -> n c d', n=n_add, c=nm1_c) 

        # Create new vector to fill with old and new values
        y_combined = torch.zeros(
            (len(y)+len(y_add), nm1_c+1, y_add.shape[-1]),
            dtype=torch.float64,
            requires_grad=False,
            device=self.device
        ).detach()
        t_combined = torch.zeros_like(
            y_combined,
            dtype=torch.float64,
            requires_grad=False,
            device=self.device
        ).detach()
        
        # Add old t and y values, skipping regions with new points
        idx_offset = torch.zeros(len(y), dtype=torch.long)
        idx_offset[idxs_add] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idx_input = torch.arange(len(y)) + idx_offset
        y_combined[idx_input,:] = y
        t_combined[idx_input,:] = t

        # Add new t and y values to added rows
        idxs_add_offset = idxs_add + torch.arange(
            len(idxs_add), device=self.device
        )
        t_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (nm1_c+1)*2-1, d),
            dtype=torch.float64,
            device=self.device
        )
        t_add_combined[:,torch.arange(nm1_c+1)*2] = t[idxs_add]
        t_add_combined[:,torch.arange(nm1_c)*2+1] = t_steps_add
        t_combined[idxs_add_offset,:,:] = t_add_combined[:,:nm1_c+1]
        t_combined[idxs_add_offset+1,:,:] = t_add_combined[:,nm1_c:]

        y_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (nm1_c+1)*2-1, d),
            dtype=torch.float64,
            device=self.device
        )
        y_add_combined[:,torch.arange(nm1_c+1)*2] = y[idxs_add]
        y_add_combined[:,torch.arange(nm1_c)*2+1] = y_add
        y_combined[idxs_add_offset,:,:] = y_add_combined[:,:nm1_c+1]
        y_combined[idxs_add_offset+1,:,:] = y_add_combined[:,nm1_c:]

        assert torch.all(t_combined[:-1,-1] == t_combined[1:,0])
        return y_combined, t_combined


    def _adaptively_add_y(
            self,
            ode_fxn,
            y,
            t,
            idxs_add,
            t_add,
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

        if len(t_add) == 0:
            return y, t

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
 
    def remove_excess_y(self, t, error_ratios_2steps):
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
        
        t_pruned = self._remove_excess_t(t, ratio_idxs_cut)
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


    def _compute_error_ratios(self, sum_step_errors, sum_steps=None, integral=None):
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
                sum_step_errors, sum_steps
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
    
    
    def _compute_error_ratios_cumulative(self, sum_step_errors, sum_steps):
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
        cum_steps = torch.cumsum(sum_steps, dim=0)
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
    
    def _record_results(self, record, is_batched, method_output, t, t_pruned, y, error_ratios, loss):
        if len(record) == 0 and not is_batched:
            record['integral'] = method_output.integral
            record['t_pruned'] = t_pruned
            record['t'] = t
            record['h'] = method_output.h
            record['y'] = y
            record['sum_steps'] = method_output.sum_steps
            record['sum_step_errors'] = method_output.sum_step_errors
            record['integral_error'] = method_output.integral_error
            record['error_ratios'] = error_ratios
            record['loss'] = loss
            return record
        elif len(record) == 0 and is_batched:
            record['integral'] = method_output.integral.detach()
            record['t_pruned'] = t_pruned.detach()
            record['t'] = t.detach()
            record['h'] = method_output.h.detach()
            record['y'] = y.detach()
            record['sum_steps'] = method_output.sum_steps.detach()
            record['sum_step_errors'] = method_output.sum_step_errors.detach()
            record['integral_error'] = method_output.integral_error.detach()
            record['error_ratios'] = error_ratios.detach()
            record['loss'] = loss.detach()
            return record 

        record['integral'] = record['integral']\
            + method_output.integral.detach()
        record['t_pruned'] = torch.concatenate(
            [record['t_pruned'], t_pruned.detach()], dim=0
        )
        record['t'] = torch.concatenate([record['t'], t.detach()], dim=0)
        record['h'] = torch.concatenate(
            [record['h'], method_output.h.detach()], dim=0
        )
        record['y'] = torch.concatenate([record['y'], y.detach()], dim=0)
        record['sum_steps'] = torch.concatenate(
            [record['sum_steps'], method_output.sum_steps.detach()], dim=0
        )
        record['sum_step_errors'] = torch.concatenate(
            [record['sum_step_errors'], method_output.sum_step_errors.detach()],
            dim=0
        )
        record['integral_error'] = record['integral_error']\
            + method_output.integral_error.detach()
        record['error_ratios'] = torch.concatenate(
            [record['error_ratios'], error_ratios.detach()], dim=0
        )
        record['loss'] = record['loss'] + loss.detach()

        return record

    def integrate(
            self,
            ode_fxn=None,
            y0=None,
            t=None,
            t_init=None,
            t_final=None,
            ode_args=(),
            max_batch=None,
            loss_fxn=None,
            verbose=False,
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
        if t is not None:
            if len(t.shape) == 2:
                assert torch.all(t[1:] > t[:-1])
                if self.Cm1 > 1:
                    error_message = f"When giving t (with 2 dims) the first dimension (N) must satisfy N % {self.Cm1} = 1 in order to be properly divided into integration steps for the {self.method_name} method."
                    assert len(t) % self.Cm1 == 1, error_message
            if len(t.shape) == 3:
                error_message = f"When giving t (with 3 dims) the second dimension must match the integration method's number of samples per step. For {self.method_name} it's {self.C}"
                assert t.shape[1] == self.C, error_message
                assert torch.all(
                    torch.flatten(t, 0, 1)[1:] >= torch.flatten(t, 0, 1)[:-1]
                )
            if t_init is None:
                t_init = t[0,0]
            else:
                assert torch.all(t[0,0] == t_init)
            if t_final is None:
                t_final = t[-1,-1]
            else:
                assert torch.all(t[-1,-1] == t_final)
        
        # Get variables or populate with default values, send to correct device
        ode_fxn, t_init, t_final, y0 = self._check_variables(
            ode_fxn, t_init, t_final, y0
        )
        loss_fxn = loss_fxn if loss_fxn is not None else self._integral_loss

        # Make sure ode_fxn exists and provides the correct output
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        test_output = ode_fxn(
            torch.tensor([[t_init]], dtype=torch.float64, device=self.device),
            *ode_args
        ) 
        assert len(test_output.shape) >= 2
        del test_output
        
        if t is None:
            same_fxn = self.previous_ode_fxn != ode_fxn.__name__
            if self.previous_t is not None and same_fxn:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t = self.previous_t[mask]
        t_steps_init = self._get_initial_t_steps(
            t, t_init, t_final, inforce_endpoints=True
        )
        t, y = None, None

        batch_buffer = 0.8
        if max_batch is not None:
            assert max_batch >= 2*self.Cm1 + 1, f"max_batch ({max_batch}) must be >= to the number of samples in two integration steps ({2*self.Cm1 + 1})"
            max_steps = (max_batch - 1)//self.Cm1
            batch_size = int(max_steps*batch_buffer)

        record = {}
        while t is None or torch.all(t[-1,-1] != t_final):
            # Initial time should start at the end of the previous batch
            _t_init = t_init if t is None else t[-1,-1]
            #if t is not None:
            #    print("PREV T", t[:,:,0])
            #print("T INIT", _t_init)
            t_steps = self._get_initial_t_steps(
                t_steps_init, _t_init, t_final, inforce_endpoints=True
            )
            t_add = t_steps if max_batch is None else t_steps[:batch_size]
            #print("INIT TADD PORTION", t_add.shape, t_steps.shape)
            #print("TADD INIT", t_add[:,:,0])
            t = None

            error_ratios=None
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
                if error_ratios is not None and t is not None:
                    idxs_add = torch.where(error_ratios > 1.)[0]
                    t_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]
                else:
                    idxs_add= None
                
                # If unique evaluations in y exceeds max_batch after adding
                # t_add points, remove latest time evaluations
                if y is not None and max_batch is not None:
                    n_next_steps = len(y) + len(idxs_add)
                    if n_next_steps > max_steps:
                        #print("Removing", n_next_steps, max_steps, t.shape, t_add.shape)
                        eval_counts = torch.ones(len(y))
                        eval_counts[idxs_add] = 2
                        eval_mask = torch.cumsum(eval_counts, dim=0).to(torch.int) <= max_steps
                        y = y[eval_mask]
                        t = t[eval_mask]
                        t_add = t_add[eval_mask[idxs_add]]
                        idxs_add = idxs_add[eval_mask[idxs_add]]

                #print("B4 Adding Y", t_add.shape, len(t_add))
                #if y is not None:
                #    print(y.shape, t.shape)
                y, t = self._adaptively_add_y(
                    ode_fxn, y, t, idxs_add, t_add, ode_args
                )
                #print("ADDED Y (t)", y.shape)#, t[:,:,0])
                t_flat = torch.flatten(t, start_dim=0, end_dim=1)
                assert torch.all(t_flat[1:] - t_flat[:-1] >= 0)
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
                error_ratios, error_ratios_2steps = self._compute_error_ratios(
                    sum_step_errors=method_output.sum_step_errors,
                    sum_steps=method_output.sum_steps,
                    integral=method_output.integral
                )
                if verbose_speed: print("\t calculate errors", time.time() - t0)
                assert len(y) == len(error_ratios)
                assert len(y) - 1 == len(error_ratios_2steps), f" y: {y.shape} | ratios: {error_ratios_2steps.shape} | t: {t.shape}"
                if verbose:
                    print("ERROR1", error_ratios)
                    print("ERROR2", error_ratios_2steps)
                
                t0 = time.time()
                #print("end inner while", y.shape, t.shape)
                # Actions to perform once integral range is within tolerance
                if torch.all(error_ratios <= 1.):
                    # Create mask for remove points that are too close
                    t_pruned = self.remove_excess_y(t, error_ratios_2steps)
                    t_flat = torch.flatten(t, start_dim=0, end_dim=1)
                    assert torch.all(t_flat[1:] - t_flat[:-1] >= 0)
                    # Calculate loss
                    loss = loss_fxn(
                        integral=method_output.integral,
                        t=t,
                        h=method_output.h,
                        sum_steps=method_output.sum_steps,
                        integral_error=method_output.integral_error,
                        errors=torch.abs(method_output.sum_step_errors),
                        error_ratios=error_ratios
                    )
                    # Add results to record
                    record = self._record_results(
                        record=record,
                        is_batched=max_batch is not None,
                        method_output=method_output,
                        t=t,
                        t_pruned=t_pruned,
                        y=y,
                        error_ratios=error_ratios,
                        loss=loss
                    )
                    # If batching, take the gradient and free memory
                    if max_batch is not None:
                        if loss.requires_grad:
                            loss.backward()
                        del y
                        y = None
                    #print(t[:,:,0])
                    #print(y[:,:,0])
                    #print("LOSS!!!!!", loss)
                if verbose_speed: print("\t removal mask", time.time() - t0)
                if verbose_speed: print("\tLOOP TIME", time.time() - tl0)

        self.previous_ode_fxn = ode_fxn.__name__
        self.t_previous = t
        return IntegralOutput(
            integral=record['integral'],
            loss=record['loss'],
            t_pruned=record['t_pruned'],
            t=record['t'],
            h=record['h'],
            y=record['y'],
            sum_steps=record['sum_steps'],
            integral_error=record['integral_error'],
            errors=torch.abs(record['sum_step_errors']),
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
 

    def _initial_t_steps(self,
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
                integration method; [N, C, T] will return t if C is the same
                as the number of evaluations per step, otherwise it will create
                C steps between the first and last values in the second dim 
            t_init: [T]
            t_final: [T]
        """
        
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
                    assert torch.all(t[:-1,-1] == t[1:,0])
                t = t[:,torch.tensor([0,-1], dtype=torch.int),:]
                t = torch.flatten(t, start_dim=0, end_dim=1)
        return self._t_step_interpolate(t[:-1], t[1:])
 

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


    def _remove_excess_t(self, t, remove_idxs):
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

        remove_mask = torch.ones(len(t), dtype=torch.bool)
        remove_mask[remove_idxs] = False
        #print(remove_idxs[:5])
        #print(remove_mask[:5])
        t_pruned = t[remove_mask]
        t_pruned[remove_idxs-torch.arange(len(remove_idxs))] = t_replace
        #print(t_replace[:5,:,0])
        #print(t_pruned[:5,:,0])
        #t_pruned_flat = torch.flatten(t_pruned, start_dim=0, end_dim=1)
        #assert torch.all(t_pruned_flat[1:] - t_pruned_flat[:-1] >= 0)
        #t_flat = torch.flatten(t, start_dim=0, end_dim=1)
        #assert torch.all(t_flat[1:] - t_flat[:-1] >= 0)
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
                integration method; [N, C, T] will return t if C is the same
                as the number of evaluations per step, otherwise it will create
                C steps between the first and last values in the second dim 
            t_init: [T]
            t_final: [T]
        """
 
        # Get variables or populate with default values, send to correct device
        _, t_init, t_final, _ = self._check_variables(
            t_init=t_init, t_final=t_final
        )
        if t is None:
            t = torch.linspace(0, 1., 7).unsqueeze(-1)
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
                assert torch.all(t[:-1,-1] == t[1:,0])
            t_left = t[:,0] 
            t_right = t[:,-1] 
            #steps = torch.tile(
            #    torch.arange(self.C)[None,:,None], (len(t), 1, t.shape[-1])
            #)
            #return t[:,0,:] + steps*(t[:,-1,:] - t[:,0,:])/self.Cm1
        steps = torch.arange(self.C)[None,:,None]/self.Cm1
        t_left = t_left.unsqueeze(1)
        t_right = t_right.unsqueeze(1)
        return t_left + steps*(t_right - t_left)
        print("here", t.shape, t)
        _t = torch.reshape(t[:-1], (-1, self.Cm1, 1))
        _t_ends = torch.concatenate([_t[1:,0], t[None,-1]]).unsqueeze(1)
        return torch.concatenate([_t, _t_ends], dim=1)
    
   
    def _remove_excess_t(self, t, remove_idxs):
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
        assert torch.all(t_flat[1:] - t_flat[:-1] >= 0)
        combined_steps = torch.concatenate(
            [t[remove_idxs,:], t[remove_idxs+1,1:]], dim=1
        )
        keep_idxs = torch.arange(self.C, dtype=torch.long)*2
        #print(combined_steps[:5,:,0])
        #print(keep_idxs)
        #print(combined_steps[:5,keep_idxs,0])
        
        remove_mask = torch.ones(len(t), dtype=torch.bool)
        remove_mask[remove_idxs] = False
        #print(remove_idxs[:5])
        #print(remove_mask[:5])
        t_pruned = t[remove_mask]
        update_idxs = remove_idxs-torch.arange(len(remove_idxs))
        t_pruned[update_idxs] = combined_steps[:,keep_idxs]
        #t_pruned_flat = torch.flatten(t_pruned, start_dim=0, end_dim=1)
        #assert torch.all(t_pruned_flat[1:] - t_pruned_flat[:-1] >= 0)
        #assert np.allclose(t_pruned[:-1,-1,:], t_pruned[1:,0,:])

        #t[remove_idxs+1] = combined_steps[:,keep_idxs]
        #remove_mask = torch.ones(len(t), dtype=torch.bool)
        #remove_mask[remove_idxs] = False
        return t_pruned
        return t[remove_mask]