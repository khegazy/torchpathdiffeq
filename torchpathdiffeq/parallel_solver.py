import time
import torch
from einops import rearrange

from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .base import SolverBase, IntegralOutput

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
        self.C = None
        self.Cm1 = None
        self.previous_t = None
        self.previous_ode_fxn = None

    def _initial_t_steps(
            self,
            t,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64)
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
    

    def _add_initial_y(
            self,
            ode_fxn,
            t,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            ode_args=()
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
        
        # Calculate new geometries
        y_add = ode_fxn(t_add, *ode_args).to(torch.float64)

        # Add repetitive y values to the end of each integration step
        y_steps = torch.reshape(y_add[1:], (N, C-1, -1))
        y_steps = torch.concatenate(
            [
                torch.concatenate(
                    [y_add[None,None,0,...], y_steps[:-1,None,-1]], dim=0
                ),
                y_steps
            ],
            dim=1
        )

        return y_steps, t_steps


    def adaptively_add_y(
            self,
            ode_fxn,
            y=None,
            t=None,
            error_ratios=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
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
        if y is None or t is None or error_ratios is None:
            return self._add_initial_y(ode_fxn=ode_fxn, ode_args=ode_args, t=t, t_init=t_init, t_final=t_final)

        if torch.all(error_ratios <= 1.):
            return y, t
            
        # Get new time steps for merged steps
        idxs_add = torch.where(error_ratios > 1.)[0]
        t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, C-1, 1]

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
            requires_grad=False
        ).detach()
        t_combined = torch.zeros_like(
            y_combined, dtype=torch.float64, requires_grad=False
        ).detach()
        
        # Add old t and y values, skipping regions with new points
        idx_offset = torch.zeros(len(y), dtype=torch.long)
        idx_offset[idxs_add] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idx_input = torch.arange(len(y)) + idx_offset
        y_combined[idx_input,:] = y
        t_combined[idx_input,:] = t

        # Add new t and y values to added rows
        idxs_add_offset = idxs_add + torch.arange(len(idxs_add))
        t_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (nm1_c+1)*2-1, d), dtype=torch.float64
        )
        t_add_combined[:,torch.arange(nm1_c+1)*2] = t[idxs_add]
        t_add_combined[:,torch.arange(nm1_c)*2+1] = t_steps_add
        t_combined[idxs_add_offset,:,:] = t_add_combined[:,:nm1_c+1]
        t_combined[idxs_add_offset+1,:,:] = t_add_combined[:,nm1_c:]

        y_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (nm1_c+1)*2-1, d), dtype=torch.float64
        )
        y_add_combined[:,torch.arange(nm1_c+1)*2] = y[idxs_add]
        y_add_combined[:,torch.arange(nm1_c)*2+1] = y_add
        y_combined[idxs_add_offset,:,:] = y_add_combined[:,:nm1_c+1]
        y_combined[idxs_add_offset+1,:,:] = y_add_combined[:,nm1_c:]

        assert torch.all(t_combined[:-1,-1] == t_combined[1:,0])
        return y_combined, t_combined


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
            return torch.empty(0, dtype=torch.long)
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
    

    def integrate(
            self,
            ode_fxn=None,
            y0=None,
            t=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            ode_args=(),
            verbose=False,
            verbose_speed=False
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

        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        assert len(ode_fxn(torch.tensor([[t_init]]), *ode_args).shape) >= 2
        
        if t is None:
            same_fxn = self.previous_ode_fxn != ode_fxn.__name__
            if self.previous_t is not None and same_fxn:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t = self.previous_t[mask]

        y = None
        error_ratios=None
        while y is None or torch.any(error_ratios > 1.):
            tl0 = time.time()
            if verbose:
                print("BEGINNING LOOP")

            # Evaluate new points and add new evals and points to arrays
            t0 = time.time()
            y, t = self.adaptively_add_y(
                ode_fxn, y, t, error_ratios, t_init, t_final, ode_args
            )
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
            assert len(y) - 1 == len(error_ratios_2steps)
            if verbose:
                print("ERROR1", error_ratios)
                print("ERROR2", error_ratios_2steps)
            
            # Create mask for remove points that are too close
            t0 = time.time()
            if torch.all(error_ratios <= 1.):
                t_pruned = self.remove_excess_y(t, error_ratios_2steps)
            if verbose_speed: print("\t removal mask", time.time() - t0)
            if verbose_speed: print("\tLOOP TIME", time.time() - tl0)

        self.previous_ode_fxn = ode_fxn.__name__
        self.t_previous = t
        return IntegralOutput(
            integral=method_output.integral,
            t_pruned=t_pruned,
            t=t,
            h=method_output.h,
            y=y,
            sum_steps=method_output.sum_steps,
            integral_error=method_output.integral_error,
            errors=torch.abs(method_output.sum_step_errors),
            error_ratios=error_ratios,
        )


class ParallelUniformAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        error_message = f"Cannot find method '{self.method_name}' in supported method: {list(UNIFORM_METHODS.keys())}"
        assert self.method_name in UNIFORM_METHODS, error_message
        self.method = UNIFORM_METHODS[self.method_name]
        self.C = self.method.order
        self.Cm1 = self.C - 1
 

    def _initial_t_steps(self,
            t,
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64)
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
        
        if t is None:
            t = torch.linspace(0, 1., 7).unsqueeze(-1)
            t = t_init + t*(t_final - t_init)
        elif len(t.shape) == 3:
            return t
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
        if len(remove_idxs) == 0:
            return t
        t_replace = self._t_step_interpolate(
            t[remove_idxs,0], t[remove_idxs+1,-1]
        )
        t[remove_idxs+1] = t_replace
        remove_mask = torch.ones(len(t), dtype=torch.bool)
        remove_mask[remove_idxs] = False
        return t[remove_mask]
    

class ParallelVariableAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.method_name in VARIABLE_METHODS
        self.method = VARIABLE_METHODS[self.method_name]()
        self.C = self.method.order 
        self.Cm1 = self.C - 1
    
    def _initial_t_steps(
            self,
            t,             
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64)
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
                intetgration method, [N, C, T] will retun t
            t_init: [T]
            t_final: [T]
        """
 
        if t is None:
            t = torch.linspace(0, 1., 7*self.Cm1+1).unsqueeze(-1)
            t = t_init + t*(t_final - t_init)
        elif len(t.shape) == 3:
            return t
        else:
            error_message = f"Input time must be of length N*({self.method.order-1}) + 1, instead got {t.shape}"
            assert (len(t) - 1) % self.Cm1 == 0, error_message
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
        if len(remove_idxs) == 0:
            return t
        combined_steps = torch.concatenate(
            [t[remove_idxs,:], t[remove_idxs+1,1:]], dim=1
        )
        keep_idxs = torch.arange(self.C, dtype=torch.long)*2
        t[remove_idxs+1] = combined_steps[:,keep_idxs]
        remove_mask = torch.ones(len(t), dtype=torch.bool)
        remove_mask[remove_idxs] = False
        return t[remove_mask]