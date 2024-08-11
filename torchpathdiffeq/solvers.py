import time
import torch
import torch.distributed as dist
import numpy as np
from torchdiffeq import odeint
from dataclasses import dataclass
from enum import Enum

from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .adaptivity import IntegralAdaptivity
#_adaptively_add_y, _remove_excess_y, _find_sparse_y, _compute_error_ratios, _add_initial_y


class steps(Enum):
    FIXED = 0
    ADAPTIVE_UNIFORM = 1
    ADAPTIVE_VARIABLE = 2

ADAPTIVE_METHOD_P = {
    'heun' : 2
}

@dataclass
class IntegralOutput():
    integral: torch.Tensor
    t_pruned: torch.Tensor = None
    t: torch.Tensor = None
    h: torch.Tensor = None
    y: torch.Tensor = None
    sum_steps: torch.Tensor = None
    integral_error: torch.Tensor = None
    errors: torch.Tensor = None
    error_ratios: torch.Tensor = None

@dataclass
class MethodOutput():
    integral: torch.Tensor
    integral_error: torch.Tensor
    sum_steps: torch.Tensor
    sum_step_errors: torch.Tensor
    h: torch.Tensor

    

class SolverBase():
    def __init__(self, method, atol, rtol, y0=torch.tensor([0], dtype=torch.float), ode_fxn=None, t_init=0., t_final=1.) -> None:

        self.method_name = method.lower()
        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.y0 = y0
        self.t_init = t_init
        self.t_final = t_final

    def _calculate_integral(self, t, y, y0=0):
        raise NotImplementedError
    
    def integrate(self, ode_fxn, y0=None, t_init=0., t_final=1., t=None, ode_args=None):
        raise NotImplementedError




class SerialAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, method, atol, rtol, y0=torch.tensor([0], dtype=torch.float), ode_fxn=None, t_init=0, t_final=1.) -> None:
        super().__init__(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            y0=y0,
            t_init=t_init,
            t_final=t_final
        )

    
    def integrate(self, ode_fxn=None, state=None, y0=None, t_init=0., t_final=1., t=None, ode_args=None):
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        if t is None:
            t=torch.tensor([t_init, t_final])
        if state is not None:
            t = state.t
        
        integral = odeint(
            func=ode_fxn,
            y0=y0,
            t=t,
            method=self.method_name,
            rtol=self.rtol,
            atol=self.atol
        )

        return IntegralOutput(
            integral=integral[-1],
            t=t,
        )




class ParallelAdaptiveStepsizeSolver(SolverBase, IntegralAdaptivity):
    
    def __init__(
            self,
            method,
            atol,
            rtol,
            remove_cut=0.1,
            y0=torch.tensor([0], dtype=torch.float64),
            ode_fxn=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
        ):
        super().__init__(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            y0=y0,
            t_init=t_init,
            t_final=t_final
        )

        self.method = None
        self.p1 = None
        self.p = None
        self.atol = atol
        self.rtol = rtol
        self.remove_cut = remove_cut
        self.previous_t = None
        self.previous_ode_fxn = None

    
    def _get_tableau_b(self, t):
        raise NotImplementedError


    def _error_norm(self, error):
        return torch.sqrt(torch.mean(error**2, -1))
    
    
    def integrate(
            self,
            ode_fxn=None,
            state=None,
            y0=None,
            t_init=0.,
            t_final=1.,
            t=None,
            ode_args=None,
            remove_mask=None
        ):
        
        verbose=False
        speed_verbose=False
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        assert len(ode_fxn(torch.tensor([[t_init]])).shape) >= 2
        if state is not None:
            t = state.t
        
        t_add = t
        if t_add is None:
            if self.previous_t is None\
                or self.previous_ode_fxn != ode_fxn.__name__:
                t_add = torch.unsqueeze(
                    torch.linspace(t_init, t_final, 13), 1
                )
            else:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t_add = self.previous_t[mask]

        """
        if error_ratios is None:
            t_steps, y_steps = _add_initial_y(t_add, self._t_spacing_fxn)
            integral_p, y_p, _ = self._calculate_integral(
                t_steps, y_steps, y0=y0, degr=degree.P
            )
            integral_p1, y_p1, h = self._calculate_integral(
                t_steps, y_steps, y0=y0, degr=degree.P1
            )
            error_ratios, error_ratios_2steps = _compute_error_ratios(
                y_p, y_p1, self.rtol, self.atol, self._error_norm
            )
        else:
            integral_p1 = None
        """ 

        y = None
        error_ratios=None
        loop_cnt = 0
        while y is None or torch.any(error_ratios > 1.):
            # check remove
            # remove idxs
            # check and add
            # new errors
            tl0 = time.time()
            #print("START OF LOOP", loop_cnt)
            if verbose:
                print("BEGINNING LOOP")
                print("TADD", t_add.shape, t_add, idxs_add)

            #idxs_remove_pair = _find_excess_y(error_ratios_2steps, self.remove_cut)

            # Evaluate new points and add new evals and points to arrays
            t0 = time.time()
            y, t = self.adaptively_add_y(
                ode_fxn, y, t, error_ratios, t_init, t_final
            )
            if speed_verbose: print("\t add time", time.time() - t0)
            if verbose or True:
                print("NEW T", t.shape, t[:,:,0])
                print("NEW Y", y.shape, y[:,:,0])

            # Evaluate integral
            t0 = time.time()
            method_output = self._calculate_integral(t, y, y0=y0)
            #integral_p1, sum_steps_p1, h = self._calculate_integral(t, y, y0=y0, degr=degree.P1)
            if speed_verbose: print("\t calc integrals 1", time.time() - t0)
            
            # Calculate error
            t0 = time.time()
            #print("YP SHAPES", y_p.shape, y_p1.shape)
            error_ratios, error_ratios_2steps = self.compute_error_ratios(
                method_output.sum_steps, method_output.sum_step_errors
            )
            print("ER SHAPES", error_ratios.shape, error_ratios_2steps.shape, y.shape, t.shape)
            if speed_verbose: print("\t calculate errors", time.time() - t0)
            assert len(y) == len(error_ratios)
            assert len(y) - 1 == len(error_ratios_2steps)
            #print(error_ratios)
            if verbose:
                print("ERROR1", error_ratios)
                print("ERROR2", error_ratios_2steps)
                print(integral_p, integral_p1)
            
            # Create mask for remove points that are too close
            t0 = time.time()
            if torch.all(error_ratios <= 1.):
                t_pruned = self.remove_excess_y(
                    t, error_ratios_2steps, self._remove_excess_t
                )
            if speed_verbose: print("\t removal mask", time.time() - t0)

            #y_pruned = y[~remove_mask]
            #t_pruned = t[~remove_mask]

            """
            # Find indices where error is too large to add new points
            # Evaluate integral
            t0 = time.time()
            _, y_p_pruned, _ = self._calculate_integral(
                t_pruned, y_pruned, y0=y0, degr=degree.P
            )
            _, y_p1_pruned, _ = self._calculate_integral(
                t_pruned, y_pruned, y0=y0, degr=degree.P1
            )
            if speed_verbose: print("\t calc integrals 2", time.time() - t0)
            
            # Calculate error
            error_ratios_pruned, _ = _compute_error_ratios(
                y_p_pruned, y_p1_pruned, self.rtol, self.atol, self._error_norm
            )

            t_add, idxs_add = _find_sparse_y(
                t_pruned, self.p, error_ratios_pruned
            )
            """

            if speed_verbose: print("\tLOOP TIME", time.time() - tl0)

        print("SUM T", t.shape, torch.sum(t), torch.all(t[:-1,1]==t[1:,0]))

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


    def _integrate(
            self,
            ode_fxn=None,
            state=None,
            y0=None,
            t_init=0.,
            t_final=1.,
            t=None,
            ode_args=None,
            remove_mask=None
        ):
        
        verbose=False
        speed_verbose=False
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        assert len(ode_fxn(torch.tensor([[t_init]])).shape) >= 2
        if state is not None:
            t = state.t
            remove_mask = state.remove_mask
            error_ratios = state.error_ratios
        if remove_mask is not None:
            t = t[~remove_mask]
        
        
        t_add = t
        if t_add is None:
            if self.previous_t is None\
                or self.previous_ode_fxn != ode_fxn.__name__:
                t_add = torch.unsqueeze(
                    torch.linspace(t_init, t_final, 13), 1
                )
            else:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t_add = self.previous_t[mask]
        
        if error_ratios is None:
            t_steps, y_steps = _add_initial_y(t_add, self._t_spacing_fxn)
            integral_p, y_p, _ = self._calculate_integral(
                t_steps, y_steps, y0=y0, degr=degree.P
            )
            integral_p1, y_p1, h = self._calculate_integral(
                t_steps, y_steps, y0=y0, degr=degree.P1
            )
            error_ratios, error_ratios_2steps = _compute_error_ratios(
                y_p, y_p1, self.rtol, self.atol, self._error_norm
            )
        else:
            integral_p1 = None

        loop_cnt = 0
        while torch.any(error_ratios > 1.) or integral_p1 is None:
            # check remove
            # remove idxs
            # check and add
            # new errors
            tl0 = time.time()
            #print("START OF LOOP", loop_cnt)
            if verbose:
                print("BEGINNING LOOP")
                print("TADD", t_add.shape, t_add, idxs_add)

            idxs_remove_pair = _find_excess_y(error_ratios_2steps, self.remove_cut)

            # Evaluate new points and add new evals and points to arrays
            t0 = time.time()
            y, t = _adaptively_add_y(
                ode_fxn, t_steps, y_steps, error_ratios, idxs_remove_pair
            )
            if speed_verbose: print("\t add time", time.time() - t0)
            if verbose:
                print("NEW T", t.shape, t)
                print("NEW Y", y.shape, y)

            # Evaluate integral
            t0 = time.time()
            integral_p, y_p, _ = self._calculate_integral(t, y, y0=y0, degr=degree.P)
            integral_p1, y_p1, h = self._calculate_integral(t, y, y0=y0, degr=degree.P1)
            if speed_verbose: print("\t calc integrals 1", time.time() - t0)
            
            # Calculate error
            t0 = time.time()
            error_ratios, error_ratios_2steps = _compute_error_ratios(
                y_p, y_p1, self.rtol, self.atol, self._error_norm
            )
            if speed_verbose: print("\t calculate errors", time.time() - t0)
            assert (len(y) - 1)//self.p == len(error_ratios)
            assert (len(y) - 1)//self.p - 1 == len(error_ratios_2steps)
            #print(error_ratios)
            if verbose:
                print("ERROR1", error_ratios)
                print("ERROR2", error_ratios_2steps)
                print(integral_p, integral_p1)
            
            # Create mask for remove points that are too close
            """
            t0 = time.time()
            remove_mask = _find_excess_y(self.p, error_ratios_2steps, self.remove_cut)
            if speed_verbose: print("\t removal mask", time.time() - t0)
            assert (len(remove_mask) == len(t))
            if verbose:
                print("RCF", remove_mask)

            y_pruned = y[~remove_mask]
            t_pruned = t[~remove_mask]
            """

            # Find indices where error is too large to add new points
            # Evaluate integral
            t0 = time.time()
            _, y_p_pruned, _ = self._calculate_integral(
                t_pruned, y_pruned, y0=y0, degr=degree.P
            )
            _, y_p1_pruned, _ = self._calculate_integral(
                t_pruned, y_pruned, y0=y0, degr=degree.P1
            )
            if speed_verbose: print("\t calc integrals 2", time.time() - t0)
            
            # Calculate error
            error_ratios_pruned, _ = _compute_error_ratios(
                y_p_pruned, y_p1_pruned, self.rtol, self.atol, self._error_norm
            )

            t_add, idxs_add = _find_sparse_y(
                t_pruned, self.p, error_ratios_pruned
            )

            if speed_verbose: print("\tLOOP TIME", time.time() - tl0)

        self.previous_ode_fxn = ode_fxn.__name__
        self.t_previous = t
        return IntegralOutput(
            integral=integral_p1,
            t=t,
            h=h,
            y=y_p1,
            errors=torch.abs(y_p - y_p1),
            error_ratios=error_ratios,
            remove_mask=remove_mask
        )
    
    
class ParallelUniformAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.method_name in UNIFORM_METHODS
        self.method = UNIFORM_METHODS[self.method_name]
        self.p1 = self.method.order
        self.p = self.p1 - 1
 

    def _adaptive_t_steps(self, t, idxs_add):
        asdfas
        t_bisect = (t[idxs_add,0] + t[idxs_add,-1])/2.
        t_left = torch.concatenate(
            [t[idxs_add,0].unsqueeze(1), t_bisect.unsqueeze(1)], dim=1
        )
        t_right = torch.concatenate(
            [t_bisect.unsqueeze(1), t[idxs_add,-1].unsqueeze(1)], dim=1
        )
        return self._t_step_interpolate_uniform(t_left.flatten(), t_right.flatten())


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
            t_init (Tensor, optional) [D]: Minimum of integral range
            t_final (Tensor, optional) [D]: Maximum of integral range
        
        Shapes:
            t : [N, D] will populate intermediate evaluations according to
                integration method, [N, C, D] will retun t
            t_init: [D]
            t_final: [D]
        """
        
        if t is None:
            t = torch.linspace(0, 1., 7).unsqueeze(-1)
            t = t_init + t*(t_final - t_init)
        elif len(t.shape) == 3:
            return t
        return self._t_step_interpolate(t[:-1], t[1:])
 

    def _t_step_interpolate(self, t_left, t_right):
        dt = (t_right - t_left).unsqueeze(1)
        #steps = torch.arange(self.p1).unsqueeze(-1)
        steps = self.method.tableau.c.unsqueeze(-1)*dt
        #print(dt.shape, self.method.tableau.c.unsqueeze(-1).shape, steps.shape, t_left.shape)
        #steps = steps*dt/self.p
        return t_left.unsqueeze(1) + steps



    def _remove_excess_t(self, t, remove_idxs):
        """
        Merge two integration steps together through the time tensor

        Args:
            t (Tensor): Time points previously evaluated and will be pruned
            remove_idxs (Tensor): Index corresponding to the first time step
                in the contiguous pair to remove
        
        Shapes:
            t: [N, C, D]
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
    
    def _get_tableau_b(self, t):
        return self.method.tableau.b.unsqueeze(-1),\
            self.method.tableau.b_error.unsqueeze(-1)
    
    def _get_num_tableau_c(self):
        return len(self.method.tableau.c)



class ParallelVariableAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("METHOD", self.method_name)
        assert self.method_name in VARIABLE_METHODS
        self.method = VARIABLE_METHODS[self.method_name]()
        self.p1 = self.method.order 
        self.p = self.p1 - 1
    
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
            t_init (Tensor, optional) [D]: Minimum of integral range
            t_final (Tensor, optional) [D]: Maximum of integral range
        
        Shapes:
            t : [N, D] will populate intermediate evaluations according to
                integration method, [N, C, D] will retun t
            t_init: [D]
            t_final: [D]
        """
 
        if t is None:
            t = torch.linspace(0, 1., 7*self.p+1).unsqueeze(-1)
            t = t_init + t*(t_final - t_init)
        elif len(t.shape) == 3:
            return t
        else:
            error_message = f"Input time must be of length N*({self.method.order-1}) + 1, instead got {t.shape}"
            assert (len(t) - 1) % self.p == 0, error_message
        _t = torch.reshape(t[:-1], (-1, self.p, 1))
        print("INIT T", _t.shape, _t[1:,0].shape, t[None,-1].shape)
        _t_ends = torch.concatenate([_t[1:,0], t[None,-1]]).unsqueeze(1)
        return torch.concatenate([_t, _t_ends], dim=1)
    
    def _adaptive_t_steps(self, t, idxs_add):
        """
        Add points between current points to double the sampling
        """
        asdfsdfasdfd
        return (t[idxs_add,1:] +  t[idxs_add,:-1])/2
    
    def _remove_excess_t(self, t, remove_idxs):
        """
        Merge two integration steps together through the time tensor

        Args:
            t (Tensor): Time points previously evaluated and will be pruned
            remove_idxs (Tensor): Index corresponding to the first time step
                in the contiguous pair to remove
        
        Shapes:
            t: [N, C, D]
            remove_idxs: [R]
        """
        if len(remove_idxs) == 0:
            return t
        combined_steps = torch.concatenate(
            [t[remove_idxs,:], t[remove_idxs+1,1:]], dim=1
        )
        keep_idxs = torch.arange(self.p1, dtype=torch.long)*2
        t[remove_idxs+1] = combined_steps[:,keep_idxs]
        remove_mask = torch.ones(len(t), dtype=torch.bool)
        remove_mask[remove_idxs] = False
        return t[remove_mask]
    
    def _get_tableau_b(self, t):
        norm_dt = t - t[:,0,None]
        norm_dt = norm_dt/norm_dt[:,-1,None]
        b, b_error = self.method.tableau_b(norm_dt)
        return b.unsqueeze(-1), b_error.unsqueeze(-1)
    
    def _get_num_tableau_c(self):
        return self.method.n_tableau_c

