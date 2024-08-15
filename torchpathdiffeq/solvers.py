import time
import torch
import torch.distributed as dist
import numpy as np
from torchdiffeq import odeint
from dataclasses import dataclass
from enum import Enum

from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .adaptivity import IntegralAdaptivity


class steps(Enum):
    FIXED = 0
    ADAPTIVE_UNIFORM = 1
    ADAPTIVE_VARIABLE = 2

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
    def __init__(
            self,
            method,
            atol,
            rtol,
            y0=torch.tensor([0], dtype=torch.float64),
            ode_fxn=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
        ) -> None:

        self.method_name = method.lower()
        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.y0 = y0
        self.t_init = t_init
        self.t_final = t_final
        print("MAKE SURE ATOL RTOL MATCH 1", atol, rtol)

    def _calculate_integral(self, t, y, y0=torch.tensor([0], dtype=torch.float64)):
        """
        Internal integration method of a specific numerical integration scheme,
        e.g. Runge-Kutta, that carries out the method on the given time (t) and
        ode_fxn evaluation points (y).

        Args:
            t (Tensor): Evaluation time steps for the RK integral
            y (Tensor): Evalutions of the integrad at time steps t
            y0 (Tensor): Initial values of the integral
        
        Shapes:
            t: [N, C, T]
            y: [N, C, D]
            y0: [D]
        """
        raise NotImplementedError
    
    def integrate(
            self,
            ode_fxn,
            y0=torch.tensor([0], dtype=torch.float64),
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            t=None,
            ode_args=()
        ):
        """
        Perform the numerical path integral on ode_fxn over a path
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
        """
        raise NotImplementedError




class SerialAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, *args, **kwargs):
        #method, atol, rtol, y0=torch.tensor([0], dtype=torch.float), ode_fxn=None, t_init=0, t_final=1.) -> None:
        super().__init__(*args, **kwargs)
        """
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            y0=y0,
            t_init=t_init,
            t_final=t_final
        )
        """

    
    def integrate(self, ode_fxn=None, y0=None, t_init=0., t_final=1., t=None, ode_args=None):
        """
        Perform the sequential numerical path integral on ode_fxn over a path
        parameterized by time (t), which ranges from t_init to t_final. This 
        is done by using the torchdiffeq odeint function.

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
        """
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        if t is None:
            t=torch.tensor([t_init, t_final])
        
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
    
    def __init__(self, *args, **kwargs):
        """
            method,
            atol,
            rtol,
            y0=torch.tensor([0], dtype=torch.float64),
            ode_fxn=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
        ):
        """
        super(SolverBase, self).__init__(*args, **kwargs)
        super(IntegralAdaptivity, self).__init__(*args, **kwargs)
        """
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            y0=y0,
            t_init=t_init,
            t_final=t_final
        )
        """
        self.method = None
        self.p1 = None
        self.p = None
        self.previous_t = None
        self.previous_ode_fxn = None

    
    def _error_norm(self, error):
        """
        Normalize multivariate errors to determine the step's total error
        """
        return torch.sqrt(torch.mean(error**2, -1))
    
    
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
            #integral_p1, sum_steps_p1, h = self._calculate_integral(t, y, y0=y0, degr=degree.P1)
            if verbose_speed: print("\t calc integrals 1", time.time() - t0)
            
            # Calculate error
            t0 = time.time()
            #print("YP SHAPES", y_p.shape, y_p1.shape)
            error_ratios, error_ratios_2steps = self.compute_error_ratios(
                method_output.sum_steps, method_output.sum_step_errors
            )
            print("ERRORS RATIOS", len(error_ratios), torch.sum(error_ratios>1))
            #print("ER SHAPES", error_ratios.shape, error_ratios_2steps.shape, y.shape, t.shape)
            if verbose_speed: print("\t calculate errors", time.time() - t0)
            assert len(y) == len(error_ratios)
            assert len(y) - 1 == len(error_ratios_2steps)
            #print(error_ratios)
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
        assert self.method_name in UNIFORM_METHODS
        self.method = UNIFORM_METHODS[self.method_name]
        self.p1 = self.method.order
        self.p = self.p1 - 1
 

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
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
        
        Shapes:
            t : [N, T] will populate intermediate evaluations according to
                intetgration method, [N, C, T] will retun t
            t_init: [T]
            t_final: [T]
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
        keep_idxs = torch.arange(self.p1, dtype=torch.long)*2
        t[remove_idxs+1] = combined_steps[:,keep_idxs]
        remove_mask = torch.ones(len(t), dtype=torch.bool)
        remove_mask[remove_idxs] = False
        return t[remove_mask]
