import time
import torch
import torch.distributed as dist
import numpy as np
from torchdiffeq import odeint
from dataclasses import dataclass
from enum import Enum

from .adaptivity import _adaptively_add_y, _find_excess_y, _find_sparse_y, _compute_error_ratios


class steps(Enum):
    FIXED = 0
    ADAPTIVE = 1

class degree(Enum):
    P = 0
    P1 = 1

ADAPTIVE_SOLVER_P = {
    'euler' : 1,
    'heun' : 1,
    'generic3' : 2
}

@dataclass
class IntegralOutput():
    integral: torch.Tensor
    t: torch.Tensor
    h: torch.Tensor
    y: torch.Tensor
    errors: torch.Tensor
    error_ratios: torch.Tensor
    remove_mask: torch.Tensor
    

class SolverBase():
    def __init__(self, solver, atol, rtol, y0=torch.tensor([0], dtype=torch.float), ode_fxn=None, t_init=0., t_final=1.) -> None:

        self.solver = solver.lower()
        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.y0 = y0
        self.t_init = t_init
        self.t_final = t_final

    def _calculate_integral(self, t, y, y0=0, degr=degree.P1):
        raise NotImplementedError
    
    def integrate(self, ode_fxn, y0=None, t_init=0., t_final=1., t=None, ode_args=None):
        raise NotImplementedError

    def _error_norm(self, error):
        return torch.sqrt(torch.mean(error**2, -1))



class SerialAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, solver, atol, rtol, y0=torch.tensor([0], dtype=torch.float), ode_fxn=None, t_init=0, t_final=1.) -> None:
        super().__init__(
            solver=solver,
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
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )

        return IntegralOutput(
            integral=integral[-1],
            t=t,
            h=None,
            y=None,
            errors=None,
            error_ratios=None,
            remove_mask=None
        )



class ParallelAdaptiveStepsizeSolver(SolverBase):
    
    def __init__(self, solver, atol, rtol, remove_cut=0.1, y0=torch.tensor([0], dtype=torch.float), ode_fxn=None, t_init=0, t_final=1.):
        super().__init__(
            solver=solver,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            y0=y0,
            t_init=t_init,
            t_final=t_final
        )

        assert solver in ADAPTIVE_SOLVER_P,\
            f"Do not recognize solver {self.solver}, choose from: {list(ADAPTIVE_SOLVER_P.keys())}"
        self.p = ADAPTIVE_SOLVER_P[self.solver]
        self.atol = atol
        self.rtol = rtol
        self.remove_cut = remove_cut
        self.previous_t = None
        self.previous_ode_fxn = None

    def integrate(self, ode_fxn=None, state=None, y0=None, t_init=0., t_final=1., t=None, ode_args=None, remove_mask=None):
        verbose=False
        speed_verbose=False
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        assert len(ode_fxn(torch.tensor([[t_init]])).shape) >= 2
        if state is not None:
            t = state.t
            remove_mask = state.remove_mask
        if remove_mask is not None:
            t = t[~remove_mask]
        
        
        t_add = t
        if t_add is None:
            if self.previous_t is None\
                or self.previous_ode_fxn != ode_fxn.__name__:
                t_add = torch.unsqueeze(
                    torch.linspace(t_init, t_final, 1+self.p*13), 1
                )
            else:
                mask = (self.previous_t[:,0] <= t_final)\
                    + (self.previous_t[:,0] >= t_init)
                t_add = self.previous_t[mask]

        y_pruned, t_pruned = None, None
        idxs_add = torch.arange(len(t_add))
        loop_cnt = 0
        while len(t_add) > 0:
            tl0 = time.time()
            #print("START OF LOOP", loop_cnt)
            if verbose:
                print("BEGINNING LOOP")
                print("TADD", t_add.shape, t_add, idxs_add)
            # Evaluate new points and add new evals and points to arrays
            t0 = time.time()
            y, t = _adaptively_add_y(
                ode_fxn, y_pruned, t_pruned, t_add, idxs_add
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
            t0 = time.time()
            remove_mask = _find_excess_y(self.p, error_ratios_2steps, self.remove_cut)
            if speed_verbose: print("\t removal mask", time.time() - t0)
            assert (len(remove_mask) == len(t))
            if verbose:
                print("RCF", remove_mask)

            y_pruned = y[~remove_mask]
            t_pruned = t[~remove_mask]

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
