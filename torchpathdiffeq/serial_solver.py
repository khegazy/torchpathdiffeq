import time
import torch
from torchdiffeq import odeint

from .base import SolverBase, IntegralOutput


class SerialAdaptiveStepsizeSolver(SolverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
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
            t: [N, T]
            t_init: [T]
            t_final: [T]
        
        Note:
            The integral is evaluated within the range [t[0], t[-1]] and
            returns the integral up to each specified point between. If t is
            None, it will be initialized as [t_init, t_final].
        """
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        y0 = self.y0 if y0 is None else y0
        assert ode_fxn is not None, "Must specify ode_fxn or pass it during class initialization."
        if t is None:
            t=torch.tensor([t_init, t_final])
        else:
            assert len(t.shape) == 2
        
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