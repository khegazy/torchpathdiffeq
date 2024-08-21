import torch
from dataclasses import dataclass
from enum import Enum

class steps(Enum):
    FIXED = 0
    ADAPTIVE_UNIFORM = 1
    ADAPTIVE_VARIABLE = 2

def get_sampling_type(sampling_type : str):
    """
    Convert string sampling type into steps enum type
    """
    types = {
        'fixed' : steps.FIXED,
        'adaptive_uniform' : steps.ADAPTIVE_UNIFORM,
        'uniform' : steps.ADAPTIVE_UNIFORM,
        'adaptive_variable' : steps.ADAPTIVE_VARIABLE,
        'variable' : steps.ADAPTIVE_VARIABLE
    }
    return types[sampling_type]

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
            device=None
        ) -> None:

        self.method_name = method.lower()
        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.y0 = y0
        self.t_init = t_init
        self.t_final = t_final
        self.device = device

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
            t: [N, C, T] or [N, T] for parallel integration, [N, T] for serial
                integration
            t_init: [T]
            t_final: [T]
        
        Note:
            Handling of the input time t is different across methods, see
            the method's documentions for detail.
        """
        raise NotImplementedError
