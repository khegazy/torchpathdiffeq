import torch
from dataclasses import dataclass
from enum import Enum

from .distributed import DistributedEnvironment

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
    loss: torch.Tensor = None
    gradient_taken: bool = None
    t_optimal: torch.Tensor = None
    t: torch.Tensor = None
    h: torch.Tensor = None
    y: torch.Tensor = None
    sum_steps: torch.Tensor = None
    integral_error: torch.Tensor = None
    sum_step_errors: torch.Tensor = None
    error_ratios: torch.Tensor = None
    t_init: torch.Tensor = None
    t_final: torch.Tensor = None
    y0: torch.Tensor = None

@dataclass
class MethodOutput():
    integral: torch.Tensor
    integral_error: torch.Tensor
    sum_steps: torch.Tensor
    sum_step_errors: torch.Tensor
    h: torch.Tensor


class SolverBase(DistributedEnvironment):
    def __init__(
            self,
            method,
            atol,
            rtol,
            y0=torch.tensor([0], dtype=torch.float64),
            ode_fxn=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            dtype=torch.float64,
            eval=False,
            device=None,
            *args,
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs, device_type=device)

        self.method_name = method.lower()
        self.atol = atol
        self.rtol = rtol
        self.ode_fxn = ode_fxn
        self.dtype = dtype
        self.y0 = y0.to(self.dtype).to(self.device)
        self.t_init = t_init.to(self.dtype).to(self.device)
        self.t_final = t_final.to(self.dtype).to(self.device)
        self.training = not eval

        if self.dtype == torch.float64:
            self.atol_assert = 1e-15
            self.rtol_assert = 1e-7
        elif self.dtype == torch.float32:
            self.atol_assert = 1e-7
            self.rtol_assert = 1e-5
        elif self.dtype == torch.float16:
            self.atol_assert = 1e-3
            self.rtol_assert = 1e-1
        else:
            raise ValueError("Given dtype must be torch.float64 or torch.float32")


    def _check_variables(self, ode_fxn=None, t_init=None, t_final=None, y0=None):
        """ Replaces missing values with defaults on correct device """
        ode_fxn = self.ode_fxn if ode_fxn is None else ode_fxn
        t_init = self.t_init if t_init is None else t_init
        t_final = self.t_final if t_final is None else t_final
        y0 = self.y0 if y0 is None else y0

        if t_init is not None:
            t_init = t_init.to(self.dtype).to(self.device)
        if t_final is not None:
            t_final = t_final.to(self.dtype).to(self.device)
        if y0 is not None:
            y0 = y0.to(self.dtype).to(self.device)
        return ode_fxn, t_init, t_final, y0

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


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
    
    def _integral_loss(self, integral, *args, **kwargs):
        return integral.integral
    
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
    

    def __del__(self):
        """
        Class destructor terminates distributed process group
        """
        self.end_process()
