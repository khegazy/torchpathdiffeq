import torch
from .base import steps
from .serial_solver import SerialAdaptiveStepsizeSolver
from .runge_kutta import get_parallel_RK_solver



def ode_path_integral(
        ode_fxn,
        method,
        computation='parallel',
        sampling='uniform',
        atol=1e-5,
        rtol=1e-5,
        t=None,
        t_init=torch.tensor([0], dtype=torch.float64),
        t_final=torch.tensor([1], dtype=torch.float64),
        y0=torch.tensor([0], dtype=torch.float64),
        remove_cut=0.1,
        total_mem_usage=0.9,
        use_absolute_error_ratio=True,
        device=None,
        **kwargs
    ):
    """
    Integrate ode_fxn over either over t or from t_init to t_final. This
    instantiates the serial or parallel integration method specified by
    computation and method.

    Args:
        ode_fxn (Callable): The function to integrate over along the path
            parameterized by t
        method (str): Name of the integration method to use
        computation (str): Integral computation type: [parallel, serial]
        sampling (str): Sampling type for parallel integration methods:
            [uniform, variable]
        atol (float): The absolute tolerance when determining to add or remove
            integration steps
        rtol (float): The relative tolerance when determining to add or remove
            integration steps
        t (Tensor): Initial time points to evaluate ode_fxn and perform the
            numerical integration over
        t_init (Tensor): Initial integration time points
        t_final (Tensor): Final integration time points
        y0 (Tensor): Initial value of the integral
        ode_args (Tuple): Extra arguments provided to ode_fxn
        remove_cut (float): Cut to remove integration steps with error ratios
            less than this value, must be < 1
        max_batch (int): Maximum number of ode_fxn evaluations to hold in
            memory at a time
        use_absolute_error_ratio (bool): Use the total integration value when
            calulating the error ratio for adding or removing points, otherwise
            use integral value up to the time step being evaluated
        device (str): Name of the device to run the integration on
    
    Shapes:
        t: [N, C, T] or [N, T] for parallel integration, [N, T] for serial
            integration
        t_init: [T]
        t_final: [T]
        y0: [D]
    
    Note:
        Parallel Integration:
        If t is None it will be initialized with a few integration steps
        in the range [t_init, t_final]. If t is specified integration steps
        between t[0] and t[-1] may be removed and/or added, but the bounds
        of integration will remain [t[0], t[-1]]. If t is 2 dimensional the 
        intermediate time points will be calculated.

        Serial Integration:
        The integral is evaluated within the range [t[0], t[-1]] and
        returns the integral up to each specified point between. If t is
        None, it will be initialized as [t_init, t_final].
    """
    if computation.lower() == 'parallel':
        if sampling.lower() == 'uniform':
            sampling_type = steps.ADAPTIVE_UNIFORM
        elif sampling.lower() == 'variable':
            sampling_type = steps.ADAPTIVE_VARIABLE
        else:
            raise ValueError(f"Sampling method must be either 'uniform' or 'variable', instead got {sampling}")
        integrator = get_parallel_RK_solver(
            sampling_type=sampling_type,
            method=method,
            ode_fxn=ode_fxn,
            atol=atol,
            rtol=rtol,
            remove_cut=remove_cut,
            t_init=t_init,
            t_final=t_final,
            use_absolute_error_ratio=use_absolute_error_ratio,
            device=device,
            **kwargs
        )
        
        integral_output = integrator.integrate(
            y0=y0,
            t=t,
            t_init=t_init,
            t_final=t_final,
            total_mem_usage=total_mem_usage
        )
    elif computation.lower() == 'serial':
        integrator = SerialAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final,
            device=device,
            **kwargs
        )
    
        integral_output = integrator.integrate(
            y0=y0,
            t=t,
            t_init=t_init,
            t_final=t_final,
        )
    else:
        raise ValueError(f"Path integral computation type must be 'parallel' or 'serial', not {computation}.")

    return integral_output