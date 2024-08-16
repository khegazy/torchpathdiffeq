import torch
from .base import steps
from .serial_solver import SerialAdaptiveStepsizeSolver
from .runge_kutta import get_parallel_RK_solver



def ode_path_integral(
        ode_fxn,
        t,
        method,
        atol,
        rtol,
        t_init=torch.tensor([0], dtype=torch.float64),
        t_final=torch.tensor([1], dtype=torch.float64),
        y0=torch.tensor([0], dtype=torch.float64),
        computation='parallel',
        sampling='uniform',
        remove_cut=0.1,
        use_absolute_error_ratio=True
    ):

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
            use_absolute_error_ratio=use_absolute_error_ratio
        )
    elif computation.lower() == 'serial':
        integrator = SerialAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final,
            use_absolute_error_ratio=use_absolute_error_ratio
        )
    else:
        raise ValueError(f"Path integral computation type must be 'parallel' or 'serial', not {computation}.")

    integral_output = integrator.integrate(
        t_init=t_init,
        t_final=t_final
    )

    return integral_output