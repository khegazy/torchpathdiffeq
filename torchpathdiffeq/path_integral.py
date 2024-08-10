from .solvers import SerialAdaptiveStepsizeSolver, steps
from .runge_kutta import get_parallel_RK_solver



def ode_path_integral(ode_fxn, y0, t, method, atol, rtol, computation='parallel', sampling='uniform', remove_cut=0.1, t_init=0, t_final=1):

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
            t_final=t_final
        )
    elif computation.lower() == 'serial':
        integrator = SerialAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=ode_fxn,
            t_init=t_init,
            t_final=t_final
        )
    else:
        raise ValueError(f"Path integral computation type must be 'parallel' or 'serial', not {computation}.")

    integral_output = integrator.integrate(
        t_init=t_init,
        t_final=t_final
    )

    return integral_output