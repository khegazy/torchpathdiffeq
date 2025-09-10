import torch
from torchpathdiffeq import ode_path_integral, UNIFORM_METHODS, VARIABLE_METHODS, RKParallelUniformAdaptiveStepsizeSolver, RKParallelVariableAdaptiveStepsizeSolver, SerialAdaptiveStepsizeSolver

def integrand(t, y=0):
    return torch.exp(-5*(t-0.5)**2)*4*torch.cos(3*t**2)

def test_ode_path_integral_fxn():
    atol = 1e-9
    rtol = 1e-7

    ##############################
    #####  Parallel Uniform  #####
    ##############################
    for method in UNIFORM_METHODS.keys():
    
        OPI_integral = ode_path_integral(
            ode_fxn=integrand,
            method=method,
            computation='parallel',
            sampling='uniform',
            atol=atol,
            rtol=rtol,
            y0=torch.tensor([0], dtype=torch.float64),
            t=None,
        )

        RK_integrator = RKParallelUniformAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=integrand
        )
        RK_integral = RK_integrator.integrate()

        torch.allclose(OPI_integral.integral, RK_integral.integral)
        torch.allclose(OPI_integral.integral_error, RK_integral.integral_error)
        torch.allclose(OPI_integral.t_optimal, RK_integral.t_optimal)
        torch.allclose(OPI_integral.y, RK_integral.y)
        torch.allclose(OPI_integral.t, RK_integral.t)
        torch.allclose(OPI_integral.h, RK_integral.h)
        torch.allclose(OPI_integral.sum_steps, RK_integral.sum_steps)
        torch.allclose(OPI_integral.sum_step_errors, RK_integral.sum_step_errors)
        torch.allclose(OPI_integral.error_ratios, RK_integral.error_ratios)

    ###############################
    #####  Parallel Variable  #####
    ###############################
    """
    for method in VARIABLE_METHODS.keys():
    
        OPI_integral = ode_path_integral(
            ode_fxn=integrand,
            method=method,
            computation='parallel',
            sampling='variable',
            atol=atol,
            rtol=rtol,
            y0=torch.tensor([0], dtype=torch.float64),
            t=None,
        )

        RK_integrator = RKParallelVariableAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=integrand
        )
        RK_integral = RK_integrator.integrate()

        torch.allclose(OPI_integral.integral, RK_integral.integral)
        torch.allclose(OPI_integral.integral_error, RK_integral.integral_error)
        torch.allclose(OPI_integral.t_optimal, RK_integral.t_optimal)
        torch.allclose(OPI_integral.y, RK_integral.y)
        torch.allclose(OPI_integral.t, RK_integral.t)
        torch.allclose(OPI_integral.h, RK_integral.h)
        torch.allclose(OPI_integral.sum_steps, RK_integral.sum_steps)
        torch.allclose(OPI_integral.sum_step_errors, RK_integral.sum_step_errors)
        torch.allclose(OPI_integral.error_ratios, RK_integral.error_ratios)
    
    """
    
    ####################
    #####  Serial  #####
    ####################
    for method in ['adaptive_heun', 'fehlberg2', 'bosh3', 'rk4', 'dopri5']:
    
        OPI_integral = ode_path_integral(
            ode_fxn=integrand,
            method=method,
            computation='serial',
            atol=atol,
            rtol=rtol,
            y0=torch.tensor([0], dtype=torch.float64),
            t=None,
        )

        RK_integrator = SerialAdaptiveStepsizeSolver(
            method=method,
            atol=atol,
            rtol=rtol,
            ode_fxn=integrand
        )
        RK_integral = RK_integrator.integrate()

        torch.allclose(OPI_integral.integral, RK_integral.integral)
        torch.allclose(OPI_integral.t, RK_integral.t)