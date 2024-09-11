import torch
from numpy.testing import assert_allclose, assert_array_equal
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

        assert_allclose(OPI_integral.integral, RK_integral.integral)
        assert_allclose(OPI_integral.integral_error, RK_integral.integral_error)
        assert_allclose(OPI_integral.t_pruned, RK_integral.t_pruned)
        assert_allclose(OPI_integral.y, RK_integral.y)
        assert_allclose(OPI_integral.t, RK_integral.t)
        assert_allclose(OPI_integral.h, RK_integral.h)
        assert_allclose(OPI_integral.sum_steps, RK_integral.sum_steps)
        assert_allclose(OPI_integral.errors, RK_integral.errors)
        assert_allclose(OPI_integral.error_ratios, RK_integral.error_ratios)

    ###############################
    #####  Parallel Variable  #####
    ###############################
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

        assert_allclose(OPI_integral.integral, RK_integral.integral)
        assert_allclose(OPI_integral.integral_error, RK_integral.integral_error)
        assert_allclose(OPI_integral.t_pruned, RK_integral.t_pruned)
        assert_allclose(OPI_integral.y, RK_integral.y)
        assert_allclose(OPI_integral.t, RK_integral.t)
        assert_allclose(OPI_integral.h, RK_integral.h)
        assert_allclose(OPI_integral.sum_steps, RK_integral.sum_steps)
        assert_allclose(OPI_integral.errors, RK_integral.errors)
        assert_allclose(OPI_integral.error_ratios, RK_integral.error_ratios)
    
    
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

        assert_allclose(OPI_integral.integral, RK_integral.integral)
        assert_allclose(OPI_integral.t, RK_integral.t)