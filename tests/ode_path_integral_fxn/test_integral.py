import torch
from numpy.testing import assert_allclose, assert_array_equal
from torchpathdiffeq import ode_path_integral, UNIFORM_METHODS, RKParallelUniformAdaptiveStepsizeSolver, RKParallelVariableAdaptiveStepsizeSolver
#from ..integrals.test_chemistry import wolf_schlegel

def integrand(t):
    return torch.exp(-5*(t-0.5)**2)*4*torch.cos(3*t**2)

def test_ode_path_integral_fxn():
    atol = 1e-5
    rtol = 1e-7
    for method in UNIFORM_METHODS.keys():
    
        OPI_integral = ode_path_integral(
            ode_fxn=integrand,
            y0=0,
            t=None,
            method=method,
            atol=atol,
            rtol=rtol,
            computation='parallel',
            sampling='uniform'
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

test_ode_path_integral_fxn()