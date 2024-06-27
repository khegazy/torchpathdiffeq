import torch
from numpy.testing import assert_allclose, assert_array_equal
from torchpathdiffeq import ode_path_integral, RKParallelAdaptiveStepsizeSolver
from torchpathdiffeq.path_integral import ADAPTIVE_SOLVER_P
from ..integrals.test_chemistry import wolf_schlegel

def test_ode_path_integral_fxn():
    method_name = 'euler'
    p=1
    atol = 1e-5
    rtol = 1e-7
    assert ADAPTIVE_SOLVER_P[method_name] == p
    
    OPI_integral = ode_path_integral(
        ode_fxn=wolf_schlegel,
        y0=0,
        t=None,
        method=method_name,
        atol=atol,
        rtol=rtol,
    )

    RK_integrator = RKParallelAdaptiveStepsizeSolver(
        p=p,
        atol=atol,
        rtol=rtol,
        ode_fxn=wolf_schlegel
    )
    RK_integral = RK_integrator.integrate()

    assert_allclose(OPI_integral.integral, RK_integral.integral)
    assert_allclose(OPI_integral.y, RK_integral.y)
    assert_allclose(OPI_integral.t, RK_integral.t)
    assert_allclose(OPI_integral.h, RK_integral.h)
    assert_allclose(OPI_integral.errors, RK_integral.errors)
    assert_allclose(OPI_integral.error_ratios, RK_integral.error_ratios)
    assert_array_equal(OPI_integral.remove_mask, RK_integral.remove_mask)
