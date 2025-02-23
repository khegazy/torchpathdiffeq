import torch
import numpy as np
from torchpathdiffeq import ode_path_integral, UNIFORM_METHODS, RKParallelUniformAdaptiveStepsizeSolver, RKParallelVariableAdaptiveStepsizeSolver

def integrand(t):
    return torch.exp(-5*(t-0.5)**2)*4*torch.cos(3*t**2)

def test_removal():
    dense_t = torch.linspace(0, 1, 997).unsqueeze(1)
    atol = 1e-5
    rtol = 1e-5

    uniform_integrator = RKParallelUniformAdaptiveStepsizeSolver(
        method='dopri5', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    variable_integrator = RKParallelVariableAdaptiveStepsizeSolver(
        method='generic3', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    for type, integrator in zip(['Uniform', 'Variable'], [uniform_integrator, variable_integrator]):
        t = dense_t
        for idx in range(3):
            integral_output = integrator.integrate(t=t)
            t_optimal = integral_output.t_optimal
            if idx < 1:
                error_message = f"For {type} integrator: length of t {t.shape} shoud be > to t_optimal {t_optimal.shape}"
                assert len(t) > len(t_optimal), error_message
            else:
                error_message = f"For {type} integrator: length of t {t.shape} shoud be >= to t_optimal {t_optimal.shape}"
                assert len(t) >= len(t_optimal), error_message
            t_flat = torch.flatten(integral_output.t, start_dim=0, end_dim=1)
            t_optimal_flat = torch.flatten(t_optimal, start_dim=0, end_dim=1)
            assert torch.all(t_flat[1:] - t_flat[:-1] >= 0)
            assert np.allclose(integral_output.t[:-1,-1,:], integral_output.t[1:,0,:])
            t = t_optimal
