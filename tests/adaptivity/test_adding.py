import torch
import numpy as np
from torchpathdiffeq import ode_path_integral, UNIFORM_METHODS, RKParallelUniformAdaptiveStepsizeSolver, RKParallelVariableAdaptiveStepsizeSolver

def integrand(t):
    return torch.exp(-5*(t-0.5)**2)*4*torch.cos(3*t**2)


def damped_sine(t, w=3.7, a=5):
    return torch.exp(-a*t)*torch.sin(w*t*2*torch.pi)

def damped_sine_solution(t_init, t_final, w=3.7, a=0.2):
    _w = 2*torch.pi*w
    def numerator(t, w, a):
        t = torch.tensor([t])
        return torch.exp(-a*t)*(torch.sin(_w*t) + _w*torch.cos(_w*t))
    return (numerator(t_final, w, a) - numerator(t_init, w, a))/(a**2 + _w**2)


def test_adding():
    atol = 1e-9
    rtol = 1e-7
    t_init = 0
    t_final = 1

    correct = damped_sine_solution(t_init, t_final)
    uniform_heun_integrator = RKParallelUniformAdaptiveStepsizeSolver(
        method='adaptive_heun', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    uniform_dopri5_integrator = RKParallelUniformAdaptiveStepsizeSolver(
        method='dopri5', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    variable_integrator = RKParallelVariableAdaptiveStepsizeSolver(
        method='generic3', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    #loop = zip(
    #    ['Uniform', 'Uniform', 'Variable'],
    #    ['adaptive_heun', 'dopri5', 'generic3'],
    #    [uniform_heun_integrator, uniform_dopri5_integrator, variable_integrator]
    #)
    loop = zip(
        ['Uniform', 'Uniform'],
        ['adaptive_heun', 'dopri5'],
        [uniform_heun_integrator, uniform_dopri5_integrator]
    )
    for type, method, integrator in loop:
        t = torch.linspace(0, 1., integrator.Cm1+1).unsqueeze(1)
        for idx in range(3):
            integral_output = integrator.integrate(t=t)
            assert (integral_output.integral - correct)/correct < atol
            t_optimal = integral_output.t_optimal
            if idx < 1:
                error_message = f"For {type} integrator method {method}: length of t {t.shape} shoud be < to t_optimal {t_optimal.shape}"
                assert len(t) < len(integral_output.t_optimal), error_message
            else:
                error_message = f"For {type} integrator method {method}: length of t {t.shape} shoud be <= to t_optimal {t_optimal.shape}"
                assert len(t) <= len(integral_output.t_optimal), error_message
            t_flat = torch.flatten(integral_output.t, start_dim=0, end_dim=1)
            t_optimal_flat = torch.flatten(t_optimal, start_dim=0, end_dim=1)
            assert torch.all(t_flat[1:] - t_flat[:-1] >= 0)
            assert np.allclose(integral_output.t[:-1,-1,:], integral_output.t[1:,0,:])
            t = t_optimal