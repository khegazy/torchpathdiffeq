import torch 
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
    sparse_t = torch.linspace(0, 1, 3).unsqueeze(1)
    atol = 1e-5
    rtol = 1e-5
    t_init = 0
    t_final = 1

    correct = damped_sine_solution(t_init, t_final)
    uniform_integrator = RKParallelUniformAdaptiveStepsizeSolver(
        method='adaptive_heun', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    variable_integrator = RKParallelVariableAdaptiveStepsizeSolver(
        method='generic3', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    for type, integrator in zip(['Uniform', 'Variable'], [uniform_integrator, variable_integrator]):
        t = sparse_t
        for idx in range(3):
            integral_output = integrator.integrate(t=t)
            assert (integral_output.integral - correct)/correct < atol
            if idx < 1:
                error_message = f"For {type} integrator: length of t {t.shape} shoud be < to t_pruned {integral_output.t_pruned.shape}"
                assert len(t) < len(integral_output.t_pruned), error_message
            else:
                error_message = f"For {type} integrator: length of t {t.shape} shoud be <= to t_pruned {integral_output.t_pruned.shape}"
                assert len(t) <= len(integral_output.t_pruned), error_message
            t = integral_output.t_pruned

test_adding()
