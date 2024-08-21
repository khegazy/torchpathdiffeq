import torch 
from torchpathdiffeq import ode_path_integral, UNIFORM_METHODS, RKParallelUniformAdaptiveStepsizeSolver, RKParallelVariableAdaptiveStepsizeSolver

def integrand(t):
    return torch.exp(-5*(t-0.5)**2)*4*torch.cos(3*t**2)

def test_removal():
    dense_t = torch.linspace(0, 1, 999).unsqueeze(1)
    atol = 1e-5
    rtol = 1e-5

    uniform_integrator = RKParallelUniformAdaptiveStepsizeSolver(
        method='dopri5', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    variable_integrator = RKParallelVariableAdaptiveStepsizeSolver(
        method='generic3', ode_fxn=integrand, atol=atol, rtol=rtol
    )
    for type, integrator in zip(['Uniform', 'Variable'], [uniform_integrator, variable_integrator]):
        print(type)
        t = dense_t
        for idx in range(3):
            integral_output = integrator.integrate(t=t)
            if idx < 1:
                error_message = f"For {type} integrator: length of t {t.shape} shoud be > to t_pruned {integral_output.t_pruned.shape}"
                assert len(t) > len(integral_output.t_pruned), error_message
            else:
                error_message = f"For {type} integrator: length of t {t.shape} shoud be >= to t_pruned {integral_output.t_pruned.shape}"
                assert len(t) >= len(integral_output.t_pruned), error_message
            t = integral_output.t_pruned