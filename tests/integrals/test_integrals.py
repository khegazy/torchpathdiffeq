import torch

from torchpathdiffeq import RKParallelAdaptiveStepsizeSolver
from torchpathdiffeq.solvers import ADAPTIVE_SOLVER_P

integrator_list = [RKParallelAdaptiveStepsizeSolver]

def identity(t):
    return 1

def identity_solution(t_init, t_final):
    return t_final - t_init

def t(t):
    return t

def t_solution(t_init, t_final):
    return 0.5*(t_final**2 - t_init**2)

def t_squared(t):
    return t**2

def t_squared_solution(t_init, t_final):
    return (t_final**3 - t_init**3)/3.

def sine_squared(t, w=3.7):
    return torch.sin(t*w*2*torch.pi)**2

def sine_squared_solution(t_init, t_final, w=3.7):
    _w = 4*torch.pi*w
    return (t_final - t_init)/2.\
        + (torch.sin(torch.tensor([_w*t_final])) - torch.sin(torch.tensor([_w*t_init])))/(2*_w)

def exp(t, a=5):
    return torch.exp(a*t)

def exp_solution(t_init, t_final, a=5):
    return (torch.exp(torch.tensor([t_final*a]))\
        - torch.exp(torch.tensor([t_init*a])))/a

def damped_sine(t, w=3.7, a=5):
    return torch.exp(-a*t)*torch.sin(w*t*2*torch.pi)

def damped_sine_solution(t_init, t_final, w=3.7, a=0.2):
    _w = 2*torch.pi*w
    def numerator(t, w, a):
        t = torch.tensor([t])
        return torch.exp(-a*t)*(torch.sin(_w*t) + _w*torch.cos(_w*t))
    return (numerator(t_final, w, a) - numerator(t_init, w, a))/(a**2 + _w**2)


ODE_dict = {
    "t" : (t, t_solution),
    "t_squared" : (t_squared, t_squared_solution),
    "sine_squared" : (sine_squared, sine_squared_solution),
    "exp" : (exp, exp_solution),
    "damped_sine" : (damped_sine, damped_sine_solution)
}

def test_integrals():
    cutoff = 0.05
    t_init = 0.
    t_final = 1.
    for solver in ADAPTIVE_SOLVER_P.keys():
        print("SOLVER", solver)
        integrator = RKParallelAdaptiveStepsizeSolver(
            solver=solver, atol=1e-7, rtol=1e-7, remove_cut=0.105
        )
        for name, (ode, solution) in ODE_dict.items():
            integral_output = integrator.integrate(ode, t_init=t_init, t_final=t_final)
            correct = solution(t_init=t_init, t_final=t_final)
            error_string = f"{solver} failed to properly integrate {name}, calculated {integral_output.integral} but expected {correct}"

            assert torch.abs(integral_output.integral - correct)/correct < cutoff, error_string
            #print(name, integral_output.h.shape)

test_integrals()