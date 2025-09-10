import torch

def identity(t, y=None):
    return 1

def identity_solution(t_init, t_final):
    return t_final - t_init

def t(t, y=None):
    return t

def t_solution(t_init, t_final):
    return 0.5*(t_final**2 - t_init**2)

def t_squared(t, y=None):
    return t**2

def t_squared_solution(t_init, t_final):
    return (t_final**3 - t_init**3)/3.

def sine_squared(t, w=3.7, y=None):
    return torch.sin(t*w*2*torch.pi)**2

def sine_squared_solution(t_init, t_final, w=3.7):
    _w = 4*torch.pi*w
    return (t_final - t_init)/2.\
        + (torch.sin(torch.tensor([_w*t_final])) - torch.sin(torch.tensor([_w*t_init])))/(2*_w)

def exp(t, a=5, y=None):
    return torch.exp(a*t)

def exp_solution(t_init, t_final, a=5):
    return (torch.exp(torch.tensor([t_final*a]))\
        - torch.exp(torch.tensor([t_init*a])))/a

def damped_sine(t, w=3.7, a=5, y=None):
    return torch.exp(-a*t)*torch.sin(w*t*2*torch.pi)

def damped_sine_solution(t_init, t_final, w=3.7, a=0.2):
    _w = 2*torch.pi*w
    def numerator(t, w, a):
        t = torch.tensor([t])
        return torch.exp(-a*t)*(torch.sin(_w*t) + _w*torch.cos(_w*t))
    return (numerator(t_final, w, a) - numerator(t_init, w, a))/(a**2 + _w**2)


ODE_dict = {
    "t" : (t, t_solution, 1e-7),
    "t_squared" : (t_squared, t_squared_solution, 1e-5),
    "sine_squared" : (sine_squared, sine_squared_solution, 5e-2),
    "exp" : (exp, exp_solution, 1e-5),
    "damped_sine" : (damped_sine, damped_sine_solution, 2.)
}