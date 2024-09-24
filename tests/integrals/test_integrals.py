import torch
import numpy as np

from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    SerialAdaptiveStepsizeSolver,\
    ode_path_integral,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS\

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
    "t" : (t, t_solution, 1e-10),
    "t_squared" : (t_squared, t_squared_solution, 1e-5),
    "sine_squared" : (sine_squared, sine_squared_solution, 5e-2),
    "exp" : (exp, exp_solution, 1e-5),
    "damped_sine" : (damped_sine, damped_sine_solution, 2.)
}

def test_integrals():
    atol = 1e-9
    rtol = 1e-7
    t_init = torch.tensor([0], dtype=torch.float64)
    t_final = torch.tensor([1], dtype=torch.float64)
    max_batches = [None, 512, 15]
    loop_items = zip(
        ['Uniform', 'Variable'],
        [UNIFORM_METHODS, VARIABLE_METHODS],
        [steps.ADAPTIVE_UNIFORM, steps.ADAPTIVE_VARIABLE]
    )
    for sampling_name, sampling, sampling_type in loop_items:
        for method in sampling.keys():
            #if method != 'fehlberg2':
            #    continue
            for name, (ode, solution, cutoff) in ODE_dict.items():
                correct = solution(t_init=t_init, t_final=t_final)
                for max_batch in max_batches:
                    #if max_batch != 7:
                    #    continue
                    print("STARTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", method, name, max_batch)
                    parallel_integrator = get_parallel_RK_solver(
                        sampling_type, method=method, atol=atol, rtol=rtol, remove_cut=0.1
                    )
                    integral_output = parallel_integrator.integrate(
                        ode, t_init=t_init, t_final=t_final, max_batch=max_batch
                    )
                    #cutoff = 10000*10**(-1*parallel_integrator.order)
                    
                    error_string = f"{sampling_name} {method} failed to properly integrate {name} with max_batch {max_batch}, calculated {integral_output.integral.item()} but expected {correct.item()}"
                    t_flat = torch.flatten(integral_output.t[:,:,0])
                    t_flat_unique = torch.flatten(integral_output.t[:,1:,0])
                    print("VALS", integral_output.integral, correct, integral_output.t.shape, torch.sum(t_flat[1:] - t_flat[:-1] + 1e-15 < 0), torch.sum(t_flat_unique[1:] - t_flat_unique[:-1] + 1e-15 < 0))
                    print("CUTOFF", cutoff, torch.abs((integral_output.integral - correct)/correct))
                    """
                    serial_integral = ode_path_integral(
                        ode_fxn=ode,
                        method=method,
                        computation='serial',
                        atol=atol,
                        rtol=rtol,
                        y0=torch.tensor([0], dtype=torch.float64),
                        t=None,
                    )
                    print("SERIAL", serial_integral.integral, torch.abs((serial_integral.integral - correct)/correct))
                    """
                    assert torch.abs((integral_output.integral - correct)/correct) < cutoff, error_string

                    t_flat = torch.flatten(integral_output.t, start_dim=0, end_dim=1)
                    t_pruned_flat = torch.flatten(integral_output.t_pruned, start_dim=0, end_dim=1)
                    assert torch.all(t_flat[1:] - t_flat[0:-1] >= 0)
                    assert torch.all(t_pruned_flat[1:] - t_pruned_flat[0:-1] >= 0)
                    assert np.allclose(integral_output.t[1:,0,:], integral_output.t[:-1,-1,:])
                    assert np.allclose(integral_output.t_pruned[1:,0,:], integral_output.t_pruned[:-1,-1,:])
                    
                    
                    if max_batch is None:
                        no_batch_integral = integral_output
                        no_batch_delta = torch.abs(no_batch_integral.integral - correct)
                    else:
                        rel_tol = 1e-3
                        if torch.abs(integral_output.integral - correct) > no_batch_delta:
                            print(no_batch_integral.integral, integral_output.integral)
                            assert torch.abs(1 - (no_batch_integral.integral/integral_output.integral)) < rel_tol
                            assert 10*torch.abs(no_batch_integral.integral_error) >= torch.abs(integral_output.integral_error)
                        assert len(no_batch_integral.t) <= len(integral_output.t)
                        assert len(no_batch_integral.t_pruned) <= len(integral_output.t_pruned) 
        break

#test_integrals()
