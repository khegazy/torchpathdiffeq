import torch
import numpy as np

from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    SerialAdaptiveStepsizeSolver,\
    ode_path_integral,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS,\
    ODE_dict


def test_integrals():
    atol = 1e-9
    rtol = 1e-7
    t_init = torch.tensor([0], dtype=torch.float64)
    t_final = torch.tensor([1], dtype=torch.float64)
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
                print("INTEGRAL", method, name, sampling_name)
                correct = solution(t_init=t_init, t_final=t_final)
                parallel_integrator = get_parallel_RK_solver(
                    sampling_type, method=method, atol=atol, rtol=rtol, remove_cut=0.1
                )
                integral_output = parallel_integrator.integrate(
                    ode, t_init=t_init, t_final=t_final
                )
                #cutoff = 10000*10**(-1*parallel_integrator.order)
                
                error_string = f"{sampling_name} {method} failed to properly integrate {name}, calculated {integral_output.integral.item()} but expected {correct.item()}"
                t_flat = torch.flatten(integral_output.t[:,:,0])
                t_flat_unique = torch.flatten(integral_output.t[:,1:,0])
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
                t_optimal_flat = torch.flatten(integral_output.t_optimal, start_dim=0, end_dim=1)
                assert torch.all(t_flat[1:] - t_flat[0:-1] >= 0)
                assert torch.all(t_optimal_flat[1:] - t_optimal_flat[0:-1] >= 0)
                assert torch.allclose(integral_output.t[1:,0,:], integral_output.t[:-1,-1,:])
                
                """
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
                    assert len(no_batch_integral.t_optimal) <= len(integral_output.t_optimal) 
                """
        break