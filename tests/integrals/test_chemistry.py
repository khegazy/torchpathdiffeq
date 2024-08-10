import torch

from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    SerialAdaptiveStepsizeSolver,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS\

WS_min_init = torch.tensor([1.133, -1.486])
WS_min_final = torch.tensor([-1.166, 1.477])
def wolf_schlegel(t, y=None):
    assert torch.all(t) >= 0 and torch.all(t) <= 1
    while len(t.shape) < 2:
        t = t.unsqueeze(0)

    interpolate = WS_min_init + (WS_min_final - WS_min_init)*t
    x = interpolate[:,0].unsqueeze(-1)
    y = interpolate[:,1].unsqueeze(-1)

    return 10*(x**4 + y**4 - 2*x**2 - 4*y**2\
        + x*y + 0.2*x + 0.1*y)


def test_chemistry():
    atol = 1e-5
    rtol = 1e-5
    loop_items = zip(
        ['Uniform', 'Variable'],
        [UNIFORM_METHODS, VARIABLE_METHODS],
        [steps.ADAPTIVE_UNIFORM, steps.ADAPTIVE_VARIABLE]
    )
    for sampling_name, sampling, sampling_type in loop_items:
        for method in sampling.keys():
            print("TESTING METHOD", method)
            #parallel_integrator = RKParallelAdaptiveStepsizeSolver(
            parallel_integrator = get_parallel_RK_solver(
                sampling_type, method, atol, rtol, remove_cut=0.1, ode_fxn=wolf_schlegel
            )
            if method == 'generic3':
                serial_method = 'bosh3'
            else:
                serial_method = method
            serial_integrator = SerialAdaptiveStepsizeSolver(
                serial_method, atol, rtol, ode_fxn=wolf_schlegel
            )

            parallel_integral = parallel_integrator.integrate()
            serial_integral = serial_integrator.integrate()

            print("INTEGRALS", parallel_integral.integral, serial_integral.integral)
            error = torch.abs(parallel_integral.integral - serial_integral.integral)
            assert error/serial_integral.integral < atol/10, f"Failed with {sampling_name} ingegration method {method}"


test_chemistry()