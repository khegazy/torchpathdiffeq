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

class wf():
    def __init__(self):
        self.calls = 0
    def __call__(self, t, y=None):
        assert torch.all(t) >= 0 and torch.all(t) <= 1
        while len(t.shape) < 2:
            t = t.unsqueeze(0)

        interpolate = WS_min_init + (WS_min_final - WS_min_init)*t
        x = interpolate[:,0].unsqueeze(-1)
        y = interpolate[:,1].unsqueeze(-1)

        self.calls = self.calls + 1
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
    print("STARTING")
    loop_items = zip(
        ['Uniform'],
        [UNIFORM_METHODS],
        [steps.ADAPTIVE_UNIFORM])
    for sampling_name, sampling, sampling_type in loop_items:
        for method in sampling.keys():
            parallel_integrator = get_parallel_RK_solver(
                sampling_type,
                method=method,
                atol=atol,
                rtol=rtol,
                remove_cut=0.1,
                ode_fxn=wolf_schlegel,
            )
            if method == 'generic3':
                serial_method = 'bosh3'
            else:
                serial_method = method
            wf_class = wf()
            serial_integrator = SerialAdaptiveStepsizeSolver(
                serial_method, atol, rtol, ode_fxn=wf_class#wolf_schlegel
            )

            parallel_integral = parallel_integrator.integrate()
            serial_integral = serial_integrator.integrate()
            print("SERIAL INTEGRAL STEPS", wf_class.calls)
            print("PARALLEL INTEGRAL STEPS", parallel_integral.t.shape, parallel_integral.t_pruned.shape)
            print("USING METHOD", method, serial_method)

            error = torch.abs(parallel_integral.integral - serial_integral.integral)
            error_tolerance = atol + rtol*torch.abs(serial_integral.integral)
            error_tolerance = error_tolerance*len(parallel_integral.t)
            assert error < error_tolerance, f"Failed with {sampling_name} ingegration method {method}"