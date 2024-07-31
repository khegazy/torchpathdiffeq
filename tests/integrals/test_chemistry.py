import torch

from torchpathdiffeq import RKParallelAdaptiveStepsizeSolver, SerialAdaptiveStepsizeSolver

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
    parallel_integrator = RKParallelAdaptiveStepsizeSolver(
        'euler', atol, rtol, remove_cut=0.1, ode_fxn=wolf_schlegel
    )
    serial_integrator = SerialAdaptiveStepsizeSolver(
        "adaptive_heun", atol, rtol, ode_fxn=wolf_schlegel
    )

    parallel_integral = parallel_integrator.integrate()
    serial_integral = serial_integrator.integrate()

    print("INTEGRALS", parallel_integral.integral, serial_integral.integral)
    error = torch.abs(parallel_integral.integral - serial_integral.integral)
    assert error/serial_integral.integral < 0.01


test_chemistry()