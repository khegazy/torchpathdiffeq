from __future__ import annotations

import time

import torch

import torchpathdiffeq as tpdiffeq

############################
#####  Test Variables  #####
############################
n_runs = 100
device = "cuda"
method = "dopri5"
atol = 1e-9
rtol = 1e-7
t_init = torch.tensor([0], dtype=torch.float64, device=device)
t_final = torch.tensor([1], dtype=torch.float64, device=device)
y0 = torch.tensor([0], dtype=torch.float64, device=device)


########################
#####  Integrands  #####
########################

WS_min_init = torch.tensor([1.133, -1.486], dtype=torch.float64, device=device)
WS_min_final = torch.tensor([-1.166, 1.477], dtype=torch.float64, device=device)


def wolf_schlegel(t, y=None):
    assert torch.all(t) >= 0
    assert torch.all(t) <= 1
    while len(t.shape) < 2:
        t = t.unsqueeze(0)

    interpolate = WS_min_init + (WS_min_final - WS_min_init) * t
    x = interpolate[:, 0].unsqueeze(-1)
    y = interpolate[:, 1].unsqueeze(-1)

    return 10 * (x**4 + y**4 - 2 * x**2 - 4 * y**2 + x * y + 0.2 * x + 0.1 * y)


def damped_sine(
    t,
    y=None,
    w=torch.tensor([3.7], dtype=torch.float64, device=device),
    a=torch.tensor([5], dtype=torch.float64, device=device),
):
    return torch.exp(-a * t) * torch.sin(w * t * 2 * torch.pi)


integrands = [wolf_schlegel, damped_sine]

###################################
#####  torchdiffeq Speed Test #####
###################################
print("torchdiffeq")
tdiffeq_results = []
for ode_fxn in integrands:
    total_time = 0
    integrator = tpdiffeq.SerialAdaptiveStepsizeSolver(
        ode_fxn=ode_fxn,
        method=method,
        atol=atol,
        rtol=rtol,
        t_init=t_init,
        t_final=t_final,
        y0=y0,
        device=device,
    )
    for _ in range(n_runs):
        t0 = time.time()
        _ = integrator.integrate()
        total_time = total_time + (time.time() - t0)
    tdiffeq_results.append(total_time / n_runs)


#######################################
#####  torchpathdiffeq Speed Test #####
#######################################

print("torchpathdiffeq api")
tpdiffeq_api_results = []
for ode_fxn in integrands:
    total_time = 0
    for _ in range(n_runs):
        t0 = time.time()
        _ = tpdiffeq.ode_path_integral(
            ode_fxn=ode_fxn,
            method=method,
            computation="parallel",
            sampling="uniform",
            atol=atol,
            rtol=rtol,
            t_init=t_init,
            t_final=t_final,
            y0=y0,
            device=device,
        )
        total_time = total_time + (time.time() - t0)
    tpdiffeq_api_results.append(total_time / n_runs)


print("torchpathdiffeq integrator")
tpdiffeq_int_results = []
for ode_fxn in integrands:
    total_time = 0
    integrator = tpdiffeq.get_parallel_RK_solver(
        sampling_type="uniform",
        ode_fxn=ode_fxn,
        method=method,
        atol=atol,
        rtol=rtol,
        t_init=t_init,
        t_final=t_final,
        y0=y0,
        device=device,
    )
    for _ in range(n_runs):
        t0 = time.time()
        _ = integrator.integrate()
        total_time = total_time + (time.time() - t0)
    tpdiffeq_int_results.append(total_time / n_runs)

# Print Results
message = "Problem \\ Method|    torchdiffeq    |    torchpathdiffeq API\t  ratio    |    torchpathdiffeq Integrator\t  ratio\n"
for idx, fxn in enumerate(integrands):
    td_out = tdiffeq_results[idx]
    tpd_api_out = tpdiffeq_api_results[idx]
    tpd_int_out = tpdiffeq_int_results[idx]
    ratio_api = td_out / tpd_api_out
    ratio_int = td_out / tpd_int_out
    message += f"{fxn.__name__}\t|      {td_out:.5f}      |"
    message += f"         {tpd_api_out:.5f}\t\t {ratio_api:.5f}   |"
    message += f"               {tpd_int_out:.5f}\t\t {ratio_int:.5f}\n"
print(message)
