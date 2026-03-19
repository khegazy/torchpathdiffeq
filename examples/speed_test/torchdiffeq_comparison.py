from __future__ import annotations

import logging
import time

import torch

import torchpathdiffeq as tpdiffeq
from torchpathdiffeq.examples import (
    damped_sine,
    exp,
    sine_squared,
    t_squared,
    wolf_schlegel,
)

# Set up logging to both screen and file
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("speed_test.log"),
    ],
)

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

integrands = [wolf_schlegel, damped_sine, sine_squared, exp, t_squared]

###################################
#####  torchdiffeq Speed Test #####
###################################
logger.info("torchdiffeq")
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

logger.info("torchpathdiffeq api")
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


logger.info("torchpathdiffeq integrator")
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

# Log Results
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
logger.info("\n%s", message)
