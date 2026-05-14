from __future__ import annotations

import argparse
import copy

import torch
from torch import nn

from model import DenseNet
from problems import get_problem
from trainer import Trainer

experiments = {
    "exp_test_sol": {
        "problem": "exp_test_sol",
        "ode": {},
        "model": {"activation": nn.Tanh(), "layers": [1, 64, 64]},
        "trainer": {
            "integrator_config": {"method": "dopri5", "atol": 1e-6, "rtol": 1e-5},
            "loss_fxn": "relative_MSE",
            "curr_type": "exponential",
            "curr_config": {"metric": "loss", "cutoff": 1e-2, "scale": 0.05},
            "t_pred": 0.1,
            "t_max": 10,
            "N_epochs": 100000,
            #'t_init_lr' : 1e-10,
            "lr": 1e-6,
        },
        "dtype": torch.float64,
    },
    "exp_test": {
        "problem": "exp_test",
        "ode": {},
        "model": {"activation": nn.GELU(), "layers": [1, 64, 64]},
        "trainer": {
            "integrator_config": {"method": "dopri5", "atol": 1e-7, "rtol": 1e-6},
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {"metric": "loss", "cutoff": 3e-4, "scale": 0.05},
            "t_pred": 0.1,
            "t_max": 10,
            "N_epochs": 100000,
            #'t_init_lr' : 1e-10,
            "lr": 1e-3,
        },
        "dtype": torch.float64,
    },
    "x_squared_sol": {
        "problem": "x_squared",
        "ode": {},
        "model": {"activation": nn.GELU(), "layers": [1, 64, 64]},
        "trainer": {
            "integrator_config": {"method": "dopri5", "atol": 1e-6, "rtol": 1e-5},
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {"metric": "loss", "cutoff": 1e-3, "scale": 0.05},
            "t_pred": 0.1,
            "t_max": 10,
            "N_epochs": 1000000,
            #'t_init_lr' : 1e-10,
            "lr": 1e-4,
        },
        "dtype": torch.float64,
    },
    "linear": {
        "problem": "linear",
        "ode": {},
        "model": {"activation": nn.GELU(), "layers": [1, 64, 64]},
        "trainer": {
            "integrator_config": {"method": "dopri5", "atol": 1e-7, "rtol": 1e-6},
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {"metric": "loss", "cutoff": 1e-3, "scale": 0.05},
            "t_pred": 0.1,
            "t_max": 25,
            "N_epochs": 100000,
            #'t_init_lr' : 1e-10,
            "lr": 1e-3,
        },
        "dtype": torch.float64,
    },
    "quadratic": {
        "problem": "quadratic",
        "ode": {},
        "model": {"activation": nn.GELU(), "layers": [1, 64, 64]},
        "trainer": {
            "integrator_config": {"method": "dopri5", "atol": 1e-7, "rtol": 1e-6},
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {"metric": "loss", "cutoff": 1e-3, "scale": 0.05},
            "t_pred": 0.1,
            "t_max": 25,
            "N_epochs": 100000,
            #'t_init_lr' : 1e-10,
            "lr": 1e-3,
        },
        "dtype": torch.float64,
    },
    "lotka_volterra": {
        "problem": "lotka_volterra",
        "ode": {},
        "model": {"layers": [1, 64, 128, 256, 256, 128, 64]},
        "trainer": {
            "integrator_config": {
                "method": "dopri5",
                'atol' : 1e-7,
                'rtol' : 1e-6
                #"atol": 1e-10,
                #"rtol": 1e-9,
            },
            #'loss_fxn' : 'MSE_MAE',
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {
                "metric": "loss",
                "scale": 0.02,
                "cutoff": 1e-4,
                #'cutoff_patience': 1000,
                #'cutoff_scale': 1.2
            },
            "t_pred": 0.1,
            "t_max": 20,
            "N_epochs": 100000000,
            #'t_init_lr' : 1e-10,
            #'lr' : 1e-2, #5e-3
            "lr": 1e-2,
            "lr_patience": 1000,
            "lr_scale": 0.5,
        },
        "dtype": torch.float64,
    },
    "poisson": {
        "problem": "poisson",
        "ode": {"N_dims": 1, "force_type": "source_analytical"},
        "model": {"activation": nn.GELU(), "layers": [1, 64, 64]},
        "trainer": {
            "integrator_config": {"method": "dopri5", "atol": 1e-7, "rtol": 1e-6},
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {"metric": "loss", "cutoff": 1e-3, "scale": 0.05},
            "t_pred": 0.1,
            "t_max": 25,
            "N_epochs": 100000,
            #'t_init_lr' : 1e-10,
            "lr": 1e-3,
        },
        "dtype": torch.float64,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Run a PODE experiment.")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="lotka_volterra",
        choices=list(experiments.keys()),
        help="Name of the experiment to run (default: lotka_volterra)",
    )
    args = parser.parse_args()

    config = copy.deepcopy(experiments[args.experiment])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get ODE
    # ode_fxn = linear()
    # ode_fxn = quadratic()
    ode_fxn = get_problem(
        config["problem"], dtype=config["dtype"], **config["ode"], device=device
    )

    print("ODE", ode_fxn)
    # Get Model
    model = DenseNet(
        t_init=ode_fxn.t_init,
        initial_condition=ode_fxn.initial_condition,
        N_output_dims=ode_fxn.N_dims,
        **config["model"],
        dtype=config["dtype"],
        device=device,
    )

    # Get Integrator

    # integral_output = parallel_integrator.integrate(
    #    ode, mesh_init=t_init, mesh_final=t_final
    # )

    # Setup Trainer
    trainer = Trainer(
        model=model, **config["trainer"], dtype=config["dtype"], device=device
    )
    """
        integrator_config={
            'method' : 'adaptive_heun',
            'atol' : 1e-5,
            'rtol' : 1e-4
        },
        curr_type=None,#'exponential',
        curr_config={'metric' : 'loss', 'cutoff' : 1e-4, 'scale' : 0.05},
        t_pred=10,
        t_max=100,
        N_epochs=100000,
        #t_init_lr=1e-10,
        lr=1e-5#1e-5
    )
    """

    # Solve ODE
    trainer.train(config, ode_fxn, experiment_name=args.experiment)

    # Evaluate Solver


if __name__ == "__main__":
    main()
