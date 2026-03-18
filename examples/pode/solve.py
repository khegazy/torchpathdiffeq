from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override
import hashlib
import json
import os
from glob import glob

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.integrate import solve_bvp
from torch import nn
from torchdiffeq import odeint

import torchpathdiffeq as tpd

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
                #'atol' : 1e-7,
                #'rtol' : 1e-6
                "atol": 1e-10,
                "rtol": 1e-9,
            },
            #'loss_fxn' : 'MSE_MAE',
            "loss_fxn": "MSE",
            "curr_type": "exponential",
            "curr_config": {
                "metric": "loss",
                "scale": 0.05,
                "cutoff": 1e-4,
                #'cutoff_patience': 1000,
                #'cutoff_scale': 1.2
            },
            "t_pred": 0.1,
            "t_max": 20,
            "N_epochs": 100000000,
            #'t_init_lr' : 1e-10,
            #'lr' : 1e-2, #5e-3
            "lr": 1e-6,
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


class BaseODE(abc.ABC):
    def __init__(self, N_dims, device, dtype=torch.float64):
        self.N_dims = N_dims
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def ode(self, t, y):
        """ The ODE function (integrand) to be integrated """

    def solve_ode(self, t_max):
        self.sol_times = torch.linspace(self.t_init, t_max, 1000)
        self.solution = odeint(self.ode, self.initial_condition[0], self.sol_times)
        self.sol_times = torch.unsqueeze(self.sol_times, dim=-1)

    def first_derivative(self, model, t):
        return torch.autograd.functional.jacobian(
            lambda t: torch.sum(model(t), axis=0),
            t,
            create_graph=True,
            vectorize=True,
        ).transpose(0, 1)[:, :, 0]

    def second_derivative(self, model, t):
        return torch.autograd.functional.jacobian(
            self.first_derivative(model, t),
            t,
            create_graph=True,
            vectorize=True,
        ).transpose(0, 1)[:, :, 0]


class linear(BaseODE):
    __name__ = "linear"

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)
        self.initial_condition = torch.tensor(
            [0], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.t_init = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @override
    def ode(self, t, y):
        return t

    def train_eval(self, model, t):
        return self.first_derivative(model, t), self.ode(t, model(t))


class quadratic(BaseODE):
    __name__ = "quadratic"

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)
        self.initial_condition = torch.tensor(
            [0], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.t_init = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @override
    def ode(self, t, y):
        return t**2

    def train_eval(self, model, t):
        return self.first_derivative(model, t), self.ode(t, model(t))


class exp_test(BaseODE):
    __name__ = "exp_test"

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)
        self.initial_condition = torch.tensor(
            [4], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.t_init = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @override
    def ode(self, t, y):
        return torch.exp(-2 * t) - 3 * y

    def train_eval(self, model, t):
        return self.first_derivative(model, t), self.ode(t, model(t))


class exp_test_sol(BaseODE):
    __name__ = "exp_test_sol"

    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)
        self.initial_condition = torch.tensor(
            [4], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.t_init = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @override
    def ode(self, t, y=None):
        return torch.exp(-2 * t) + 3 * torch.exp(-3 * t)


class LotkaVolterra(BaseODE):
    __name__ = "lotka_volterra"

    def __init__(self, alpha=1.0, beta=1.0, delta=1.0, gamma=1.0, **kwargs):
        super().__init__(2, **kwargs)
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        self.t_init = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.initial_condition = torch.tensor(
            [1, 2], dtype=self.dtype, device=self.device
        ).unsqueeze(0)  # DEBUG
        print("DO NOT KNOW INIT CONDITIONS, FIX!!!!!!!!!!!!!!!!!!!!!!")

    @override
    def ode(self, t, state_vec):
        if len(state_vec.shape) > 1:
            x, y = state_vec[:, 0], state_vec[:, 1]
        else:
            x, y = state_vec
        dx_dt = self.alpha * x - self.beta * x * y
        dy_dt = self.delta * x * y - self.gamma * y
        return torch.stack([dx_dt, dy_dt], dim=-1)

    def train_eval(self, model, t):
        return self.first_derivative(model, t), self.ode(t, model(t))


class Poisson(BaseODE):
    __name__ = "poisson"

    def __init__(self, N_dims, force_type, dtype=torch.float64):
        super().__init__(N_dims, dtype=dtype)
        self.initial_condition = torch.tensor(
            [0], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.t_init = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self.bc = [0.0, 0.0]

        self.ode = getattr(self, f"_{force_type}")
        self.ode_np = getattr(self, f"_{force_type}_np")

    def _source_analytical(self, t, y):
        return -torch.sin(torch.pi * t)

    def _source_analytical_np(self, t, y):
        return -np.sin(np.pi * t)

    def _source_numerical(self, t, y):
        return torch.tanh(5 * (t - 0.5)) - torch.cos(10 * t)

    def _source_numerical_np(self, t, y):
        return np.tanh(5 * (t - 0.5)) - np.cos(10 * t)

    def solve_ode(self, t_max):
        def ode_system(x, y):
            return np.vstack((y[1], self.ode_np(x, None)))

        def bc(ya, yb):
            return np.array([ya[0] - self.bc[0], yb[0] - self.bc[-1]])

        self.sol_times = np.linspace(self.t_init, t_max, 100)
        y_guess = np.zeros((2, 100))
        self.solution = solve_bvp(ode_system, bc, self.sol_times, y_guess)
        self.sol_times = np.expand_dims(self.solution.x, -1)
        self.solution = np.expand_dims(self.solution.y[0], -1)
        self.sol_times = torch.tensor(self.sol_times)
        self.solution = torch.tensor(self.solution)

    def train_eval(self, model, t):
        return self.second_derivative(model, t), self.ode(t, model(t))


def get_problem(name, dtype=torch.float64, **kwargs):
    if "linear" in name:
        return linear(dtype=dtype, **kwargs)
    elif "quadratic" in name:
        return quadratic(dtype=dtype, **kwargs)
    elif name == "exp_test":
        return exp_test(dtype=dtype, **kwargs)
    elif name == "exp_test_sol":
        return exp_test_sol(dtype=dtype, **kwargs)
    elif name == "lotka_volterra":
        return LotkaVolterra(dtype=dtype, **kwargs)
    elif name == "poisson":
        return Poisson(dtype=dtype, **kwargs)
    else:
        raise ValueError(f"Cannot get find problem {name}")


class CurriculumClass:
    def __init__(
        self, curr_type, t_pred, t_max, config, dtype=torch.float64, N_epochs=None
    ):
        self.curr_type = curr_type
        self.config = config
        self.N_epochs = N_epochs
        self.previous_loss = 0
        self.ema_loss = 0
        self.N_history = 10
        self.loss_history = torch.zeros(self.N_history)
        self._idx = 0
        self.dtype = dtype

        self.prev_t_pred = 1e-9
        self.t_pred = torch.tensor(t_pred, dtype=self.dtype)
        self.t_max = torch.tensor(t_max, dtype=self.dtype)
        if self.curr_type is not None:
            self._is_curriculum_available(self.curr_type)
            self._update_curriculum = getattr(self, self.curr_type)
        else:
            self._update_curriculum = self._pass

        self.curr_patience = None
        if "cutoff_patience" in self.config:
            assert "cutoff" in self.config
            if "cutoff_scale" not in self.config:
                self.config["cutoff_scale"] = 1.2
            assert self.config["cutoff_scale"] > 1.0
            self.curr_patience = 0

            self.config["default_cutoff"] = self.config["cutoff"]

        """
        if curriculum_config is not None:
            self.curr_type = curriculum_config['type']
            self.curriculum_config = curriculum_config
            if N_epochs is not None and 'N_epochs' not in curriculum_config:
                self.curriculum_config['N_epochs'] = N_epochs
            self.curriculum_state = {}

            self._is_curriculum_available(self.curr_type)
            self._update_curriculum = getattr(self, self.curr_type)
        else:
            self.curr_type = None
            self.update_curriculum = self._pass
        """

    def _is_curriculum_available(self, curr_type):
        if curr_type not in dir(self):
            curr_types = [
                attr
                for attr in dir(CurriculumClass)
                if attr[0] != "_" and callable(getattr(self, attr))
            ]
            raise ValueError(
                f"Cannot evaluate {curr_type}, either add a new function to the Metrics class or use one of the following:\n\t{curr_types}"
            )

    # def _calculate_filter(self):

    def update_curriculum(self, epoch, loss):
        self.loss = loss

        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss

        self.loss_history[self._idx] = loss
        self._idx = (self._idx + 1) % self.N_history
        self.loss_std_ratio = torch.std(self.loss_history) / torch.abs(
            torch.mean(self.loss_history)
        )

        if self.t_pred != self.t_max:
            # Increase the cutoff is training stagnates and t_max has not been reached
            if self.curr_patience is not None:
                if self.loss > self.config["cutoff"]:
                    self.curr_patience += 1
                    if self.curr_patience >= self.config["cutoff_patience"]:
                        self.config["cutoff"] = (
                            self.config["cutoff"] * self.config["cutoff_scale"]
                        )
                        print(
                            f"Hit curriculum patience limit, increasing cutoff to {self.config['cutoff']}"
                        )
                        self.curr_patience = 0
                else:
                    # if self.curr_patience < self.config['cutoff_patience']/2:
                    #    self.config['cutoff'] = self.config['cutoff']/self.config['cutoff_scale']
                    self.config["cutoff"] = self.config["default_cutoff"]
                    self.curr_patience = 0
            t_update, updated_t = self._update_curriculum(epoch)
            t_update = np.minimum(t_update, self.t_max)
            self.prev_t_pred = self.t_pred
            self.t_pred = t_update
            return updated_t and t_update < self.t_max
        return None

    def _pass(self, *args, **kwargs):
        pass

    def load_curriculum(self, curriculum_state):
        self.curriculum_state = curriculum_state

    def initialize_curriculum(self, epoch, model, train_loader, eval_loaders):
        if self.curr_type is not None:
            getattr(self, f"_{self.curr_type}_init")(
                epoch, model, train_loader, eval_loaders
            )

    def exponential(self, epoch):
        if self.config["metric"] == "loss":
            update = self.loss < self.config["cutoff"]
        elif self.config["metric"] == "loss_std_ratio":
            update = self.loss_std_ratio < self.config["cutoff"]
        else:
            raise ValueError(f"Cannot handle metric type {self.config['metric']}")

        if update:
            return self.t_pred + self.config["scale"], True
            # return self.t_pred*(1 + self.config['scale']), True #TODO THIS IS EXPONENTIAL
        return self.t_pred, False

    def __exponential(self, epoch, **kwargs):
        # Smooth exponential progression
        progress = epoch / self.curriculum_config["N_epochs"]
        exp_progress = 1 - np.exp(-3 * progress)  # Asymptotic to 1
        length = (
            self.curriculum_config["init_length"]
            + (
                self.curriculum_config["final_length"]
                - self.curriculum_config["init_length"]
            )
            * exp_progress
        )
        t_pred = max(
            self.curriculum_config["init_length"],
            min(self.curriculum_config["final_length"], int(round(length))),
        )

        # Check if new prediction length will fit into gpu memory
        pass_memory, max_t_pred = self._memory_check(
            model.device, mem_unit, t_pred, mem_scale
        )
        if t_pred != self.t_pred and pass_memory:
            self.t_pred = t_pred
            train_loader.dataset.set_t_pred(t_pred)
            # CHANGING EVAL SET
            for label in eval_loaders:
                eval_loaders[label].dataset.set_t_pred(t_pred)
            self.curriculum_state["t_pred"] = t_pred
            return True
        return False

    def _exponential_init(self, epoch, model, train_loader, eval_loaders):
        if len(self.curriculum_state) == 0:
            self.curriculum_state["last_change"] = 1
            self.exponential(epoch, model, train_loader, eval_loaders)
        else:
            self.t_pred = self.curriculum_state["t_pred"]
            train_loader.dataset.set_t_pred(self.t_pred)
            # CHANGING EVAL SET
            for label in eval_loaders:
                eval_loaders[label].dataset.set_t_pred(self.t_pred)


class DenseNet(nn.Module):
    def __init__(
        self,
        t_init,
        initial_condition,
        layers,
        N_output_dims,
        device,
        dtype=torch.float64,
        activation=nn.GELU(),
        output_activation=None,
        normalize=False,
    ):
        super().__init__()

        self.initial_condition = initial_condition
        self.t_init = t_init.to(device)

        self.n_layers = len(layers)
        assert self.n_layers >= 1
        layers.append(N_output_dims)
        self.device = device
        self.dtype = dtype
        self.activation = activation
        # self.activation = nn.GELU()
        # self.activation = nn.Tanh()
        # self.activation = nn.ELU()

        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                nn.Linear(
                    layers[i],
                    layers[i + 1],
                    dtype=dtype,
                    bias=False,
                    device=self.device,
                )
            )
            if i != self.n_layers - 1:
                if normalize:
                    raise NotImplementedError
                    self.layers.append(nn.BatchNorm1d(layers[i + 1], dtype=dtype))
                self.layers.append(self.activation)

        if output_activation is not None:
            self.layers.append(nn.GELU())

    def forward(self, x):
        x = x - self.t_init
        x = torch.movedim(x, 1, -1)
        for l in self.layers:
            x = l(x)
        x = torch.movedim(x, -1, 1)
        return x + self.initial_condition


class Trainer(CurriculumClass):
    def __init__(
        self,
        model,
        integrator_config,
        t_max,
        loss_fxn="MSE",
        lr=1e-3,
        N_epochs=None,
        t_pred=0.1,
        t_init=0.0,
        curr_type=None,
        curr_config=None,
        lr_patience=None,
        lr_scale=None,
        dtype=torch.float64,
        device="cuda",
    ) -> None:
        super().__init__(
            curr_type=curr_type,
            t_pred=t_pred,
            t_max=t_max,
            config=curr_config,
            dtype=dtype,
        )
        self.model = model
        self.loss_fxn_name = loss_fxn
        self.plot_colors = ["k", "b", "r", "g"]
        self.device = device

        assert N_epochs is not None or t_max is not None
        self.N_epochs = np.inf if N_epochs is None else N_epochs

        lr_check = lr_patience is not None and lr_scale is not None
        lr_check = lr_check or (lr_patience is None and lr_scale is None)
        assert lr_check, "Must specify both 'lr_scal' and 'lr_patience'"
        self.lr_patience = lr_patience
        self.lr_scale = lr_scale

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.init_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(), tolerance_change=0.0, line_search_fn="strong_wolfe"
        )

        assert hasattr(self, f"_{loss_fxn}")
        self.loss_fxn = getattr(self, f"_{loss_fxn}")

        self.integrator = tpd.RKParallelUniformAdaptiveStepsizeSolver(
            **integrator_config, device=self.device
        )

    def _find_saved_weights(self):
        files = glob(os.path.join(self.chkpt_dir, "*pth"))
        if len(files) == 0:
            return None, None
        max_t = -1.0
        filename = None
        for f in files:
            idx = f.find("_t-") + 3
            print(f, f[idx:-4])
            t_str = f[idx:-4]
            t = float(t_str)
            if t > max_t:
                max_t = t
                filename = f
                t_label = t_str
        return filename, t_label

    def _save_model(self, t):
        if self.prev_saved_model is not None:
            os.remove(self.prev_saved_model)
        filename = os.path.join(self.chkpt_dir, self._get_chkpt_filename())
        torch.save(self.model.state_dict(), filename)
        self.prev_saved_model = filename

    def _get_results_filename(self):
        t_label = np.round(np.squeeze(self.t_pred)).item()
        return f"training_t-{t_label:.3f}.h5"

    def _get_chkpt_filename(self):
        t_label = np.round(np.squeeze(self.t_pred)).item()
        return f"model_weights_t-{t_label:.3f}.pth"

    def _load_model(self):
        filename, t_label = self._find_saved_weights()
        print("FILENAME", filename)
        if filename is None:
            print("Start training from scratch")
            return 0
        else:
            print(f"Starting training from {filename}")
            self.model.load_state_dict(torch.load(filename))
            results_filename = self._get_results_filename()
            with h5.File(os.path.join(self.results_dir, results_filename), "r") as f:
                self.eval_integral_values = list(f["integrals"])
                self.eval_integral_limits = list(f["t_final"])
                self.t_pred = f["t_pred"][...]
                idx = -1
                while self.eval_integral_limits[idx][0] >= self.t_pred.item():
                    idx -= 1
                self.prev_t_pred = self.eval_integral_limits[idx][0]
                return f["epoch"][...]

    def plot_results(self, time, pred, solution, epoch):
        device = pred.device
        pred = pred.cpu()
        solution = solution.cpu()
        time = time.cpu()
        plt.figure(figsize=(10, 7))
        colors = ["red", "green", "purple", "orange"]
        plt.plot(
            time[:, 0].detach().numpy(),
            pred[:, 0].detach().numpy(),
            label="Predicted x",
            color=colors[0],
            lw=2.5,
        )
        plt.plot(
            time[:, 0].detach().numpy(),
            pred[:, 1].detach().numpy(),
            label="Predicted y",
            color=colors[1],
            lw=2.5,
        )
        plt.plot(
            time[:, 0].detach().numpy(),
            solution[:, 0].detach().numpy(),
            label="Ground Truth x",
            color=colors[2],
            linestyle="--",
            lw=2,
            alpha=0.8,
        )
        plt.plot(
            time[:, 0].detach().numpy(),
            solution[:, 1].detach().numpy(),
            label="Ground Truth y",
            color=colors[3],
            linestyle="--",
            lw=2,
            alpha=0.8,
        )
        # plt.title('Lotka-Volterra learned ODE', fontsize=16)
        (
            plt.ylabel("Prey/Predator Population (x/y)", fontsize=16),
            plt.xlabel("Time (t)", fontsize=16),
        )
        plt.legend(fontsize=13), plt.grid(True)  # plt.axis('equal')
        plt.xticks(fontsize=13)  # Sets x-axis tick label font size
        plt.yticks(fontsize=13)  # Sets y-axis tick label font size
        plt.xlim(0, 20)
        plt.ylim(0, 3)

        plt.savefig(os.path.join(self.plot_dir, f"ppr_training_{epoch}.png"))
        pred.to(device)
        solution.to(device)

    def _plot_results(self, time, pred, solution, epoch):
        N_vars = pred.shape[-1]
        fig, axes = plt.subplots(2, 1)
        y_max, y_min = -np.inf, np.inf
        for i in range(N_vars):
            if torch.amax(solution[:, i]) > y_max:
                y_max = torch.amax(solution[:, i])
            if torch.amax(solution[:, i]) < y_min:
                y_min = torch.amin(solution[:, i])
            for j in range(2):
                axes[j].plot(
                    time[:, 0], solution[:, i], color=self.plot_colors[i], alpha=0.5
                )
                axes[j].plot(
                    time[:, 0],
                    pred[:, i].detach().numpy(),
                    color=self.plot_colors[i],
                    linestyle=":",
                )
        y_delta = 0.05 * np.abs(y_max - y_min)
        if y_delta > 0.1 * np.abs(y_max):
            y_max += y_delta
        else:
            y_max += 0.1 * np.abs(y_max)

        if y_delta > 0.1 * np.abs(y_min):
            y_min -= y_delta
        else:
            y_min += 0.1 * np.abs(y_min)

        for i in range(2):
            axes[i].set_xlim(time[0], time[-1])

        axes[0].set_ylim(y_min, y_max)
        axes[1].set_ylim(np.maximum(y_min, 1e-9), y_max)
        axes[1].set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, f"training_{epoch}.png"))

    def eval_results(self, t_init, epoch_count):
        t_eval_max = self.t_pred if self.t_max == np.inf else self.t_max
        t_eval = torch.linspace(
            t_init, t_eval_max, 1000, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        eval_output = odeint(
            self.ode_fxn.ode, self.ode_fxn.initial_condition[0], t_eval[:, 0]
        )
        self.plot_results(t_eval, self.model(t_eval), eval_output, epoch_count)

    @staticmethod
    def _MSE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        return torch.sum((pred - target) ** 2, dim=1, keepdim=True)

    @staticmethod
    def _MAE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        return torch.sum(torch.abs(pred - target), dim=1, keepdim=True)

    @staticmethod
    def _relative_MSE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        # print("diff", (pred - target))
        # print("pred", pred)
        # print("targ", target)
        return torch.sum(
            ((pred - target) / (torch.abs(target) + 1e-2))
            ** 2,  # TODO: make eps smaller
            dim=1,
            keepdim=True,
        )

    def _MSE_combined(self, pred, target):
        return self._MSE(pred, target) + self._relative_MSE(pred, target)

    def _MSE_MAE(self, pred, target):
        return self._MSE(pred, target) + self._MAE(pred, target)

    def solution_integrad(self, t, model, verbose=False):
        pred = model(t)

        # TODO: use gaussian instead and t_init
        # scale = torch.exp(-1*t)
        # scale /= 1.0 - torch.exp(-1*torch.tensor(self.t_pred))
        if verbose:
            print("pred", pred)
            print("targ", self.ode_fxn(t, pred))
            # print("scale", scale)
        return self.loss_fxn(pred, self.ode_fxn(t, pred))

    def _integrad(self, t, model, verbose=False):
        pred, target = ode_fxn.train_eval(model, t)

        # TODO: use gaussian instead and t_init
        # scale = torch.exp(-1*t)
        # scale /= 1.0 - torch.exp(-1*torch.tensor(self.t_pred))
        """
        if verbose:
            print("pred", jac)
            print("targ", self.ode_fxn(t, model(t)))
            #print("scale", scale)
        """
        return self.loss_fxn(pred, target)

    def train_iteration(self):
        self.optimizer.zero_grad()
        self.integral_output = self.integrator.integrate(
            ode_fxn=self._integrad, t=self.input_times, ode_args=(self.model,)
        )
        # print(epoch_count, integral_output.t.shape)
        loss = self.integral_output.loss
        # print("BEFORE", self.model.layers[0].weight.data[:5])
        # print("OUTPUT", integral_output)

        # T0 loss
        init_loss = torch.mean(
            self.loss_fxn(self.model(self.t_init_eval), self.ode_fxn.initial_condition)
        )
        loss = loss + init_loss
        if not self.integral_output.gradient_taken:
            # print("taking gradients")
            loss.backward()
        else:
            init_loss.backward()
        return loss

    def train(self, config, ode_fxn):
        self.ode_fxn = ode_fxn
        self.ode_fxn.solve_ode(self.t_max)
        t_init = self.ode_fxn.t_init
        t_init_eval = torch.tensor(
            [t_init], requires_grad=True, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self.prev_t_pred = 1e-9

        config_vars = json.dumps(config, sort_keys=True).encode()
        hash_label = hashlib.blake2s(config_vars, digest_size=4).hexdigest()
        sub_dir = os.path.join(self.ode_fxn.__name__, self.loss_fxn_name, hash_label)
        self.chkpt_dir = os.path.join("checkpoints", sub_dir)
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.results_dir = os.path.join("results", sub_dir)
        self.plot_dir = os.path.join(self.results_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self.eval_integral_values, self.eval_integral_limits = [], []

        with open(os.path.join(self.results_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        self.prev_saved_model = None
        epoch_count = self._load_model()

        self.model.train()
        # self.model.compile()

        print("Optimizer Parameters:")
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"\nParameter Group {i}:")
            for key, value in param_group.items():
                if key == "params":
                    # 'params' contains a list of actual parameter tensors
                    print(f"  {key}: {len(value)} tensors")
                    for j, param in enumerate(value):
                        print(
                            f"    Tensor {j} - Shape: {param.shape}, Requires Grad: {param.requires_grad}"
                        )
                        # You can also print the tensor values if desired (be mindful of large tensors)
                        # print(f"      Value: {param.data}")
                else:
                    print(f"  {key}: {value}")
        # Train t0
        loss_ratio, error_ratio, t0_count = 1, torch.tensor([1e10]), 0
        prev_loss, prev_filename = 1.0, None
        pred = torch.ones(1)
        self.t_init_eval = t_init_eval
        print("Training initial conditions")
        while torch.any(error_ratio > 1e-3):
            if t0_count % 100 == 0:
                print(
                    f"T0 count {t0_count} | loss ratio: {loss_ratio} / error ratio: {error_ratio} / prediction: {torch.squeeze(pred).item()}"
                )
            self.init_optimizer.zero_grad()
            pred = self.model(t_init_eval)
            loss = torch.mean(self.loss_fxn(pred, self.ode_fxn.initial_condition))
            # print("LOSS 0", loss.shape, loss)
            # print(self.model(t_init), self.ode_fxn(t_init))
            loss.backward()
            # print("BEFORE", self.model.layers[0].bias.data[:5])
            # print("GRAD", self.model.layers[0].bias.grad[:5]*1e5)
            self.init_optimizer.step()
            loss_ratio = torch.abs(prev_loss - loss) / prev_loss
            error_ratio = torch.abs(pred - self.ode_fxn.initial_condition) / (
                torch.abs(self.ode_fxn.initial_condition) + 1e-9
            )
            error_ratio = torch.squeeze(error_ratio)
            print(
                "WTF",
                self.ode_fxn.initial_condition.shape,
                pred.shape,
                error_ratio.shape,
            )
            prev_loss = loss
            # print("AFTER", self.model.layers[0].bias.data[:5])
            # print(t0_count, self.t_pred, loss, loss_ratio)
            t0_count += 1
        print("Finished training initial conditions", loss_ratio, error_ratio)

        # Make output folder
        prev_loss = 0
        N_const_loss = 0
        cutoff_count = 0
        self.input_times, epoch_count = t_init_eval, 0
        train_criteria = True
        self.integral_output = None
        while train_criteria:
            eval_time = 5
            if epoch_count % eval_time == 0:
                # print("INIT", t_init, torch.squeeze(self.model(t_init_eval)).detach().numpy())
                if self.integral_output is not None:
                    print(
                        f"Epoch/Time {epoch_count}/{self.t_pred}: {loss.item()} | {self.config['cutoff']} | {len(self.integral_output.t)}"
                    )
                    # self.loss_integrad(torch.arange(5, dtype=self.dtype).unsqueeze(-1)*5./4., self.model, verbose=True)
                    # if integral_output is not None:
                    #    print(integral_output.t[:,0,0])
                    integral_values = [loss.item()]
                    integral_limits = [self.t_pred]
                    eval_t_preds = np.concatenate(
                        [
                            np.array([self.prev_t_pred]),
                            self.t_pred * np.array([0.95, 0.9, 0.75, 0.5, 0.25]),
                        ]
                    )
                    # print("EVAL", self.t_pred, eval)
                    half_idx = -2
                    for eval_t in eval_t_preds:
                        self.eval_input_times = torch.tensor(
                            [t_init, eval_t], dtype=self.dtype, device=self.device
                        ).unsqueeze(-1)
                        with torch.no_grad():
                            eval_integral_output = self.integrator.integrate(
                                ode_fxn=self._integrad,
                                t=self.eval_input_times,
                                ode_args=(self.model,),
                            )
                        integral_values.append(eval_integral_output.loss.item())
                        integral_limits.append(eval_t)
                    self.eval_integral_values.append(np.array(integral_values))
                    data_values = np.array(self.eval_integral_values)
                    self.eval_integral_limits.append(np.array(integral_limits))
                    data_limits = np.array(self.eval_integral_limits)
                    filename = os.path.join(
                        self.results_dir, self._get_results_filename()
                    )
                    with h5.File(filename, "w") as f:
                        f.create_dataset("epoch", data=epoch_count)
                        f.create_dataset("t_pred", data=self.t_pred)
                        f.create_dataset("t_final", data=data_limits)
                        f.create_dataset("integrals", data=data_values)
                    # if prev_filename is not None:
                    #     print("preve filename", prev_filename)
                    #     os.remove(prev_filename)
                    # prev_filename = filename

                    fig, ax = plt.subplots(2, 1, height_ratios=[5, 1])
                    ax[0].plot(data_values[:, -2], "y--")
                    ax[0].plot(data_values[:, -3], "g--")
                    ax[0].plot(data_values[:, -4], "r--")
                    ax[0].plot(data_values[:, 1], "b--")
                    ax[0].plot(data_values[:, 0], "k-")
                    ax[1].plot(data_limits[:, 0])
                    fig.tight_layout()
                    fig.savefig(os.path.join(self.results_dir, "training_progress.png"))

                    if integral_values[half_idx] / integral_values[0] > 0.25:
                        cutoff_count = cutoff_count + 1
                        if cutoff_count > 50:
                            self.config["cutoff"] = 1.25 * self.config["cutoff"]
                            cutoff_count = 0
                    else:
                        cutoff_count = 0

                if epoch_count % 50 == 0:
                    self._integrad(
                        torch.arange(
                            10, dtype=self.dtype, device=self.device
                        ).unsqueeze(-1)
                        * self.t_pred
                        / 9.0,
                        self.model,
                        verbose=True,
                    )
                    if self.integral_output is not None:
                        print("Y", self.integral_output.y[0, :, 0])
                    self.eval_results(t_init, epoch_count)
                    self._save_model(self.t_pred)

            updated_curr = self.update_curriculum(epoch_count, loss)
            if updated_curr:
                self.optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    tolerance_change=0.0,
                    line_search_fn="strong_wolfe",
                )

            if updated_curr or self.integral_output is None:
                if self.input_times[-1] < self.t_pred:
                    self.input_times = torch.concatenate(
                        [
                            self.input_times,
                            torch.tensor([self.t_pred], device=self.device).unsqueeze(
                                -1
                            ),
                        ],
                        dim=0,
                    )
                self.input_times = torch.tensor(
                    [t_init, self.t_pred], dtype=self.dtype, device=self.device
                ).unsqueeze(-1)
            else:
                self.input_times = self.integral_output.t_optimal
            # self.input_times = torch.tensor([t_init, self.t_pred], dtype=self.dtype, device=self.device).unsqueeze(-1)
            # print("TIMES", times.shape, loss, torch.std(self.loss_history), self.loss_history, times)
            """
            integral_output = self.integrator.integrate(
                ode_fxn=self._integrad, t=self.input_times, ode_args=(self.model,)
            )
            #print(epoch_count, integral_output.t.shape)
            loss = integral_output.loss
            #print("BEFORE", self.model.layers[0].weight.data[:5])
            #print("OUTPUT", integral_output)
            if not integral_output.gradient_taken:
                #print("taking gradients")
                loss.backward()
            """
            """

            t_eval = torch.arange(100).unsqueeze(1)*self.t_pred/99
            loss = torch.mean(
                self.loss_fxn(self.model(t_eval), self.ode_fxn(t_eval))
            )
            loss.backward()
            """

            # print("GRAD", self.model.layers[0].weight.grad[:5]*1e5)
            loss = self.optimizer.step(self.train_iteration)
            if loss == prev_loss:
                N_const_loss += 1
            else:
                prev_loss = loss
                N_const_loss = 0
            if N_const_loss > 200:
                print("NEW SOLVER")
                self.optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    tolerance_change=0.0,
                    line_search_fn="strong_wolfe",
                )
                # asdfasd
            # print("AFTER", self.model.layers[0].weight.data[:5])
            # times = integral_output.t_optimal.clone()
            # times = integral_output.t_optimal.detach()
            # times.requires_grad_(True)

            # # Update learning rate
            # if self.t_pred != self.t_max and self.lr_patience is not None:
            #     if loss < prev_loss or updated_curr:
            #         patience = 0
            #     else:
            #         patience += 1
            #         if patience >= self.lr_patience:
            #             self.optimizer.lr = self.optimizer.lr*self.lr_scale
            #             patience = 0
            #     prev_loss = loss.detach().item()

            epoch_count += 1
            if self.N_epochs is not None:
                train_criteria = epoch_count < self.N_epochs
            else:
                train_criteria = self.input_times[-1, 0] < self.t_max

        self.eval_results(t_init, epoch_count)


if __name__ == "__main__":
    config = experiments["lotka_volterra"]
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
    #    ode, t_init=t_init, t_final=t_final
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
    del config["dtype"]
    trainer.train(config, ode_fxn)  # torch.zeros((1,1)))

    # Evaluate Solver
