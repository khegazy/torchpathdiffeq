from __future__ import annotations

from abc import ABC, abstractmethod
from typing_extensions import override

import numpy as np
import torch
from scipy.integrate import solve_bvp
from torchdiffeq import odeint


class BaseODE(ABC):
    def __init__(self, N_dims, device, dtype=torch.float64):
        self.N_dims = N_dims
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def ode(self, t, y):
        """The ODE function (integrand) to be integrated"""

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

    def _source_analytical(self, t, _y):
        return -torch.sin(torch.pi * t)

    def _source_analytical_np(self, t, _y):
        return -np.sin(np.pi * t)

    def _source_numerical(self, t, _y):
        return torch.tanh(5 * (t - 0.5)) - torch.cos(10 * t)

    def _source_numerical_np(self, t, _y):
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
