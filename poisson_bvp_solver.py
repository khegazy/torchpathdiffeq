import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp

from torchpathdiffeq import (
    steps,
    get_parallel_RK_solver
)

# Base configuration
BASE_CONFIG = {
    'domain_start': 0.0,
    'domain_end': 1.0,
    'boundary_start_val': 0.0,
    'boundary_end_val': 0.0,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'lr': 1e-4,
    'lambda_bc': 100,  # Weight for the boundary condition loss
    'path_net_hidden_dim': 64,

    'method': 'adaptive_heun',
    'sampling_type': steps.ADAPTIVE_UNIFORM,
    'atol': 1e-5,
    'rtol': 1e-3,
}

PROBLEM_CONFIGS = {
    'analytical': {
        'epochs': 8000,
    },
    'numerical': {
        'epochs': 10000,
    }
}

class BVP_pathnet(nn.Module):
    def __init__(self):
        super(BVP_pathnet, self).__init__()
        hidden_dim = BASE_CONFIG['path_net_hidden_dim']
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        return out

class PoissonResidualIntegrand(nn.Module):
    def __init__(self, path_net, problem_type='analytical'):
        super(PoissonResidualIntegrand, self).__init__()
        self.path_net = path_net

        # Two choices of source functions
        if problem_type == 'analytical':
            self.source_function = self._source_analytical
        elif problem_type == 'numerical':
            self.source_function = self._source_numerical
        else:
            raise ValueError(f"Unknown problem_type: {problem_type}")

    def _source_analytical(self, x):
        return -torch.sin(torch.pi * x)

    def _source_numerical(self, x):
        return torch.tanh(5 * (x - 0.5)) - torch.cos(10 * x)

    def compute_second_derivative(self, x):
        """Computes u''(x)"""
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        u = self.path_net(x)
        u_prime = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_double_prime = torch.autograd.grad(u_prime, x, torch.ones_like(u_prime), create_graph=True)[0]
        return u_double_prime

    def forward(self, t, y=None):
        """
        Integrand(x) = (u''(x) - f(x))^2
        """
        x = t
        u_double_prime_approx = self.compute_second_derivative(x)
        f_val = self.source_function(x)
        ode_residual = u_double_prime_approx - f_val
        return ode_residual ** 2

# --- Training and Plotting Functions ---

def train_bvp_solver(path_net, integrator, ode_integrand_func, config):
    print(f"--- Starting Training for '{config['problem_type']}' Problem ---")
    print(f"Epochs: {config['epochs']}, LR: {config['lr']}, Device: {config['device']}")

    optimizer = torch.optim.Adam(path_net.parameters(), lr=config['lr'])

    x_start = torch.tensor([config['domain_start']], device=config['device'], dtype=torch.float32)
    x_end = torch.tensor([config['domain_end']], device=config['device'], dtype=torch.float32)
    u_start_val = torch.tensor([[config['boundary_start_val']]], device=config['device'], dtype=torch.float32)
    u_end_val = torch.tensor([[config['boundary_end_val']]], device=config['device'], dtype=torch.float32)

    loss_history = []

    for epoch in range(config['epochs']):
        path_net.train()
        optimizer.zero_grad()

        # 1. Integrate the ODE residual loss across the domain
        integral_result = integrator.integrate(
            ode_integrand_func,
            t_init=x_start,
            t_final=x_end,
        )
        integral_ode_loss = integral_result.integral

        # 2. Compute the Boundary Condition Loss
        u_pred_start = path_net(x_start.view(-1, 1))
        u_pred_end = path_net(x_end.view(-1, 1))
        loss_bc = (u_pred_start - u_start_val)**2 + (u_pred_end - u_end_val)**2
        loss_bc = torch.squeeze(loss_bc)

        # 3. Combine the losses
        total_loss = integral_ode_loss + config['lambda_bc'] * loss_bc
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

        if epoch % 500 == 0 or epoch == config['epochs'] - 1:
            print(f"Epoch {epoch:5d}/{config['epochs']}: Total Loss = {total_loss.item():.6f}, "
                  f"Integral ODE Loss = {integral_ode_loss.item():.6f}, BC Loss = {loss_bc.item():.6f}")

    print("--- Training Completed ---")
    return loss_history

def plot_analytical_results(path_net, config):
    print("--- Plotting Results (vs. Analytical Solution) ---")

    def true_solution(x):
        return (1 / (np.pi**2)) * np.sin(np.pi * x)

    path_net.eval()
    with torch.no_grad():
        x_space = torch.linspace(config['domain_start'], config['domain_end'], 200).view(-1, 1).to(config['device'])
        u_predicted = path_net(x_space).cpu().numpy()

    x_numpy = x_space.cpu().numpy()
    u_true = true_solution(x_numpy)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_numpy, u_predicted, label='Predicted Solution', color='red', linewidth=2.5)
    ax.plot(x_numpy, u_true, label='True Analytical Solution', color='blue', linestyle='--', linewidth=2.5)
    ax.set_title('1D Poisson BVP Solution: Prediction vs. Truth', fontsize=16)
    ax.set_xlabel('Domain (x)', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    bvp_string = r"$\frac{d^2u}{dx^2} = -\sin(\pi x), \quad u(0)=0, \quad u(1)=0$"
    fig.suptitle(f"BVP with Analytical Solution\n{bvp_string}", fontsize=18)
    fig.subplots_adjust(top=0.85)
    plt.savefig('poisson_analytical_result.png')
    plt.show()

def plot_numerical_results(path_net, config):
    """Solution from SciPy"""
    print("--- Plotting Results (vs. Numerical Solution) ---")

    def ode_system(x, y):
        f_x = np.tanh(5 * (x - 0.5)) - np.cos(10 * x)
        return np.vstack((y[1], f_x))

    def bc(ya, yb):
        return np.array([ya[0] - config['boundary_start_val'], yb[0] - config['boundary_end_val']])

    x_mesh = np.linspace(config['domain_start'], config['domain_end'], 100)
    y_guess = np.zeros((2, x_mesh.size))
    sol = solve_bvp(ode_system, bc, x_mesh, y_guess)
    x_numerical, u_numerical = sol.x, sol.y[0]

    path_net.eval()
    with torch.no_grad():
        x_space = torch.linspace(config['domain_start'], config['domain_end'], 200).view(-1, 1).to(config['device'])
        u_predicted = path_net(x_space).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_space.cpu().numpy(), u_predicted, label='Predicted Solution', color='red', linewidth=2.5, zorder=10)
    ax.plot(x_numerical, u_numerical, label='Numerical Solution (SciPy)', color='green', linestyle='--', linewidth=2.5)
    ax.set_title('1D Poisson BVP Solution: Prediction vs. Numerical Benchmark', fontsize=16)
    ax.set_xlabel('Domain (x)', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    bvp_string = r"$\frac{d^2u}{dx^2} = \tanh(5(x-0.5)) - \cos(10x), \quad u(0)=0, \quad u(1)=0$"
    fig.suptitle(f"BVP with Numerical Solution\n{bvp_string}", fontsize=18)
    fig.subplots_adjust(top=0.85)
    plt.savefig('poisson_numerical_result.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Solve a 1D Poisson BVP torchpathdiffeq.')
    parser.add_argument(
        '--problem',
        type=str,
        choices=['analytical', 'numerical'],
        required=True,
        help='Specify the problem type: "analytical" for a known solution, "numerical" for a complex source term.'
    )
    args = parser.parse_args()

    config = BASE_CONFIG.copy()
    config.update(PROBLEM_CONFIGS[args.problem])
    config['problem_type'] = args.problem

    model = BVP_pathnet().to(config['device'])
    parallel_integrator = get_parallel_RK_solver(
        config['sampling_type'],
        method=config['method'],
        atol=config['atol'],
        rtol=config['rtol'],
        dtype=torch.float32
    )

    ode_integrand_func = PoissonResidualIntegrand(model, problem_type=args.problem).to(config['device'])

    train_bvp_solver(model, parallel_integrator, ode_integrand_func, config)

    if args.problem == 'analytical':
        plot_analytical_results(model, config)
    else:
        plot_numerical_results(model, config)

if __name__ == '__main__':
    main()
