import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    SerialAdaptiveStepsizeSolver,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS

test_config = {
    'method': 'adaptive_heun',
    'sampling_type': steps.ADAPTIVE_UNIFORM,
    'atol': 1e-5,
    'rtol': 1e-3,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'max_batch': 250,
    't_init': torch.tensor([0.0]),
    't_final': 10.00,
    'path_net_hidden_dim': 64
}

class ODE_pathnet(nn.Module):
    def __init__(self, init_point):
        super(ODE_pathnet, self).__init__()
        self.init_point = init_point

        hidden_dim = test_config['path_net_hidden_dim']
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def _compute_mlp_output(self, t):
        x = F.tanh(self.fc1(t))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, t):
        mlp_output = self._compute_mlp_output(t).type(torch.float)
        f0 = self._compute_mlp_output(torch.tensor([0.0]).to(t.device)).type(torch.float)

        if len(mlp_output.shape) == 1:
            mlp_output = mlp_output.unsqueeze(0)

        return (mlp_output - f0 + self.init_point)


class ODEResidualLoss(nn.Module):
    def __init__(self, path_net):
        super(ODEResidualLoss, self).__init__()
        self.path_net = path_net

    def ode_func(self, t, y):
        return torch.exp(-2 * t) - 3 * y

    def compute_derivative(self, t):
        y = self.path_net(t)

        y_dot = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]

        if y_dot is None:
            # Fallback to a zero tensor if the gradient is None
            y_dot = torch.zeros_like(y)

        return y_dot

    def forward(self, t):
        # Ensure `t` is a leaf variable that requires gradients for derivative computation.
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            y_dot_approx = self.compute_derivative(t)

            y_approx = self.path_net(t)

            y_dot_true = self.ode_func(t, y_approx)

            constraint = (self.path_net(test_config['t_init']) - self.path_net.init_point) ** 2
            weight = 1

            #NOTE: .view(-1, 1) ensures the output has shape [batch_size, 1]
            loss = (y_dot_approx - y_dot_true) ** 2
            total_loss = loss + weight * constraint

        return total_loss

def train_ode_solver(path_net, integrator, epochs=1000, lr=1e-3, device='cpu', t_final_val=test_config['t_final']):
    print("--- Starting ODE Solver Training ---")
    print(f"Epochs: {epochs}, LR: {lr}, Device: {device}")

    optimizer = torch.optim.Adam(path_net.parameters(), lr=lr)
    ode_loss_func = ODEResidualLoss(path_net).to(device)

    t_init = torch.tensor([0.0], device=device)
    t_final = torch.tensor([t_final_val], device=device)

    loss_history = []

    for epoch in range(epochs):
        path_net.train()
        optimizer.zero_grad()

        integral_result = integrator.integrate(
            ode_loss_func,
            t_init=t_init,
            t_final=t_final,
        )
        integral_loss = integral_result.integral

        integral_loss.backward()
        optimizer.step()

        loss_history.append(integral_loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs}: Total Loss = {integral_loss.item():.6f} ")

    print("--- Training Completed ---")
    return loss_history

def plot_results(path_net, t_final_val=test_config['t_final'], y0_val=4.0):
    print("--- Plotting Results ---")

    def true_solution(t):
        # Analytical solution: y(t) = exp(-2t) + 3*exp(-3t)
        return np.exp(-2 * t) + 3 * np.exp(-3 * t)

    path_net.eval()
    with torch.no_grad():
        t_space = torch.linspace(0, t_final_val, 200).view(-1, 1).to(next(path_net.parameters()).device)
        y_predicted = path_net(t_space).cpu().numpy()

    t_numpy = t_space.cpu().numpy()
    y_true = true_solution(t_numpy)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(t_numpy, y_predicted, label='Predicted Solution (PathNet)', color='red', linewidth=2.5)
    ax.plot(t_numpy, y_true, label='True Analytical Solution', color='blue', linestyle='--', linewidth=2.5)

    ax.set_title('ODE Solution: Prediction vs. Truth', fontsize=16)
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('y(t)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ode_string = r"$\frac{dy}{dt} + 3y = e^{-2t}, \quad y(0)=" + f"{y0_val:.0f}$"
    fig.suptitle(f"Solving the Initial Value Problem:\n{ode_string}", fontsize=18)

    fig.subplots_adjust(top=0.85)
    plt.savefig('simple_ode.png')
    plt.show()

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = int(1e4)
    LEARNING_RATE = 1e-3
    Y_INITIAL = 4.0

    model = ODE_pathnet(init_point=Y_INITIAL).to(DEVICE)

    parallel_integrator = get_parallel_RK_solver(
        test_config['sampling_type'],
        method=test_config['method'],
        atol=test_config['atol'],
        rtol=test_config['rtol'],
        remove_cut=0.1,
        dtype=torch.float
    )

    train_ode_solver(
        path_net=model,
        integrator=parallel_integrator,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE
    )

    plot_results(model)
