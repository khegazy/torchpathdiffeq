import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scipy.integrate import solve_ivp

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
    't_final': 20.0,
    'train': True,
    'EPOCHS': 10000,
    'LEARNING_RATE': 1e-4,
    'CONSERVATION_WEIGHT': 0.5,
    'initial_points_to_run': [(1.0, 2.0)]
}

class PositionalEncoding(nn.Module):
    """
    Embeds 1D time into a higher dimensional vector of sinusoids
    """""
    def __init__(self, num_encoding_functions=6, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.frequencies = 2.0 ** torch.arange(num_encoding_functions)

    def forward(self, t):
        self.frequencies = self.frequencies.to(t.device)
        encoded = [torch.sin(t * freq) for freq in self.frequencies] + \
                  [torch.cos(t * freq) for freq in self.frequencies]
        encoded_tensor = torch.cat(encoded, dim=-1)
        if self.include_input:
            encoded_tensor = torch.cat([t, encoded_tensor], dim=-1)
        return encoded_tensor

    def get_output_dim(self):
        return (2 * self.num_encoding_functions) + (1 if self.include_input else 0)


# --- Path Parameterization Network in log-space ---
class ODEPathNet2D(nn.Module):
    """
    A PathNet that operates in log-space to guarantee positive outputs.
    The output is y(t) = exp(log(y0) + N(t) - N(0)), ensuring y(t) > 0 and y(0) = y0.
    """
    def __init__(self, initial_point_2d, num_encoding_functions=6):
        super(ODEPathNet2D, self).__init__()
        # Store the log of the initial point. Add a small epsilon for safety.
        initial_tensor = torch.tensor(initial_point_2d, dtype=torch.float32)
        self.register_buffer('log_initial_point', torch.log(initial_tensor + 1e-8).view(1, 2))

        self.encoder = PositionalEncoding(num_encoding_functions=num_encoding_functions)

        input_dim = self.encoder.get_output_dim()
        hidden_dim = 256

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def _compute_mlp_output(self, t):
        """ The MLP learns the displacement in log-space. """
        encoded_t = self.encoder(t)
        x = F.silu(self.fc1(encoded_t))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, t):
        """ The final output is exponentiated to return to normal space. """
        mlp_output = self._compute_mlp_output(t)
        t0 = torch.zeros(1, 1, device=t.device, dtype=t.dtype)
        mlp_at_t0 = self._compute_mlp_output(t0)

        # Calculate the displacement in log-space and add to the initial log-point
        current_log_coords = self.log_initial_point + (mlp_output - mlp_at_t0)

        # Exponentiate to get the final positive coordinates
        return torch.exp(current_log_coords)

class LotkaVolterraResidualLoss(nn.Module):
    def __init__(self, path_net, alpha=1.0, beta=1.0, delta=1.0, gamma=1.0, conservation_weight=0.1):
        super(LotkaVolterraResidualLoss, self).__init__()
        self.path_net = path_net
        self.alpha, self.beta, self.delta, self.gamma = alpha, beta, delta, gamma
        self.conservation_weight = conservation_weight

        with torch.no_grad():
            initial_point = self.path_net(torch.zeros(1,1)).cpu()
            x0, y0 = initial_point[0, 0], initial_point[0, 1]
            self.h_initial = self.delta * x0 - self.gamma * torch.log(x0) + self.beta * y0 - self.alpha * torch.log(y0)

    def system_equations(self, t, state_vec):
        x, y = state_vec[:, 0], state_vec[:, 1]
        dx_dt = self.alpha * x - self.beta * x * y
        dy_dt = self.delta * x * y - self.gamma * y
        return torch.stack([dx_dt, dy_dt], dim=1)

    def conserved_quantity(self, state_vec):
        x, y = state_vec[:, 0], state_vec[:, 1]
        eps = 1e-8
        return self.delta * x - self.gamma * torch.log(x + eps) + self.beta * y - self.alpha * torch.log(y + eps)

    def compute_derivative(self, t):
        state_vec = self.path_net(t)
        derivatives = []
        for i in range(state_vec.shape[1]):
            grad_outputs = torch.zeros_like(state_vec)
            grad_outputs[:, i] = 1.0
            derivative = torch.autograd.grad(outputs=state_vec, inputs=t, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            derivatives.append(derivative)
        return torch.cat(derivatives, dim=1)

    def forward(self, t):
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            path_derivative = self.compute_derivative(t)
            path_value = self.path_net(t)
            system_derivative = self.system_equations(t, path_value)
            residual_loss = torch.sum((path_derivative - system_derivative)**2, dim=1)
            h_current = self.conserved_quantity(path_value)
            h_initial_tensor = self.h_initial.to(h_current.device)
            conservation_loss = (h_current - h_initial_tensor)**2
            total_loss = residual_loss + self.conservation_weight * conservation_loss
        return total_loss.view(-1, 1)

def train_solver(path_net, integrator, epochs=5000, lr=1e-4, device='cpu', t_final_val=10.0, conservation_weight=0.1, inizz=0.00):
    initial_point_np = torch.exp(path_net.log_initial_point.data).cpu().numpy().flatten()
    print(f"\n--- Training for Initial Point: {initial_point_np} ---")
    optimizer = torch.optim.Adam(path_net.parameters(), lr=lr)
    loss_func = LotkaVolterraResidualLoss(path_net, conservation_weight=conservation_weight).to(device)
    t_init = torch.tensor([0.0], device=device)
    t_final = torch.tensor([t_final_val], device=device)

    for epoch in range(epochs):
        path_net.train()
        optimizer.zero_grad()
        integral_result = integrator.integrate(loss_func, t_init=t_init, t_final=t_final)
        loss = integral_result.integral
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d}/{epochs}: Loss = {loss.item():.8f}")
    full_checkpoint = {
        'model_state_dict': path_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(full_checkpoint, f'./path_net_volterra_init={inizz}_weight={conservation_weight}_PE.pt')
    print("--- Training Completed ---")

def plot_multi_orbit_portrait(models_and_points, t_final_val=10.0, weight=0.00):
    print("\n--- Plotting Multi-Orbit Phase Portrait ---")

    plt.figure(figsize=(10, 10))
    colors = ['red', 'green', 'purple', 'orange']

    for i, (model, initial_point) in enumerate(models_and_points):
        color = colors[i % len(colors)]

        # Generate ground truth for this initial point
        def lv_scipy(t, y):
            alpha, beta, delta, gamma = 1.0, 1.0, 1.0, 1.0
            return [alpha*y[0] - beta*y[0]*y[1], delta*y[0]*y[1] - gamma*y[1]]

        t_eval = np.linspace(0, t_final_val, 1000)
        sol = solve_ivp(lv_scipy, [0, t_final_val], initial_point, t_eval=t_eval)
        x_true, y_true = sol.y

        model.eval()
        with torch.no_grad():
            t_space = torch.linspace(0, t_final_val, 1000).view(-1, 1).to(next(model.parameters()).device)
            trajectory = model(t_space).cpu().numpy()
        x_pred, y_pred = trajectory[:, 0], trajectory[:, 1]
        phase = True

        if phase:
            plt.plot(x_pred, y_pred, label=f'Predicted (y0={initial_point[1]})', color=color, lw=2.5)
            plt.plot(x_true, y_true, label=f'Ground Truth (y0={initial_point[1]})', color=color, linestyle='--', lw=2, alpha=0.8)
            plt.plot(x_pred[0], y_pred[0], 'o', color='black', markersize=6)
            plt.title('Lotka-Volterra Phase Portrait', fontsize=16)
            plt.xlabel('Prey Population (x)'), plt.ylabel('Predator Population (y)')
        else:
            plt.plot(t_space, x_pred, label=f'Predicted x (y0={initial_point[1]})', color=colors[0], lw=2.5)
            plt.plot(t_space, y_pred, label=f'Predicted y (y0={initial_point[1]})', color=colors[1], lw=2.5)
            plt.plot(t_eval, x_true, label=f'Ground Truth x (y0={initial_point[1]})', color=colors[2], linestyle='--', lw=2, alpha=0.8)
            plt.plot(t_eval, y_true, label=f'Ground Truth y (y0={initial_point[1]})', color=colors[3], linestyle='--', lw=2, alpha=0.8)
            plt.title('Lotka-Volterra learned ODE', fontsize=16)
            plt.ylabel('Prey/Predator Population (x/y)'), plt.xlabel('Time (t)')

        plt.legend(), plt.grid(True), plt.axis('equal')
        plot_name = f'volterra_weight={weight}_nPE_phase={phase}'
        plt.savefig(plot_name + ".png")
        plt.show()

if __name__ == '__main__':

    train = test_config['train']

    trained_models_and_points = []

    if train:
        parallel_integrator = get_parallel_RK_solver(
            test_config['sampling_type'], method=test_config['method'],
            atol=test_config['atol'], rtol=test_config['rtol'],
            remove_cut=0.1, dtype=torch.float
        )

        for initial_point in test_config['initial_points_to_run']:
            model = ODEPathNet2D(initial_point_2d=initial_point).to(test_config['device'])
            train_solver(
                path_net=model, integrator=parallel_integrator, epochs=test_config['EPOCHS'],
                lr=test_config['LEARNING_RATE'], device=test_config['device'], t_final_val=test_config['t_final'],
                conservation_weight=test_config['CONSERVATION_WEIGHT'],
                inizz=initial_point[1]
            )
            trained_models_and_points.append((model, initial_point))
    else:
        for initial_point in test_config['initial_points_to_run']:
            file_name = f'path_net_volterra_init={initial_point[1]}_weight={test_config["CONSERVATION_WEIGHT"]}_nPE.pt'
            model_ckpt = torch.load(file_name, map_location=test_config['device'])
            model = ODEPathNet2D(initial_point_2d=initial_point)
            model.load_state_dict(model_ckpt['model_state_dict'])
            trained_models_and_points.append((model, test_config['initial_points_to_run'][0]))

    plot_multi_orbit_portrait(trained_models_and_points, t_final_val=test_config['t_final'], weight=test_config['CONSERVATION_WEIGHT'])
