import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchpathdiffeq import\
    steps,\
    get_parallel_RK_solver,\
    SerialAdaptiveStepsizeSolver,\
    UNIFORM_METHODS,\
    VARIABLE_METHODS\

# Wolf-Schlegel setup
WS_min_init = torch.tensor([1.133, -1.486]).type(torch.float).to('cuda')
WS_min_final = torch.tensor([-1.166, 1.477]).type(torch.float).to('cuda')

def wolf_schlegel_potential(points):
    """
    Compute Wolf-Schlegel potential for batch of 2D points
    points: [batch_size, 2] tensor
    """
    x = points[:, 0:1]  # Keep dims for broadcasting
    y = points[:, 1:2]

    return 10 * (x**4 + y**4 - 2*x**2 - 4*y**2 + x*y + 0.2*x + 0.1*y)

class PathNet(nn.Module):
    def __init__(self, init_point, final_point):
        super(PathNet, self).__init__()
        self.init_point = init_point
        self.final_point = final_point

        hidden_dim = 64
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # Output 2D point

    def _compute_mlp_output(self, t):
        x = F.tanh(self.fc1(t))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def forward(self, t):
        """
        t: [batch_size, 1] tensor
        Returns: [batch_size, 2] tensor of 2D points
        """
        coeffs = [1 - t, t]
        mlp_output = self._compute_mlp_output(t).type(torch.float)
        f0 = self._compute_mlp_output(torch.tensor([0.0]).to(t.device)).type(torch.float)
        f1 = self._compute_mlp_output(torch.tensor([1.0]).to(t.device)).type(torch.float)

        # Boundary constraint enforcement: φ(0) = init, φ(1) = final
        # logic lifted from `CurveNet`
        if len(mlp_output.shape) == 1:
            mlp_output = mlp_output.unsqueeze(0)

        return (coeffs[0] * (mlp_output - f0 + self.init_point) +
                coeffs[1] * (mlp_output - f1 + self.final_point))

class WSLoss(nn.Module):
    """Wolf-Schlegel path loss"""
    def __init__(self, path_net, adjoint=False):
        super(WSLoss, self).__init__()
        self.path_net = path_net
        self.adjoint = adjoint  # Control gradient computation

    def compute_derivative_norm(self, t):
        """Compute ||φ'_θ(t)||"""
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        points = self.path_net(t)  # [batch_size, 2]

        derivatives_norm = []
        for i in range(points.shape[0]):
            # Compute gradient of points w.r.t. t for this batch element
            grad_outputs = torch.ones_like(points[i])  # [2]
            try:
                # Lifted from `CurveNetLoss_base`
                grads = torch.autograd.grad(
                    outputs=points[i],
                    inputs=t,
                    grad_outputs=grad_outputs,
                    create_graph=not self.adjoint,
                    retain_graph=True,
                    allow_unused=True
                )[0]

                if grads is not None:
                    derivative_norm = torch.norm(grads)
                else:
                    derivative_norm = torch.tensor(1.0, device=t.device)

            except RuntimeError as e:
                print(f"Gradient computation failed: {e}")
                derivative_norm = torch.tensor(1.0, device=t.device)

            derivatives_norm.append(derivative_norm)

        return torch.stack(derivatives_norm)

    def forward(self, t):
        """
        t: [batch_size, 1] tensor
        Returns: [batch_size, 1] tensor of losses
        """
        # Control gradient computation based on adjoint flag
        maybe_no_grad = torch.no_grad() if self.adjoint else torch.enable_grad()

        with maybe_no_grad:
            # Get path points
            points = self.path_net(t)  # [batch_size, 2]

            # Compute potential at each point
            potentials = wolf_schlegel_potential(points)  # [batch_size, 1]

            # Compute derivative norms
            derivative_norms = self.compute_derivative_norm(t)  # [batch_size]

            # Combine: L(φ(t)) * ||φ'(t)||
            losses = potentials.squeeze() * derivative_norms

        return losses.view(-1, 1)

def train_wolf_schlegel_path(path_net, ws_loss, integrator, epochs=50, lr=1e-3, device='cuda', id=1):
    """
    Train PathNet to find optimal path minimizing Wolf-Schlegel potential
    """
    print("Training Wolf-Schlegel Path Optimization")
    print(f"Epochs: {epochs}, LR: {lr}")

    optimizer = torch.optim.Adam(path_net.parameters(), lr=lr)
    t_init = torch.tensor([0.0]).to(device).type(torch.float)
    t_final = torch.tensor([1.0]).to(device).type(torch.float)

    loss_history = []

    for epoch in range(epochs):
        path_net.train()
        optimizer.zero_grad()

        # Integrate the path loss
        max_batches = 200
        integral_result = integrator.integrate(
            ws_loss,
            t_init=t_init,
            t_final=t_final,
            max_batch=max_batches  # Adjust based on your integrator
        )

        # Backpropagate
        if max_batches is None:
            integral_result.integral.backward()
        optimizer.step()

        loss_val = integral_result.integral.item()
        loss_history.append(loss_val)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs}: Loss = {loss_val:.6f}")

    save_dict = {
        'epoch': epochs,
        'model_state_dict': path_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(save_dict, f"./wolf_schlegel_model_{id}.pt")
    print("Training completed!")
    return loss_history

def evaluate_and_plot_path(path_net: PathNet, num_points=1000, device='cuda'):
    """
    Evaluate trained path and create comprehensive plots
    """
    print(f"Evaluating path at {num_points} points...")

    path_net.eval()

    # Generate uniform t samples
    t_values = torch.linspace(0, 1, num_points).to(device).view(-1, 1).to(device)

    with torch.no_grad():
        # Get path points
        path_points = path_net(t_values)  # [num_points, 2]

        # Compute Wolf-Schlegel potential along path
        path_potentials = wolf_schlegel_potential(path_points)  # [num_points, 1]

        # Also compute straight line for comparison
        straight_line_points = WS_min_init.to(device) + (WS_min_final.to(device) - WS_min_init.to(device)) * t_values
        straight_potentials = wolf_schlegel_potential(straight_line_points)

    # Convert to numpy for plotting
    t_np = t_values.cpu().numpy().flatten()
    path_points_np = path_points.cpu().numpy()
    path_potentials_np = path_potentials.cpu().numpy().flatten()
    straight_points_np = straight_line_points.cpu().numpy()
    straight_potentials_np = straight_potentials.cpu().numpy().flatten()

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: 2D Path Visualization
    ax1 = axes[0, 0]
    ax1.plot(path_points_np[:, 0], path_points_np[:, 1], 'b-', linewidth=3, label='Learned Path')
    ax1.plot(straight_points_np[:, 0], straight_points_np[:, 1], 'r--', linewidth=2, label='Straight Line')
    ax1.plot(WS_min_init.cpu().numpy()[0], WS_min_init.cpu().numpy()[1], 'go', markersize=10, label='Start')
    ax1.plot(WS_min_final.cpu().numpy()[0], WS_min_final.cpu().numpy()[1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('2D Path Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Potential along path
    ax2 = axes[0, 1]
    ax2.plot(t_np, path_potentials_np, 'b-', linewidth=3, label='Learned Path')
    ax2.plot(t_np, straight_potentials_np, 'r--', linewidth=2, label='Straight Line')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Wolf-Schlegel Potential')
    ax2.set_title('Potential Along Path')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Potential difference (improvement)
    ax3 = axes[1, 0]
    improvement = straight_potentials_np - path_potentials_np
    ax3.plot(t_np, improvement, 'g-', linewidth=3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_xlabel('t')
    ax3.set_ylabel('Potential Reduction')
    ax3.set_title('Improvement Over Straight Line')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Wolf-Schlegel potential landscape (contour plot)
    ax4 = axes[1, 1]

    # Create meshgrid for contour plot
    x_range = torch.linspace(-2, 2, 100)
    y_range = torch.linspace(-2, 2, 100)
    X, Y = torch.meshgrid(x_range, y_range, indexing='xy')

    # Compute potential over grid
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    with torch.no_grad():
        Z_flat = wolf_schlegel_potential(grid_points)
    Z = Z_flat.cpu().numpy().reshape(X.shape)

    # Create contour plot
    contour = ax4.contour(X.numpy(), Y.numpy(), Z, levels=20, alpha=0.6)
    ax4.clabel(contour, inline=True, fontsize=8)

    # Overlay paths on contour
    ax4.plot(path_points_np[:, 0], path_points_np[:, 1], 'b-', linewidth=4, label='Learned Path')
    ax4.plot(straight_points_np[:, 0], straight_points_np[:, 1], 'r--', linewidth=3, label='Straight Line')
    ax4.plot(WS_min_init.cpu().numpy()[0], WS_min_init.cpu().numpy()[1], 'go', markersize=12, label='Start')
    ax4.plot(WS_min_final.cpu().numpy()[0], WS_min_final.cpu().numpy()[1], 'ro', markersize=12, label='End')

    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Wolf-Schlegel Potential Landscape')
    ax4.legend()

    plt.tight_layout()
    plt.savefig("./important_figure.png")

    # Print summary statistics
    print("\nPath Analysis Summary:")
    print(f"Learned Path - Max Potential: {path_potentials_np.max():.4f}, Min: {path_potentials_np.min():.4f}")
    print(f"Straight Line - Max Potential: {straight_potentials_np.max():.4f}, Min: {straight_potentials_np.min():.4f}")
    print(f"Average Improvement: {improvement.mean():.4f}")
    print(f"Max Improvement: {improvement.max():.4f}")

    return {
        't_values': t_np,
        'path_points': path_points_np,
        'path_potentials': path_potentials_np,
        'straight_potentials': straight_potentials_np,
        'improvement': improvement
    }

def full_wolf_schlegel_experiment(integrator, device='cuda', id=1, adjoint=False):
    """
    Complete Wolf-Schlegel path optimization experiment
    """
    print("Wolf-Schlegel Path Optimization Experiment")
    print(f"Using adjoint method: {adjoint}")

    # Setup
    init_point = WS_min_init.to(device)
    final_point = WS_min_final.to(device)

    # Create network and loss
    path_net = PathNet(init_point, final_point).to(device)
    ws_loss = WSLoss(path_net, adjoint=adjoint).to(device)

    print(f"Initial path: {init_point.cpu().numpy()} → {final_point.cpu().numpy()}")

    # Train the path
    loss_history = train_wolf_schlegel_path(
        path_net, ws_loss, integrator,
        epochs=1000, lr=1e-3, device=device, id=id
    )

    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Integral Loss')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    #TODO: Set it up as as proper test
    train = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if train:
        atol = 1e-5
        rtol = 1e-5
        loop_items = zip(
            ['Uniform'],
            [UNIFORM_METHODS],
            [steps.ADAPTIVE_UNIFORM])
        id = 1
        for sampling_name, sampling, sampling_type in loop_items:
            for method in sampling.keys():
                path_net = PathNet(WS_min_init, WS_min_final)
                parallel_integrator = get_parallel_RK_solver(
                    sampling_type,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    remove_cut=0.1,
                    dtype=torch.float
                )
                print("=" * 60)
                full_wolf_schlegel_experiment(parallel_integrator, device=device, id=id, adjoint=False)
                id += 1
    else:
        curve_ckpt = torch.load("./wolf_schlegel_model_3.pt")
        curve = PathNet(WS_min_init, WS_min_final).to(device)
        curve.load_state_dict(curve_ckpt["model_state_dict"])
        evaluate_and_plot_path(curve)


    # Later, when adjoint is implemented:
    # print("=" * 60)
    # path_net_adj, ws_loss_adj, results_adj = full_wolf_schlegel_experiment(
    #     your_integrator, device=device, adjoint=True
    # )

"""
def test_chemistry():
    atol = 1e-5
    rtol = 1e-5
    #loop_items = zip(
    #    ['Uniform', 'Variable'],
    #    [UNIFORM_METHODS, VARIABLE_METHODS],
    #    [steps.ADAPTIVE_UNIFORM, steps.ADAPTIVE_VARIABLE]
    #)
    loop_items = zip(
        ['Uniform'],
        [UNIFORM_METHODS],
        [steps.ADAPTIVE_UNIFORM])
    for sampling_name, sampling, sampling_type in loop_items:
        for method in sampling.keys():
            path_net = PathNet(WS_min_init, WS_min_final)
            criterion = WSLoss(path_net)
            parallel_integrator = get_parallel_RK_solver(
                sampling_type,
                method=method,
                atol=atol,
                rtol=rtol,
                remove_cut=0.1,
                ode_fxn=criterion
            )
            train_path(path_net, criterion, parallel_integrator)

            # error = torch.abs(parallel_integral.integral - serial_integral.integral)
            # error_tolerance = atol + rtol*torch.abs(serial_integral.integral)
            # error_tolerance = error_tolerance*len(parallel_integral.t)
            # assert error < error_tolerance, f"Failed with {sampling_name} ingegration method {method}"
"""""
