# torchpathdiffeq

**Parallelized adaptive numerical path integration with backpropagation support.**

Many problems across science and engineering reduce to computing and optimizing a path integral. In physics, the principle of least action integrates a Lagrangian along a trajectory to derive equations of motion. In chemistry, reaction rates depend on integrating over potential energy surfaces. In machine learning, mode connectivity requires integrating loss along curves in parameter space, and reinforcement learning accumulates rewards along paths of decisions. Despite their ubiquity, path integrals remain expensive to compute and difficult to backpropagate through, limiting their practical use.

torchpathdiffeq (TPD) is a PyTorch library that makes path integral computation fast and differentiable. Unlike traditional numerical integrators that evaluate integration steps **sequentially** (each step depends on the previous), TPD exploits a key property of path integrals: when the path is known, all integration steps can be evaluated **simultaneously**. TPD leverages this independence to evaluate the maximum number of integration steps, then adaptively refines only the steps that need it. The result is often **two orders of magnitude faster** than sequential integrators like torchdiffeq, while producing identical accuracy and supporting full backpropagation for gradient-based optimization.

## Installation

```bash
pip install torchpathdiffeq
```

Or install from source for development:

```bash
git clone https://github.com/khegazy/torchpathdiffeq.git
cd torchpathdiffeq
pip install -e ".[dev]"
```

**Requirements**: Python 3.10+, PyTorch, torchdiffeq, einops, psutil

## Quick Start

### The Integrand (`ode_fxn`)

TPD computes definite integrals of the form $\int_{t_i}^{t_f} f(t, \phi(t))\, dt$, where the path $\phi(t)$ is known. The integrand `ode_fxn` takes batched time points of shape `[N, T]` and returns values of shape `[N, D]`, where `N` is the batch size, `T` is the time dimensionality (usually 1), and `D` is the output dimensionality. Because the path is known, the function depends only on `t` — not on accumulated state — which is what allows TPD to evaluate all integration steps in parallel.

```python
# Scalar integrand: f(t) = sin(t), returns shape [N, 1]
def my_integrand(t):
    return torch.sin(t)

# Multi-dimensional integrand: returns shape [N, 2]
def vector_integrand(t):
    return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

# Integrand that evaluates a neural network path phi_theta(t)
def path_loss(t):
    return (path_net(t) - target(t)) ** 2
```

### Computing a Definite Integral

The simplest way to use TPD is the `ode_path_integral` function, which builds the integrator and performs the integration in one call:

```python
import torch
from torchpathdiffeq import ode_path_integral

# Compute integral of sin(t) from 0 to pi (exact answer: 2.0)
result = ode_path_integral(
    ode_fxn=lambda t: torch.sin(t),
    method='dopri5',
    t_init=torch.tensor([0.0]),
    t_final=torch.tensor([3.14159265]),
)
print(result.integral)  # tensor([2.0000])
```

### Path Optimization

For repeated integration (e.g., inside a training loop), construct the solver once and call `integrate()` repeatedly. The solver caches the optimal time mesh from the previous call and reuses it as the starting mesh for the next. The new mesh is often similar to the previous, therefore by reusing the mesh TPD eliminates the vast majority of adaptive mesh refinement.

```python
import torch
from torchpathdiffeq import get_parallel_RK_solver, steps

# Create the solver once
solver = get_parallel_RK_solver(
    sampling_type=steps.ADAPTIVE_UNIFORM,
    method='dopri5',
    atol=1e-6,
    rtol=1e-4,
    remove_cut=0.1,
    t_init=torch.tensor([0.0]),
    t_final=torch.tensor([1.0]),
)

# First call:
# - Builds mesh from scratch
# - Calculates Integral
# - Ready to backpropagate through
result = solver.integrate(
    ode_fxn=my_integrand,
    y0=torch.tensor([0.0]),
)

# Subsequent calls:
# - Reuses cached optimal mesh (warm start)
# - Calculates Integral
# - Ready to backpropagate through
result = solver.integrate(ode_fxn=my_integrand)
```

TPD's core use case is optimizing a path by backpropagating through the integral. Pass `take_gradient=True` to have the solver call `.backward()` on the loss after each batch of steps:

```python
import torch
import torch.nn as nn
from torchpathdiffeq import get_parallel_RK_solver, steps

# A neural network representing a path phi_theta(t)
class PathNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, t):
        return self.net(t)

path_net = PathNet()
optimizer = torch.optim.Adam(path_net.parameters(), lr=1e-3)

# Define a loss integrand that evaluates the path
def loss_integrand(t):
    return (path_net(t) - target_function(t)) ** 2

solver = get_parallel_RK_solver(
    sampling_type=steps.ADAPTIVE_UNIFORM,
    method='bosh3',
    atol=1e-5,
    rtol=1e-3,
)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    result = solver.integrate(
        ode_fxn=loss_integrand,
        t_init=torch.tensor([0.0]),
        t_final=torch.tensor([1.0]),
    )
    if not result.gradient_taken:
        result.integral.backward()
    optimizer.step()
```

For finer control over the loss computation, you can provide a custom `loss_fxn` to `integrate()`:

```python
result = solver.integrate(
    ode_fxn=loss_integrand,
    take_gradient=True,
    loss_fxn=lambda output: output.integral.sum(),
)
```

### Accessing Results

`integrate()` returns an `IntegralOutput` with detailed diagnostics:

```python
result = solver.integrate(ode_fxn=my_integrand)

result.integral         # Computed integral value, shape [D]
result.integral_error   # Estimated total error, shape [D]
result.t                # Evaluation time points, shape [N, C, T]
result.y                # Integrand values at each point, shape [N, C, D]
result.sum_steps        # Per-step contributions to the integral, shape [N, D]
result.error_ratios     # Per-step error ratios (< 1 means accepted), shape [N]
result.t_optimal        # Optimized mesh for reuse, shape [M, T]
result.gradient_taken   # Whether .backward() was already called (always True if take_gradient=True)
```

### Choosing a Method and Tolerances

TPD provides four uniform-sampling Runge-Kutta methods and two variable-sampling methods:

| Method | Type | Order | Quadrature Points (C) |
|---|---|---|---|
| `adaptive_heun` | uniform or variable | 2 | 2 |
| `fehlberg2` | uniform | 2 | 3 |
| `bosh3` | uniform | 3 | 4 |
| `dopri5` | uniform | 5 | 7 |
| `generic3` | variable | 3 | 3 |

Higher-order methods need fewer steps for the same accuracy but cost more per step. `dopri5` is a good default for smooth integrands; `bosh3` balances cost and accuracy; `adaptive_heun` is cheapest per step and well-suited to rough integrands.

Error is controlled by `atol` (absolute) and `rtol` (relative). A step is accepted when its estimated error satisfies `error < atol + rtol * |integral_value|`:

```python
# Tight tolerances for high accuracy
result = ode_path_integral(
    ode_fxn=my_function,
    method='dopri5',
    atol=1e-10,
    rtol=1e-8,
    t_init=torch.tensor([0.0]),
    t_final=torch.tensor([1.0]),
)
```

## Solving ODEs via Path Optimization

Traditional ODE solvers are inherently sequential: each step depends on the previous. TPD reframes ODE solving as path optimization, where a neural network learns the solution by minimizing the ODE residual integrated over the domain:

```python
# The ODE: dx/dt = f(t, x), x(0) = x0
# Train phi_theta(t) such that d/dt phi_theta(t) ≈ f(t, phi_theta(t))

def ode_residual_loss(t):
    t = t.requires_grad_(True)
    phi = path_net(t)
    dphi_dt = torch.autograd.grad(
        phi, t, grad_outputs=torch.ones_like(phi),
        create_graph=True
    )[0]
    return (dphi_dt - f(t, phi)) ** 2

# Integrate the residual over [t0, T] — the integral is the loss
result = solver.integrate(ode_fxn=ode_residual_loss, t_init=t0, t_final=T)
if not result.gradient_taken:
    result.integral.backward()
```

This approach provides three advantages over traditional sequential integration:
1. **Parallel evaluation**: all time points are evaluated simultaneously on GPU.
2. **No error accumulation**: the entire path is optimized at every training step, unlike sequential solvers where errors compound from step to step.
3. **Controllable loss landscape**: by gradually increasing the integration range T (curriculum learning), you can maintain a near-quadratic loss landscape, avoiding the optimization difficulties common with physics-informed losses.

## Uniform vs. Variable Sampling

TPD offers two sampling strategies for placing quadrature points within each integration step:

**Uniform sampling** (`sampling='uniform'`): Quadrature points are placed at fixed fractional positions within each step (e.g., [0, 1/3, 2/3, 1] for bosh3). Weights (tableau b values) are constants. When a step is split, old evaluations are discarded and fresh points are placed in each sub-step.

**Variable sampling** (`sampling='variable'`): Quadrature points can sit at arbitrary positions. Weights are computed dynamically from the actual positions using Sanderse-Veldman formulas. When a step is split, existing evaluations are **reused** and new midpoints are interleaved, avoiding redundant integrand evaluations.

```python
# Variable sampling — reuses evaluations on split, good for expensive integrands
result = ode_path_integral(
    ode_fxn=expensive_function,
    method='generic3',
    sampling='variable',
    atol=1e-6,
    rtol=1e-4,
)
```

## Memory Management

TPD automatically manages GPU memory. It benchmarks the integrand to measure memory per evaluation, then dynamically sizes batches to stay within the available budget:

```python
# Use up to 90% of GPU memory (default)
result = solver.integrate(ode_fxn=my_integrand, total_mem_usage=0.9)

# Or set a fixed batch size
result = solver.integrate(ode_fxn=my_integrand, max_batch=500)
```

## Serial Mode

For comparison or when the integrand depends on accumulated state (traditional ODE), TPD provides a serial backend powered by torchdiffeq:

```python
result = ode_path_integral(
    ode_fxn=lambda t, y: -y,  # dy/dt = -y (serial mode uses f(t, y) signature)
    method='dopri5',
    computation='serial',
    y0=torch.tensor([1.0]),
    t_init=torch.tensor([0.0]),
    t_final=torch.tensor([1.0]),
)
```

## API Reference

### `ode_path_integral()`

High-level function that creates a solver and runs integration in one call.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ode_fxn` | Callable | required | Integrand function. Parallel: `f(t) -> [N, D]`. Serial: `f(t, y)`. |
| `method` | str | required | RK method name (e.g., `'dopri5'`, `'bosh3'`, `'generic3'`). |
| `computation` | str | `'parallel'` | `'parallel'` for GPU-batched, `'serial'` for sequential. |
| `sampling` | str | `'uniform'` | `'uniform'` or `'variable'` (parallel mode only). |
| `atol` | float | `1e-5` | Absolute error tolerance. |
| `rtol` | float | `1e-5` | Relative error tolerance. |
| `t` | Tensor | `None` | Initial time mesh. If None, auto-generated. |
| `t_init` | Tensor | `[0]` | Lower integration bound. |
| `t_final` | Tensor | `[1]` | Upper integration bound. |
| `y0` | Tensor | `[0]` | Initial integral accumulator value. |
| `remove_cut` | float | `0.1` | Threshold for merging low-error step pairs. |
| `device` | str | `None` | Device (`'cuda'`, `'cpu'`). Auto-detected if None. |

### `get_parallel_RK_solver()`

Factory function for creating reusable solver instances.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sampling_type` | str or steps | required | `'uniform'`, `'variable'`, or a `steps` enum value. |
| `method` | str | required | RK method name. |
| `atol` | float | `1e-5` | Absolute error tolerance. |
| `rtol` | float | `1e-5` | Relative error tolerance. |
| `remove_cut` | float | `0.1` | Merge threshold for low-error step pairs. |
| `ode_fxn` | Callable | `None` | Default integrand (can be overridden in `integrate()`). |
| `t_init` | Tensor | `[0]` | Default lower bound. |
| `t_final` | Tensor | `[1]` | Default upper bound. |
| `device` | str | `None` | Target device. |

### `solver.integrate()`

Run integration on a constructed solver.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ode_fxn` | Callable | `None` | Integrand (uses solver default if None). |
| `y0` | Tensor | `None` | Initial accumulator value. |
| `t` | Tensor | `None` | Initial mesh (None = auto or cached). |
| `t_init` | Tensor | `None` | Lower bound (uses solver default if None). |
| `t_final` | Tensor | `None` | Upper bound (uses solver default if None). |
| `N_init_steps` | int | `13` | Number of initial steps when generating mesh. |
| `take_gradient` | bool | `False` | Call `.backward()` on each batch's loss. |
| `total_mem_usage` | float | `None` | Fraction of memory to use (0-1). |
| `max_batch` | int | `None` | Fixed batch size (overrides memory auto-sizing). |
| `loss_fxn` | Callable | `None` | Custom loss (default: returns integral). |

## How It Works

Traditional adaptive integrators evaluate steps sequentially because each step's initial condition depends on the previous step's result. For a definite path integral $\int_{t_i}^{t_f} f(t, \phi(t)) dt$ where the path $\phi(t)$ is known, this dependency vanishes: every step can be evaluated independently.

TPD exploits this by:

1. **Dividing** [t_init, t_final] into N integration steps using barrier points.
2. **Placing** C quadrature points in each step according to the Runge-Kutta tableau (for uniform sampling).
3. **Evaluating** f(t) at all N*C points in a single batched GPU call.
4. **Computing** integral contributions and error estimates via embedded RK pairs (e.g., RK4/RK3 for bosh3).
5. **Accepting** steps with error below tolerance, **splitting** steps that fail by inserting midpoint barriers.
6. **Repeating** only for the new (failed) steps until all converge.

## Speed and Complexity

For a single integration, TPD is generally **two orders of magnitude faster** than sequential integrators. Let us consider an integral that requires O(N) integration steps, and the smallest integration step requires R mesh refinements. A sequential integrator scales superlinearly as O(RN). TPD instead requires O(RN/G) evaluations, since TPD batches G integration steps to simultaneosly evaluate. Typically G is of O(100) and often N/G < 1, making TPD's complexity O(1). After convergence, the solver **prunes** the mesh by merging step pairs with very low error, producing an optimized mesh that is cached for subsequent calls.

On repeated integrations, where the path has changed due to optimization, using the pruned mesh as a warm-starting often eliminates refinement entirely — the cached mesh generally satisfies the error tolerance, so integration completes in a single parallel pass in the typical case where G > N. In practice, when the pruned mesh needs to be updated this requires 1 or 2 extra calls to the GPU.

### Benchmarking Your Integrand

You can compare TPD's parallel solver against the sequential torchdiffeq backend on your own integrand:

```python
import time
import torch
from torchpathdiffeq import ode_path_integral, get_parallel_RK_solver, SerialAdaptiveStepsizeSolver, steps

device = 'cuda'  # or 'cpu'
method = 'dopri5'
atol, rtol = 1e-9, 1e-7
t_init = torch.tensor([0.0], dtype=torch.float64, device=device)
t_final = torch.tensor([1.0], dtype=torch.float64, device=device)
y0 = torch.tensor([0.0], dtype=torch.float64, device=device)

def my_integrand(t, y=None):
    return torch.sin(t * 10) ** 2

# --- Sequential (torchdiffeq) ---
serial = SerialAdaptiveStepsizeSolver(
    ode_fxn=my_integrand, method=method,
    atol=atol, rtol=rtol,
    t_init=t_init, t_final=t_final, device=device,
)
t0 = time.time()
for _ in range(100):
    serial.integrate(y0=y0)
serial_time = (time.time() - t0) / 100

# --- Parallel (torchpathdiffeq) ---
parallel = get_parallel_RK_solver(
    sampling_type=steps.ADAPTIVE_UNIFORM,
    ode_fxn=my_integrand, method=method,
    atol=atol, rtol=rtol,
    t_init=t_init, t_final=t_final, device=device,
)
t0 = time.time()
for _ in range(100):
    parallel.integrate(y0=y0)
parallel_time = (time.time() - t0) / 100

print(f"Sequential: {serial_time:.4f}s | Parallel: {parallel_time:.4f}s | Speedup: {serial_time/parallel_time:.1f}x")
```

## License

CC-BY 4.0

## Citation

If you use torchpathdiffeq in your research, please cite:

```bibtex
@software{torchpathdiffeq,
  author = {Hegazy, Kareem and Blau, Sam and Mahoney, Michael},
  title = {torchpathdiffeq: Parallelized Adaptive Numerical Path Integration},
  url = {https://github.com/khegazy/torchpathdiffeq},
  version = {0.1.0},
}
```
