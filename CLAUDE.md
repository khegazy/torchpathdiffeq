# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**torchpathdiffeq** is a PyTorch library for **parallelized adaptive numerical quadrature** (path integration). It computes definite integrals ∫f(t)dt by evaluating multiple Runge-Kutta integration steps simultaneously on GPU, unlike traditional sequential integrators. Currently in Alpha (v0.1.0), Python 3.10+.

**Critical distinction**: The integrand `ode_fxn` depends only on `t` (and optional extra args), NOT on accumulated state `y`. This independence between steps is what enables parallelization. The `y` parameter in example functions exists only for interface compatibility and is unused. This is numerical quadrature, not traditional ODE solving where y_{n+1} depends on y_n.

## Build & Test Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests (354 cases: 142 integration + 212 unit)
pytest tests/ -v

# Run all tests with coverage
pytest --cov=torchpathdiffeq --cov-report=xml

# Run a single test file
pytest tests/test_integrals.py -v

# Run a specific parametrized case
pytest tests/test_integrals.py -k "dopri5 and damped_sine" -v

# Lint with ruff
ruff check torchpathdiffeq/

# Type check
mypy torchpathdiffeq/
```

## Test Structure

Tests are in a flat `tests/` directory using `@pytest.mark.parametrize` for method × integrand combinations. Shared constants and helpers live in `tests/_helpers.py` (not `conftest.py`, because `--import-mode=importlib` in pyproject.toml prevents importing conftest directly). The `conftest.py` adds `tests/` to `sys.path` so `_helpers` is importable.

**Integration tests** (142 cases):

| File | What it tests |
|---|---|
| `test_integrals.py` | Numerical accuracy: each uniform method × each integrand from `ODE_dict`, plus time ordering, mesh ordering, step continuity |
| `test_data_types.py` | float32 and float64 handling (damped_sine integrand, relaxed cutoffs) |
| `test_adaptivity.py` | Step adding (coarse mesh grows) and removal (dense mesh shrinks), mesh convergence |
| `test_tableau.py` | Butcher tableau b weights sum to 1 (uniform and variable methods) |
| `test_ode_path_integral.py` | `ode_path_integral()` wrapper matches direct solver construction (parallel + serial) |
| `test_chemistry.py` | Wolf-Schlegel 2D potential: parallel vs serial agreement |

**Unit tests** (212 cases) — test individual functions in isolation with hand-crafted inputs:

| File | What it tests |
|---|---|
| `test_rk_integral.py` | `_RK_integral()`: constant/linear/multi-step/multi-dim/zero-width/variable-b inputs, output shapes |
| `test_base_and_methods.py` | `get_sampling_type()`, `_Tableau` dtype conversion, `_get_method()` factory, `_VARIABLE_THIRD_ORDER` weight formulas, example integrand functions and analytical solutions |
| `test_error_computation.py` | `_error_norm()`, `_rec_remove()` (adjacent mask resolution), `_compute_error_ratios_absolute()`, `_compute_error_ratios_cumulative()` |
| `test_interpolation.py` | `_t_step_interpolate()`: endpoints, shapes, monotonicity, tableau c matching, scaling, independence across steps, tiny steps (×4 methods) |
| `test_sorted_insert.py` | `_get_sorted_indices()` and `_insert_sorted_results()`: start/end/middle insertion, 1D/2D/3D tensors |
| `test_base_solver.py` | `SolverBase._set_dtype()`, `set_dtype_by_input()`, `_check_variables()`, `_integral_loss()` |
| `test_methods_extra.py` | `_VARIABLE_SECOND_ORDER.tableau_b()`, `to_device()`, variable subclass dtype conversion |
| `test_rk_solver_methods.py` | `get_parallel_RK_solver()` factory, `_get_tableau_b()` (uniform+variable), `_get_num_tableau_c()` |
| `test_serial_solver.py` | `SerialAdaptiveStepsizeSolver.integrate()` with constant/linear/custom bounds |
| `test_adaptive_steps.py` | `_adaptively_add_steps()`: pass/fail/mixed error ratios, midpoint placement, barrier ordering |
| `test_record_and_sort.py` | `_record_results()` init/merge/accumulate, `_sort_record()` ordering |
| `test_evaluate_and_merge.py` | `_evaluate_adaptive_y()` + `_merge_excess_t()` for both uniform and variable solvers |
| `test_prune_and_optimal.py` | `prune_excess_t()`, `_get_optimal_t_step_barriers()` |

**Note**: Variable sampling integration tests are not currently enabled (only uniform methods are tested). The serial solver (torchdiffeq) may evaluate the integrand slightly outside `[t_init, t_final]` during adaptive stepping — serial-facing callables must not assert strict domain bounds. UNIFORM_METHODS are global singletons — tests that mutate dtype must save/restore original tableau tensors (float16 truncation is irreversible).

## Architecture

### Class Hierarchy

```
DistributedEnvironment (distributed.py)
  └── SolverBase (base.py)
        ├── SerialAdaptiveStepsizeSolver (serial_solver.py)
        └── ParallelAdaptiveStepsizeSolver (parallel_solver.py)
              ├── ParallelUniformAdaptiveStepsizeSolver
              │     └── RKParallelUniformAdaptiveStepsizeSolver (runge_kutta.py)
              └── ParallelVariableAdaptiveStepsizeSolver
                    └── RKParallelVariableAdaptiveStepsizeSolver (runge_kutta.py)
```

### Key Modules

- **path_integral.py** — Public entry point `ode_path_integral()`. Routes to parallel or serial solver, and to uniform or variable sampling for parallel.
- **parallel_solver.py** (~1145 lines) — Core parallel integration engine. Contains the main adaptive integration loop with memory-aware batching, error-driven step refinement (add/remove/merge), and result recording.
- **runge_kutta.py** — RK-specific solver subclasses implementing `_calculate_integral()` using tableau weights. `_RK_integral()` computes: `integral = y0 + Σ(h * Σ(b_i * f(t_i)))`.
- **methods.py** — Defines RK tableaux (`_Tableau` with c, b, b_error). Uniform methods: adaptive_heun, fehlberg2, bosh3, dopri5. Variable methods: adaptive_heun, generic3 (Sanderse-Veldman).
- **base.py** — `SolverBase` abstract class, `IntegralOutput`/`MethodOutput` dataclasses, `steps` enum.
- **serial_solver.py** — Thin wrapper around `torchdiffeq.odeint` for sequential solving (different interface — this one IS state-dependent).
- **distributed.py** — Multi-GPU/SLURM distributed training support.
- **examples.py** — Test integrands with known analytical solutions: t, t², sin², exp, damped_sine. Each entry in `ODE_dict` is (function, analytical_solution, error_cutoff).

### Tensor Shape Conventions

- **N**: number of integration steps in a batch
- **C**: number of quadrature points per step (from RK tableau, e.g. 4 for bosh3, 7 for dopri5)
- **T**: dimensionality of time (usually 1, but supports multi-dimensional)
- **D**: dimensionality of ode_fxn output

Key tensors: `t: [N, C, T]`, `y: [N, C, D]`, `tableau_b: [1, C, 1] or [N, C, 1]`, `y0: [D]`

### Core Algorithm (parallel_solver.py `integrate()`)

**t_step_barriers** are boundary points dividing [t_init, t_final] into integration steps. Between consecutive barriers, C quadrature points are placed per the RK tableau. **t_step_trackers** is a boolean array where True = step needs evaluation.

1. **Initialize barriers**: When `t=None`, creates ~√N_init_steps evenly-spaced segments with randomized sub-barriers to avoid alignment issues.
2. **Main loop** (while any t_step_trackers are True):
   a. Determine `max_steps` from GPU memory budget or explicit `max_batch`
   b. Select up to max_steps unevaluated steps from t_step_trackers
   c. Interpolate C quadrature points within each step via `_t_step_interpolate`
   d. Evaluate `ode_fxn` on all flattened quadrature points, reshape to [N, C, D]
   e. Compute integral + error via `_calculate_integral` (RK weighted sum using b and b_error tableaux)
   f. Compute error ratios (absolute or cumulative mode)
   g. **`_adaptively_add_steps`**: Steps with error_ratio < 1 are accepted (marked done). Steps with error_ratio ≥ 1 are rejected — a new midpoint barrier is inserted, splitting into two smaller steps for re-evaluation.
   h. Record accepted results into `record` dict (sorted insertion by time)
   i. If `take_gradient`: call `loss.backward()` on this batch
3. **Post-convergence**: Sort record, compute optimal barriers via `_get_optimal_t_step_barriers` (prune low-error pairs with `prune_excess_t`, add where error still high), save for reuse.

### Error Computation

Two modes controlled by `use_absolute_error_ratio`:
- **Absolute** (default): `error_ratio = |step_error| / (atol + rtol * |total_integral|)` — uses the converging total integral value
- **Cumulative**: `error_ratio = |step_error| / (atol + rtol * |cumulative_sum_to_step|)` — uses running sum, more like traditional ODE error

Error for multivariate outputs: `_error_norm = sqrt(mean(error²))` (RMS over D dimension).

**error_ratios_2steps**: Combined error of consecutive step pairs, used for merging. When `error_ratio_2steps < remove_cut` (default 0.1), pairs are merged. `_rec_remove` ensures no adjacent pairs are both flagged.

### Uniform vs Variable Sampling

- **Uniform**: Quadrature points at fixed fractional positions within each step (tableau.c values like [0, 0.5, 0.75, 1.0] for bosh3). Tableau b values are constants.
  - *Splitting*: Discards old evaluations entirely. Splits step at midpoint into two sub-steps, places fresh quadrature points at standard c positions in each.
  - *Merging*: Creates a new step spanning [A.start, B.end] and re-interpolates C quadrature points at the standard c positions.
- **Variable**: Quadrature points at arbitrary positions. Tableau b values computed dynamically from actual point positions using Sanderse-Veldman formulas (`_VARIABLE_THIRD_ORDER.tableau_b(c)`).
  - *Splitting*: Reuses existing evaluations. Inserts new midpoints between each pair of consecutive points, interleaves old+new into a 2C-length array, reshapes into two C-point sub-steps. This avoids redundant ode_fxn evaluations.
  - *Merging*: Concatenates points from both steps (C + C-1 = 2C-1, shared boundary not duplicated), then subsamples every other point to get C points. The `tableau_b(c)` call will recompute correct weights for the new positions.

### Memory Management

- `_setup_memory_checks`: Benchmarks ode_fxn with increasing N, measures memory per evaluation (×2.1 safety factor)
- `_get_max_ode_evals`: usable_memory / ode_unit_mem_size
- `_get_usable_memory`: free_memory - buffer, where buffer = (1 - total_mem_usage) × total_memory
- Supports both CUDA (`torch.cuda.mem_get_info`) and CPU (`psutil.virtual_memory`)

### Gradient/Loss Support

- `take_gradient` flag triggers `loss.backward()` after each batch
- `loss_fxn` defaults to returning the integral value itself
- Results are `.detach()`ed when gradients are taken to avoid graph accumulation across batches

### Warm-Starting & Early Exit

- Solver caches `t_step_barriers_previous` and `previous_ode_fxn`. On subsequent calls with the same integrand and `t=None`, the cached optimal barriers are reused as the initial mesh instead of generating a random one.
- `max_path_change`: When a user-provided mesh (`t` is given) has too many failing steps (fail_ratio ≥ max_path_change), the solver exits early and returns None instead of refining indefinitely.

### Dtype Management

- `SolverBase._set_dtype` maintains two tolerance levels: `atol`/`rtol` (for integration error control, can be float32) and `atol_assert`/`rtol_assert` (for geometric assertions like time ordering, always looser at 1e-6 minimum). This prevents false assertion failures when using lower-precision dtypes.

## Dependencies

Core: torch, torchdiffeq, einops, psutil, numpy
