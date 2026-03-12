"""
Example integrand functions with known analytical solutions for testing.

Each function computes f(t) and has a corresponding ``*_solution`` function that
returns the exact value of the integral from t_init to t_final. These are used
by the test suite to verify numerical integration accuracy.

The ``ODE_dict`` maps names to ``(integrand, solution, error_cutoff)`` tuples,
where ``error_cutoff`` is the maximum acceptable relative error for tests.
"""

import torch
from typing import Optional


def identity(t: torch.Tensor, y: Optional[torch.Tensor] = None) -> int:
    """
    Constant integrand f(t) = 1.

    Analytical integral: t_final - t_init.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        y: Unused. Present for interface compatibility.

    Returns:
        The constant value 1.
    """
    return 1


def identity_solution(t_init: torch.Tensor, t_final: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution for the integral of 1 from t_init to t_final.

    Args:
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].

    Returns:
        Exact integral value. Shape: [T].
    """
    return t_final - t_init


def t(t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Linear integrand f(t) = t.

    Analytical integral: (t_final^2 - t_init^2) / 2.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        y: Unused. Present for interface compatibility.

    Returns:
        The input t unchanged. Shape: [N, T].
    """
    return t


def t_solution(t_init: torch.Tensor, t_final: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution for the integral of t from t_init to t_final.

    Args:
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].

    Returns:
        Exact integral value. Shape: [T].
    """
    return 0.5*(t_final**2 - t_init**2)


def t_squared(t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Quadratic integrand f(t) = t^2.

    Analytical integral: (t_final^3 - t_init^3) / 3.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        y: Unused. Present for interface compatibility.

    Returns:
        t squared. Shape: [N, T].
    """
    return t**2


def t_squared_solution(t_init: torch.Tensor, t_final: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution for the integral of t^2 from t_init to t_final.

    Args:
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].

    Returns:
        Exact integral value. Shape: [T].
    """
    return (t_final**3 - t_init**3)/3.


def sine_squared(t: torch.Tensor, w: float = 3.7, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Oscillatory integrand f(t) = sin^2(2*pi*w*t).

    Tests the integrator's ability to handle rapid oscillations. Higher w
    values create more oscillations, requiring finer step sizes.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        w: Frequency parameter controlling oscillation rate.
        y: Unused. Present for interface compatibility.

    Returns:
        sin^2(2*pi*w*t) evaluated at each time point. Shape: [N, T].
    """
    return torch.sin(t*w*2*torch.pi)**2


def sine_squared_solution(t_init: torch.Tensor, t_final: torch.Tensor, w: float = 3.7) -> torch.Tensor:
    """
    Analytical solution for the integral of sin^2(2*pi*w*t) from t_init to t_final.

    Uses the identity sin^2(x) = (1 - cos(2x))/2 to derive the closed form.

    Args:
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].
        w: Frequency parameter (must match the integrand).

    Returns:
        Exact integral value. Shape: [T].
    """
    _w = 4*torch.pi*w
    return (t_final - t_init)/2.\
        - (torch.sin(torch.tensor([_w*t_final])) - torch.sin(torch.tensor([_w*t_init])))/(2*_w)


def exp(t: torch.Tensor, a: float = 5, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Exponential integrand f(t) = exp(a*t).

    Tests integration of rapidly growing functions. The growth rate ``a``
    makes this increasingly difficult for larger integration domains.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        a: Growth rate parameter.
        y: Unused. Present for interface compatibility.

    Returns:
        exp(a*t) evaluated at each time point. Shape: [N, T].
    """
    return torch.exp(a*t)


def exp_solution(t_init: torch.Tensor, t_final: torch.Tensor, a: float = 5) -> torch.Tensor:
    """
    Analytical solution for the integral of exp(a*t) from t_init to t_final.

    Args:
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].
        a: Growth rate parameter (must match the integrand).

    Returns:
        Exact integral value. Shape: [T].
    """
    return (torch.exp(torch.tensor([t_final*a]))\
        - torch.exp(torch.tensor([t_init*a])))/a


def damped_sine(t: torch.Tensor, w: float = 3.7, a: float = 5, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Damped oscillatory integrand f(t) = exp(-a*t) * sin(2*pi*w*t).

    Combines exponential decay with oscillation. This is one of the more
    challenging test cases as it requires the integrator to handle both
    rapid sign changes and varying amplitude.

    Args:
        t: Time points at which to evaluate the integrand. Shape: [N, T].
        w: Frequency parameter controlling oscillation rate.
        a: Damping rate parameter controlling exponential decay.
        y: Unused. Present for interface compatibility.

    Returns:
        exp(-a*t)*sin(2*pi*w*t) evaluated at each time point. Shape: [N, T].
    """
    return torch.exp(-a*t)*torch.sin(w*t*2*torch.pi)


def damped_sine_solution(t_init: torch.Tensor, t_final: torch.Tensor, w: float = 3.7, a: float = 5) -> torch.Tensor:
    """
    Analytical solution for the integral of exp(-a*t)*sin(2*pi*w*t) from t_init to t_final.

    Derived via integration by parts (or Laplace transform tables).

    Args:
        t_init: Lower integration bound. Shape: [T].
        t_final: Upper integration bound. Shape: [T].
        w: Frequency parameter (must match the integrand).
        a: Damping rate parameter (must match the integrand).

    Returns:
        Exact integral value. Shape: [T].
    """
    _w = 2*torch.pi*w
    def numerator(t, w, a):
        t = torch.tensor([t])
        return torch.exp(-a*t)*(a*torch.sin(_w*t) + _w*torch.cos(_w*t))
    return -1*(numerator(t_final, w, a) - numerator(t_init, w, a))/(a**2 + _w**2)


# Registry of test integrands: maps name -> (integrand_fn, analytical_solution_fn, error_cutoff).
# error_cutoff is the maximum acceptable relative error |computed - exact| / |exact|.
ODE_dict = {
    "t" : (t, t_solution, 1e-7),
    "t_squared" : (t_squared, t_squared_solution, 1e-6),
    "sine_squared" : (sine_squared, sine_squared_solution, 1e-6),
    "exp" : (exp, exp_solution, 1e-6),
    "damped_sine" : (damped_sine, damped_sine_solution, 1e-6)
}