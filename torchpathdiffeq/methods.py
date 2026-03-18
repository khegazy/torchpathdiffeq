"""
Runge-Kutta method definitions and Butcher tableau data.

This module defines the numerical integration methods available in
torchpathdiffeq. Each method is specified by a Butcher tableau containing:

- **c** (nodes): Fractional positions within a step where the integrand is
  evaluated. For example, c=[0, 0.5, 1] means evaluations at the start,
  midpoint, and end of each step.
- **b** (weights): Weights for combining the evaluations into the integral
  estimate. The integral over one step is: h * sum(b_i * f(t_i)).
- **b_error**: Difference between the primary (order p) and embedded
  (order p-1) method weights. Used to estimate the local truncation error
  without extra evaluations: error = h * sum(b_error_i * f(t_i)).

There are two families of methods:

**Uniform methods** (``UNIFORM_METHODS``): The tableau c values are fixed
constants. Quadrature points within each step are always at the same
fractional positions regardless of step size. Simple and efficient.

**Variable methods** (``VARIABLE_METHODS``): The tableau b weights are
recomputed dynamically based on the actual positions of the quadrature
points (the c values). This allows quadrature points to be at arbitrary
positions within each step, which is useful when refining the mesh by
inserting new points between existing ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import torch

from .base import steps


@dataclass
class _Tableau:
    """
    Butcher tableau for a Runge-Kutta integration method.

    Stores the coefficients that define how quadrature points are placed
    within a step (c) and how integrand evaluations are weighted to produce
    the integral (b) and error estimate (b_error).

    Attributes:
        c: Node positions as fractions of step size. c[0]=0 is the step start,
            c[-1]=1 is the step end. Shape: [C] where C is the number of
            quadrature points per step.
        b: Weights for the primary (higher-order) integral estimate.
            Shape: [C] for uniform methods, [1, C] for some methods.
        b_error: Difference between primary and embedded method weights.
            The error estimate is h * sum(b_error * f(t)). Shape: same as b.
    """

    c: torch.Tensor
    b: torch.Tensor
    b_error: torch.Tensor

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert all tableau tensors to the specified dtype."""
        self.c = self.c.to(dtype)
        self.b = self.b.to(dtype)
        self.b_error = self.b_error.to(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move all tableau tensors to the specified device."""
        self.c = self.c.to(device)
        self.b = self.b.to(device)
        self.b_error = self.b_error.to(device)


class MethodClass:
    """
    A complete uniform-sampling RK method: order + tableau.

    Wraps a ``_Tableau`` with metadata about the method's convergence order.
    The order determines how quickly the error decreases as step size shrinks
    (error ~ O(h^order)).

    Attributes:
        order: Convergence order of the method.
        tableau: The Butcher tableau defining the method's coefficients.
    """

    order: int
    tableau: _Tableau

    def __init__(self, order: int, tableau: _Tableau) -> None:
        self.order = order
        self.tableau = tableau

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tableau to the specified dtype."""
        self.tableau.to_dtype(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move tableau to the specified device."""
        self.tableau.to_device(device)


# =============================================================================
# Uniform Sampling Adaptive Methods
# =============================================================================
# These methods use fixed tableau c values (quadrature point positions are
# constant fractions of the step size). The b weights are also constant.

# Adaptive Heun: 2nd-order method with 2 evaluations per step (endpoints only)
_ADAPTIVE_HEUN = MethodClass(
    order=2,
    tableau=_Tableau(
        c=torch.tensor([0.0, 1.0], dtype=torch.float64),
        b=torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        b_error=torch.tensor([[0.5, -0.5]], dtype=torch.float64),
    ),
)

# Fehlberg2: 2nd-order method with 3 evaluations per step (start, mid, end)
_FEHLBERG2 = MethodClass(
    order=2,
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        b=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
        b_error=torch.tensor([-1 / 512, 0, 1 / 512], dtype=torch.float64),
    ),
)

# Bogacki-Shampine: 3rd-order method with 4 evaluations per step
_BOGACKI_SHAMPINE = MethodClass(
    order=3,
    tableau=_Tableau(
        c=torch.tensor([0.0, 0.5, 0.75, 1.0], dtype=torch.float64),
        b=torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.0], dtype=torch.float64),
        b_error=torch.tensor(
            [2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=torch.float64
        ),
    ),
)

# Dormand-Prince: 5th-order method with 7 evaluations per step
_DORMAND_PRINCE_SHAMPINE = MethodClass(
    order=5,
    tableau=_Tableau(
        c=torch.tensor(
            [0.0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1.0, 1.0], dtype=torch.float64
        ),
        b=torch.tensor(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            dtype=torch.float64,
        ),
        # b_error = b_primary - b_embedded: the difference between the 5th-order
        # and 4th-order method weights, used for error estimation
        b_error=torch.tensor(
            [
                35 / 384 - 1951 / 21600,
                0,
                500 / 1113 - 22642 / 50085,
                125 / 192 - 451 / 720,
                -2187 / 6784 - -12231 / 42400,
                11 / 84 - 649 / 6300,
                -1.0 / 60.0,
            ],
            dtype=torch.float64,
        ),
    ),
)


# Registry of uniform-sampling methods, keyed by name.
# These are singleton instances since their tableaux are constant.
UNIFORM_METHODS = {
    "adaptive_heun": _ADAPTIVE_HEUN,
    "fehlberg2": _FEHLBERG2,
    "bosh3": _BOGACKI_SHAMPINE,
    "dopri5": _DORMAND_PRINCE_SHAMPINE,
}


# =============================================================================
# Variable Sampling Adaptive Methods
# =============================================================================
# These methods compute tableau b weights dynamically based on the actual
# positions of quadrature points within each step. This allows points to
# be at arbitrary (non-fixed) positions, which is useful when the adaptive
# mesh refinement inserts new points between existing evaluations.


class _VariableSubclass(ABC):
    """
    Base class for variable-sampling RK methods.

    Unlike uniform methods where b weights are fixed, variable methods
    compute b weights from the actual quadrature point positions via
    ``tableau_b(c)``.

    Attributes:
        device: The device tensors are stored on.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = device

    @abstractmethod
    def to_device(self, device: str | torch.device) -> None:
        """Move method tensors to the specified device."""

    @abstractmethod
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert method tensors to the specified dtype."""


class _VARIABLE_SECOND_ORDER(_VariableSubclass):
    """
    Variable-sampling 2nd-order method (adaptive Heun variant).

    For 2nd-order methods, the b weights happen to be independent of the
    quadrature point positions, so this simply returns the fixed adaptive
    Heun tableau b values regardless of the input c.

    Attributes:
        order: Convergence order (2).
        n_tableau_c: Number of quadrature points per step (2).
    """

    order = 2
    n_tableau_c = 2

    def __init__(self, device: str | torch.device | None = None) -> None:
        super().__init__(device)
        self.device = device
        self.tableau = _ADAPTIVE_HEUN.tableau

    @override
    def to_device(self, device: str | torch.device) -> None:
        """Move tableau tensors to the specified device."""
        self.tableau.to_device(device)

    @override
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tableau tensors to the specified dtype."""
        self.tableau.to_dtype(dtype)

    def tableau_b(self, _c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return b weights for the given quadrature positions.

        For 2nd-order, the weights are constant regardless of c positions.

        Args:
            c: Normalized quadrature positions within each step.
                Shape: [N, C, T] where values are in [0, 1].

        Returns:
            Tuple of (b, b_error) tensors. Shape: [1, C].
        """
        b = self.tableau.b
        b_error = self.tableau.b_error
        return b, b_error


class _VARIABLE_THIRD_ORDER(_VariableSubclass):
    """
    Variable-sampling 3rd-order method (generic Sanderse-Veldman).

    Computes tableau b weights dynamically based on the actual positions
    of the quadrature points within each step. This is the key difference
    from uniform methods: the same set of evaluations at different positions
    can be correctly weighted.

    The method uses the generic 3rd-order formula from Sanderse and Veldman,
    where the b weights are functions of the middle node position 'a':

        b0(a) = 1/2 - 1/(6a)
        ba(a) = 1/(6a(1-a))
        b1(a) = (2-3a)/(6(1-a))

    The error is estimated by comparing against a reference set of weights
    (b_delta = [0.5, 0.0, 0.5], the trapezoidal rule).

    Attributes:
        order: Convergence order (3).
        n_tableau_c: Number of quadrature points per step (3).
        b_delta: Reference weights for error estimation. Shape: [1, 3].
    """

    order = 3
    n_tableau_c = 3

    def __init__(self, device: str | torch.device | None = None) -> None:
        super().__init__(device)
        self.device = device
        # Reference weights (trapezoidal rule) used for error estimation
        self.b_delta = torch.tensor(
            [[0.5, 0.0, 0.5]], dtype=torch.float64, device=self.device
        )

    @override
    def to_device(self, device: str | torch.device) -> None:
        """Move tensors to the specified device."""
        self.b_delta = self.b_delta.to(device)

    @override
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tensors to the specified dtype."""
        self.b_delta = self.b_delta.to(dtype)

    def _b0(self, a: torch.Tensor) -> torch.Tensor:
        """Weight for the left endpoint (c=0). Shape follows input a."""
        return 0.5 - 1.0 / (6 * a)

    def _ba(self, a: torch.Tensor) -> torch.Tensor:
        """Weight for the interior point (c=a). Shape follows input a."""
        return 1.0 / (6 * a * (1 - a))

    def _b1(self, a: torch.Tensor) -> torch.Tensor:
        """Weight for the right endpoint (c=1). Shape follows input a."""
        return (2.0 - 3 * a) / (6 * (1.0 - a))

    def tableau_b(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute b weights dynamically from actual quadrature positions.

        Uses the generic 3rd-order Sanderse-Veldman formula. The middle
        node position 'a' (where 0 < a < 1) determines all three weights.

        Butcher tableau (degree P-1 embedded pair):

            c |      b
            ------------------
            0 | 1/2 - 1/(6a)
            a | 1/(6a(1-a))
            1 | (2-3a)/(6(1-a))

        Args:
            c: Normalized quadrature positions within each step. The middle
                node position a = c[:,1,0] determines the weights.
                Shape: [N, C, T] where C=3, values in [0, 1].

        Returns:
            Tuple of (b, b_error) where:
                - b: Primary method weights. Shape: [N, C].
                - b_error: Difference from reference weights for error
                  estimation. Shape: [N, C].
        """
        # Extract the middle node position 'a' for each step
        a = c[:, 1, 0]
        # Compute the three b weights as functions of 'a'
        b = torch.stack([self._b0(a), self._ba(a), self._b1(a)]).transpose(0, 1)
        # Error is the difference between the computed weights and the
        # reference trapezoidal rule weights
        b_error = b - self.b_delta

        return b, b_error


# Registry of variable-sampling methods, keyed by name.
# These are classes (not instances) since they need per-device initialization.
VARIABLE_METHODS = {
    "adaptive_heun": _VARIABLE_SECOND_ORDER,
    "generic3": _VARIABLE_THIRD_ORDER,
}


def _get_method(
    sampling_type: steps,
    method_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> MethodClass | _VariableSubclass:
    """
    Retrieve and initialize an integration method by name and sampling type.

    For uniform methods, returns the singleton MethodClass instance (shared
    across calls). For variable methods, creates a new instance since each
    needs its own device-specific state.

    Args:
        sampling_type: Whether to use uniform or variable sampling.
        method_name: Name of the method (e.g. 'dopri5', 'generic3').
        device: Device to place the method's tensors on.
        dtype: Floating-point dtype for the method's tensors.

    Returns:
        An initialized method object with tensors on the correct device/dtype.
    """
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        method = UNIFORM_METHODS[method_name]
    else:
        method = VARIABLE_METHODS[method_name](device)

    method.to_device(device)
    method.to_dtype(dtype)
    return method
