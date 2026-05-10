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

import numpy as np
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
        """Convert all tableau tensors to the specified dtype.

        Mutates in place. Use ``clone()`` first if you want to leave
        the original tensors at their pre-conversion precision (which
        is irreversibly lossy when going to lower-precision dtypes).
        """
        self.c = self.c.to(dtype)
        self.b = self.b.to(dtype)
        self.b_error = self.b_error.to(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move all tableau tensors to the specified device."""
        self.c = self.c.to(device)
        self.b = self.b.to(device)
        self.b_error = self.b_error.to(device)

    def clone(self) -> _Tableau:
        """Return a deep copy of this tableau.

        Used by ``_get_method`` so that each solver instance gets its
        own tableau and dtype/device mutations on one solver do not
        propagate to other solvers via shared singletons.
        """
        return _Tableau(
            c=self.c.clone(),
            b=self.b.clone(),
            b_error=self.b_error.clone(),
        )


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
        """Initialize with the method's convergence order and Butcher tableau."""
        self.order = order
        self.tableau = tableau

    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert tableau to the specified dtype."""
        self.tableau.to_dtype(dtype)

    def to_device(self, device: str | torch.device) -> None:
        """Move tableau to the specified device."""
        self.tableau.to_device(device)

    def clone(self) -> MethodClass:
        """Return a copy with a deep-cloned tableau but the same order."""
        return MethodClass(order=self.order, tableau=self.tableau.clone())


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


# =============================================================================
# Gauss-Kronrod Quadrature Pairs
# =============================================================================
# Embedded pairs (G_n, K_{2n+1}): an n-point Gauss-Legendre rule plus an
# (n+1)-point Kronrod extension. K_{2n+1} integrates polynomials of degree
# 3n+1 exactly; the pair's error indicator is K_{2n+1} - G_n.
#
# Nodes / weights tabulated to ~30 decimal digits, sourced from QUADPACK
# (Piessens et al. 1983) and Laurie 1997 (Math. Comp. 66 #213). Truncated
# to float64 precision below. Polynomial-exactness tests in
# tests/test_exactness.py validate the values.


def _build_gauss_kronrod_tableau(
    xgk_half_with_zero: list[float],
    wgk_half_with_zero: list[float],
    wg_at_g_pos: list[float],
    g_pos_indices_in_half: list[int],
    wg_at_center: float = 0.0,
) -> _Tableau:
    """Construct a Gauss-Kronrod _Tableau on [0, 1].

    Inputs are tabulated for the canonical interval (-1, 1) with values
    given for the positive half plus the center node (which is the last
    entry of ``xgk_half_with_zero``). The full K_{2n+1} arrays are built
    by mirroring the positive-half about 0.

    Args:
        xgk_half_with_zero: K_{2n+1} node positions in (-1, 1), positive
            half (descending magnitude) followed by the center 0.0.
            Length n + 1. For K21 this has 11 entries: 10 positive +
            center.
        wgk_half_with_zero: K_{2n+1} weights at the corresponding nodes.
            Length n + 1.
        wg_at_g_pos: G_n weights at the G_n positive-half nodes.
            Length n // 2 (Kronrod's theorem: G_n has n nodes symmetric
            about 0; for even n there are n // 2 distinct positive
            magnitudes; for odd n there are (n - 1) // 2 distinct
            positive magnitudes plus the center).
        g_pos_indices_in_half: indices into the positive half of
            ``xgk_half_with_zero`` where G_n nodes coincide with K
            extension nodes. Length matches ``wg_at_g_pos``.
        wg_at_center: G_n weight at the center node 0.0. Use 0.0 if
            G_n has no center node (even n: G10, G14, ...). Use the
            published center weight for odd n: G7, G15, ...

    Returns:
        _Tableau with c (K nodes mapped to [0, 1]), b (K weights, the
        high-order estimate), and b_error (K - G_extended, where
        G_extended has 0 at K-only nodes — the standard error
        indicator for an embedded G-K pair).
    """
    xgk_half = np.array(xgk_half_with_zero, dtype=np.float64)  # [n+1]
    wgk_half = np.array(wgk_half_with_zero, dtype=np.float64)  # [n+1]
    wg_g = np.array(wg_at_g_pos, dtype=np.float64)

    # Build full K_{2n+1} arrays in (-1, 1), ascending order:
    #   [-x_n, ..., -x_1, 0, x_1, ..., x_n]
    # The positive half is xgk_half[:-1] in DESCENDING magnitude order
    # (largest first); reversed, it is in ascending order.
    pos_x = xgk_half[:-1]  # descending magnitudes
    pos_w = wgk_half[:-1]
    center_w = wgk_half[-1]

    # Negative side (ascending nodes from most negative): -pos_x in original
    # order is [-largest, -second_largest, ..., -smallest], which is exactly
    # ascending from most negative to least negative.
    xgk_full = np.concatenate([-pos_x, [0.0], pos_x[::-1]])
    wgk_full = np.concatenate([pos_w, [center_w], pos_w[::-1]])

    # Build G_n weights extended to the K_{2n+1} grid (zero at K-only
    # nodes). For even n, G_n has no center node (wg_at_center = 0).
    # For odd n (G7, G15, ...), the center node IS a G_n node and
    # wg_at_center is its non-zero weight.
    wg_extended_pos_half = np.zeros_like(pos_w)
    for i, gi in enumerate(g_pos_indices_in_half):
        wg_extended_pos_half[gi] = wg_g[i]
    wg_extended_full = np.concatenate(
        [wg_extended_pos_half, [wg_at_center], wg_extended_pos_half[::-1]]
    )

    # Map (-1, 1) -> [0, 1]: c = (x + 1) / 2, w = w / 2 (Jacobian).
    c_unit = (xgk_full + 1.0) / 2.0
    b_unit = wgk_full / 2.0
    b_g_unit = wg_extended_full / 2.0

    # The parallel solver's _RK_integral computes the per-panel step size
    # as ``h = t[:, -1] - t[:, 0]`` and assumes ``c[0] == 0`` and
    # ``c[-1] == 1``. Gauss-Kronrod nodes are interior (no endpoints) so
    # without padding ``h`` would equal 0.99566... instead of 1.0 and the
    # integral would be off by that factor. Standard fix: pad with two
    # zero-weight nodes at 0 and 1; they're evaluated but contribute
    # nothing to the integral or error estimate. The cost is 2 wasted
    # f evaluations per panel — negligible for an N+2 = 23-node rule.
    c_padded = np.concatenate([[0.0], c_unit, [1.0]])
    b_padded = np.concatenate([[0.0], b_unit, [0.0]])
    b_g_padded = np.concatenate([[0.0], b_g_unit, [0.0]])
    b_error = b_padded - b_g_padded

    return _Tableau(
        c=torch.from_numpy(c_padded).to(torch.float64),
        b=torch.from_numpy(b_padded).to(torch.float64),
        b_error=torch.from_numpy(b_error).to(torch.float64),
    )


# G10-K21 (Gauss-Kronrod 21-point). K21 polynomial exactness = 31.
# Source: QUADPACK / Laurie 1997. Positive half-axis nodes followed by
# the center node 0.0.
_GK21_XGK_HALF = [
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.0,
]
_GK21_WGK_HALF = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244582,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
]
# G10 weights at G10 positive nodes. G10 nodes coincide with K21 nodes
# at indices 1, 3, 5, 7, 9 of the positive half (the "G" rows in the
# Patterson-style Kronrod construction).
_GK21_WG = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
]
_GK21_G_INDICES = [1, 3, 5, 7, 9]

_GAUSS_KRONROD_21 = MethodClass(
    order=32,  # K21 polynomial exactness 31, global convergence rate 32
    tableau=_build_gauss_kronrod_tableau(
        _GK21_XGK_HALF,
        _GK21_WGK_HALF,
        _GK21_WG,
        _GK21_G_INDICES,
        wg_at_center=0.0,  # G10 (even n) has no center node
    ),
)


# G7-K15 (Gauss-Kronrod 15-point). K15 polynomial exactness = 22.
# G7 (odd n=7) has a center node — note wg_at_center below.
_GK15_XGK_HALF = [
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,  # G7 node
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,  # G7 node
    0.586087235467691130294144838258730,
    0.405845151377397166906606412076961,  # G7 node
    0.207784955007898467600689403773245,
    0.0,  # center, also a G7 node
]
_GK15_WGK_HALF = [
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
]
# G7 weights at the three positive G7 nodes (K15 indices 1, 3, 5).
_GK15_WG = [
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
]
_GK15_G_INDICES = [1, 3, 5]
# G7 weight at the center node (G7 has odd n, so it includes 0.0).
_GK15_WG_CENTER = 0.417959183673469387755102040816327

_GAUSS_KRONROD_15 = MethodClass(
    order=23,  # K15 polynomial exactness 22, global convergence rate 23
    tableau=_build_gauss_kronrod_tableau(
        _GK15_XGK_HALF,
        _GK15_WGK_HALF,
        _GK15_WG,
        _GK15_G_INDICES,
        wg_at_center=_GK15_WG_CENTER,
    ),
)


# G15-K31 (Gauss-Kronrod 31-point). K31 polynomial exactness = 46.
# G15 (odd n=15) has a center node.
_GK31_XGK_HALF = [
    0.998002298693397060285172840152271,
    0.987992518020485428489565718586613,  # G15 node
    0.967739075679139134257347978784337,
    0.937273392400705904307758947710209,  # G15 node
    0.897264532344081900882509656454496,
    0.848206583410427216200648320774217,  # G15 node
    0.790418501442465932967649294817947,
    0.724417731360170047416186054613938,  # G15 node
    0.650996741297416970533735895313275,
    0.570972172608538847537226737253911,  # G15 node
    0.485081863640239680693655740232351,
    0.394151347077563369897207370981045,  # G15 node
    0.299180007153168812166780024266389,
    0.201194093997434522300628303394596,  # G15 node
    0.101142066918717499027074231447392,
    0.0,  # center, also a G15 node
]
_GK31_WGK_HALF = [
    0.005377479872923348987792051430128,
    0.015007947329316122538374763075807,
    0.025460847326715320186874001019653,
    0.035346360791375846222037948478360,
    0.044589751324764876608227299373280,
    0.053481524690928087265343147239430,
    0.062009567800670640285139230960803,
    0.069854121318728258709520077099147,
    0.076849680757720378894432777482659,
    0.083080502823133021038289247286104,
    0.088564443056211770647275443693774,
    0.093126598170825321225486872747346,
    0.096642726983623678505179907627589,
    0.099173598721791959332393173484603,
    0.100769845523875595044946662617570,
    0.101330007014791549017374792767493,
]
# G15 weights at positive G15 nodes (K31 half-indices 1, 3, 5, 7, 9, 11, 13).
_GK31_WG = [
    0.030753241996117268354628393577204,
    0.070366047488108124709267416450667,
    0.107159220467171935011869546685869,
    0.139570677926154314447804794511028,
    0.166269205816993933553200860481209,
    0.186161000015562211026800561866423,
    0.198431485327111576456118326443839,
]
_GK31_G_INDICES = [1, 3, 5, 7, 9, 11, 13]
_GK31_WG_CENTER = 0.202578241925561272880620199967519

_GAUSS_KRONROD_31 = MethodClass(
    order=47,  # K31 polynomial exactness 46, global convergence rate 47
    tableau=_build_gauss_kronrod_tableau(
        _GK31_XGK_HALF,
        _GK31_WGK_HALF,
        _GK31_WG,
        _GK31_G_INDICES,
        wg_at_center=_GK31_WG_CENTER,
    ),
)


# =============================================================================
# Clenshaw-Curtis Quadrature Pairs
# =============================================================================
# Clenshaw-Curtis CC_n integrates polynomials of degree n exactly using
# n+1 Chebyshev nodes x_k = cos(k*pi/n) on (-1, 1). Two key properties:
#
#  1. CC nodes are NESTED. The CC_(n/2) grid is exactly the every-other
#     subset of the CC_n grid. This makes CC the natural method for
#     adaptive doubling refinement.
#  2. CC nodes naturally include the panel endpoints (x_0 = 1, x_n = -1
#     map to c = 1 and c = 0). No endpoint padding is needed (unlike
#     Gauss-Kronrod).
#
# We provide cc17 (n=16), cc33 (n=32), cc65 (n=64). The error indicator
# pairs CC_n with CC_(n/2): b_error = CC_n_weights - CC_(n/2)_extended,
# where the extension places 0 at CC_n-only nodes.
#
# Reference: Trefethen 2008 SIAM Review (Is Gauss quadrature better than
# Clenshaw-Curtis?); Waldvogel 2006 BIT (Fast computation of CC weights).


def _clenshaw_curtis_weights_unit(n: int) -> np.ndarray:
    """Compute Clenshaw-Curtis weights on (-1, 1) for n+1 Chebyshev nodes.

    Nodes are x_k = cos(k * pi / n) for k = 0..n; the rule integrates
    polynomials of degree at most n exactly.

    Returns weights of length n+1, indexed as the nodes (i.e. weight at
    cos(k*pi/n) is at position k).
    """
    if n == 0:
        return np.array([2.0])
    # Trefethen 2008 / Waldvogel 2006 cosine-series formula.
    c_aux = np.zeros(n + 1)
    c_aux[0] = 2.0
    for k in range(2, n + 1, 2):
        c_aux[k] = 2.0 / (1.0 - k * k)
    w = np.zeros(n + 1)
    pi_n = np.pi / n
    for k in range(n + 1):
        s = c_aux[0] / 2  # j=0 term, halved
        for j in range(1, n):
            s += c_aux[j] * np.cos(j * k * pi_n)
        s += c_aux[n] * np.cos(k * np.pi) / 2  # j=n term, halved
        w[k] = (2.0 / n) * s
    # Halve the endpoint weights (boundary correction)
    w[0] /= 2.0
    w[-1] /= 2.0
    return w


def _build_clenshaw_curtis_tableau(n: int) -> _Tableau:
    """Construct a Clenshaw-Curtis _Tableau on [0, 1] for n+1 nodes.

    The embedded error indicator pairs CC_n with CC_(n/2). The nodes of
    CC_(n/2) are exactly every other node of CC_n (CC nesting), so the
    extension to the CC_n grid places the CC_(n/2) weight at shared
    positions and 0 at the CC_n-only positions.
    """
    assert n >= 2, "CC error pair requires n >= 2"
    assert n % 2 == 0, "CC error pair requires even n for the doubling embed"

    # CC weights on (-1, 1), indexed by k=0..n where x_k = cos(k*pi/n).
    w_cc_high = _clenshaw_curtis_weights_unit(n)  # [n+1]
    w_cc_low = _clenshaw_curtis_weights_unit(n // 2)  # [n//2 + 1]

    # CC_(n/2) nodes coincide with CC_n nodes at indices 0, 2, 4, ..., n.
    # Build the extended CC_(n/2) weights on the CC_n grid.
    w_cc_low_extended = np.zeros(n + 1)
    w_cc_low_extended[0::2] = w_cc_low

    # Map (-1, 1) -> [0, 1]. The natural CC indexing has x_0 = +1,
    # x_n = -1 (descending), so to get c[0] = 0 and c[-1] = 1 we
    # reverse the index: c[k] = (cos((n-k)*pi/n) + 1) / 2.
    nodes_unit = np.array(
        [(np.cos((n - k) * np.pi / n) + 1.0) / 2.0 for k in range(n + 1)],
        dtype=np.float64,
    )
    # Force exact endpoint values to avoid sub-eps drift.
    nodes_unit[0] = 0.0
    nodes_unit[-1] = 1.0

    # Reverse the weight arrays to match the reversed node order.
    b_high = (w_cc_high[::-1] / 2.0).astype(np.float64)
    b_low_ext = (w_cc_low_extended[::-1] / 2.0).astype(np.float64)
    b_error = b_high - b_low_ext

    return _Tableau(
        c=torch.from_numpy(nodes_unit).to(torch.float64),
        b=torch.from_numpy(b_high).to(torch.float64),
        b_error=torch.from_numpy(b_error).to(torch.float64),
    )


# CC_16 (17 nodes): polynomial exactness 16, error vs CC_8 (9 nodes, exactness 8)
_CLENSHAW_CURTIS_17 = MethodClass(
    order=17,  # CC_n exactness = n; convergence rate ≈ n + 1 in RK convention
    tableau=_build_clenshaw_curtis_tableau(16),
)
# CC_32 (33 nodes): polynomial exactness 32, error vs CC_16 (17 nodes, exactness 16)
_CLENSHAW_CURTIS_33 = MethodClass(
    order=33,
    tableau=_build_clenshaw_curtis_tableau(32),
)
# CC_64 (65 nodes): polynomial exactness 64, error vs CC_32 (33 nodes, exactness 32)
_CLENSHAW_CURTIS_65 = MethodClass(
    order=65,
    tableau=_build_clenshaw_curtis_tableau(64),
)


# Registry of uniform-sampling methods, keyed by name.
# These are singleton instances since their tableaux are constant.
UNIFORM_METHODS = {
    "adaptive_heun": _ADAPTIVE_HEUN,
    "fehlberg2": _FEHLBERG2,
    "bosh3": _BOGACKI_SHAMPINE,
    "dopri5": _DORMAND_PRINCE_SHAMPINE,
    "gk15": _GAUSS_KRONROD_15,
    "gk21": _GAUSS_KRONROD_21,
    "gk31": _GAUSS_KRONROD_31,
    "cc17": _CLENSHAW_CURTIS_17,
    "cc33": _CLENSHAW_CURTIS_33,
    "cc65": _CLENSHAW_CURTIS_65,
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
        """Initialize with an optional device for tensor placement."""
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
        """Initialize the 2nd-order variable method using adaptive Heun's tableau."""
        super().__init__(device)
        self.device = device
        self.tableau = _ADAPTIVE_HEUN.tableau

    def to_device(self, device: str | torch.device) -> None:
        """Move tableau tensors to the specified device."""
        self.tableau.to_device(device)

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


class _Interpolatory3Variable(_VariableSubclass):
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
        """Initialize the 3rd-order variable method with Sanderse-Veldman weights."""
        super().__init__(device)
        self.device = device
        # Reference weights (trapezoidal rule) used for error estimation
        self.b_delta = torch.tensor(
            [[0.5, 0.0, 0.5]], dtype=torch.float64, device=self.device
        )

    def to_device(self, device: str | torch.device) -> None:
        """Move tensors to the specified device."""
        self.b_delta = self.b_delta.to(device)

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
    "interpolatory3_variable": _Interpolatory3Variable,
}


def _get_method(
    sampling_type: steps,
    method_name: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> MethodClass | _VariableSubclass:
    """
    Retrieve and initialize an integration method by name and sampling type.

    Each solver gets its OWN copy of the method's tableau. The
    ``UNIFORM_METHODS`` dict holds canonical singletons at float64,
    and we ``clone()`` them here so that downstream dtype/device
    mutations stay isolated to one solver instance. Without this
    cloning, two solvers running concurrently — or sequentially with
    different dtypes — would corrupt each other's tableau values
    (most notably, lower-precision conversions of the singleton are
    irreversible).

    Variable methods are constructed fresh per-solver too (they
    already manage their own per-instance state).

    Args:
        sampling_type: Whether to use uniform or variable sampling.
        method_name: Name of the method (e.g. 'dopri5',
            'interpolatory3_variable').
        device: Device to place the method's tensors on.
        dtype: Floating-point dtype for the method's tensors.

    Returns:
        An initialized method object with tensors on the correct
        device/dtype, independent of any other solver instance.
    """
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        # Clone the canonical float64 singleton so dtype/device
        # mutations below don't affect other solvers.
        method = UNIFORM_METHODS[method_name].clone()
    else:
        method = VARIABLE_METHODS[method_name](device)

    method.to_device(device)
    method.to_dtype(dtype)
    return method
