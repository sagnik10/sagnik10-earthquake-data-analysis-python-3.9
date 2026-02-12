"""
tanh_sinh.py
Minimal Takahashi–Mori (tanh-sinh) quadrature utilities
-------------------------------------------------------

Functions
---------
tanh_sinh_rule(order)      → list[(x, w)]
integrate(f, xi, xj, order) → float
"""

from __future__ import annotations
import math
import sys
from typing import Callable, List, Tuple
import math
from typing import Callable, List, Tuple

_PI_OVER_2 = math.pi / 2.0


def tanh_sinh_rule(order: int) -> List[Tuple[float, float]]:
    """
    Stationary tanh–sinh abscissae/weights on [-1, 1].

    Parameters
    ----------
    order : int
        Number of *positive* nodes (K).  Total points = 2*K + 1.

    Returns
    -------
    list of (x, w) pairs, symmetric around 0.
    """
    h = 1.0  # mesh spacing (fixed ⇔ fixed nodes)
    nodes: List[Tuple[float, float]] = []

    for k in range(-order, order + 1):
        t = k * h
        sh = math.sinh(t)
        ch = math.cosh(t)
        u = math.tanh(_PI_OVER_2 * sh)                       # x-coord in [-1,1]
        w = _PI_OVER_2 * ch / math.cosh(_PI_OVER_2 * sh) ** 2 * h
        w *= 0.5                                             # Jacobian of map
        nodes.append((u, w))

    return nodes


def integrate(
    f: Callable[[float], float],
    xi: float,
    xj: float,
    order: int = 10,
) -> float:
    """
    Integrate `f` over [xi, xj] with a fixed tanh–sinh rule.

    Parameters
    ----------
    f     : callable        integrand
    xi,xj : float           integration limits
    order : int, default 10 number of positive nodes (≥1)

    Returns
    -------
    float  approximation of ∫_xi^xj f(x) dx
    """
    if xi == xj:
        return 0.0

    if xi > xj:                                  # orient interval
        return -integrate(f, xj, xi, order)

    span_half = 0.5 * (xj - xi)                  # affine map constants
    mid       = 0.5 * (xj + xi)

    total = 0.0
    for u, w in tanh_sinh_rule(order):
        x = mid + span_half * u                  # map to [xi,xj]
        total += w * f(x)

    return total * span_half                     # final Jacobian factor



_PI_OVER_2 = math.pi / 2.0
_EPS = 10 * sys.float_info.epsilon


# --------------------------------------------------------------------------- #
#  1. Stationary rule (now with adjustable step h)
# --------------------------------------------------------------------------- #
def tanh_sinh_rule_h(order: int, h: float = 1.0) -> List[Tuple[float, float]]:
    """
    Return 2*order+1 symmetric nodes/weights for the tanh–sinh rule
    on [-1,1] using trapezoidal step size `h` in the t-domain.

    Parameters
    ----------
    order : int
        Number of *positive* nodes (K).  Total nodes = 2*K + 1.
    h     : float, optional
        Mesh spacing in the auxiliary variable t (default 1.0).

    Returns
    -------
    list[(x, w)]  where x∈[-1,1], w>0
    """
    nodes: List[Tuple[float, float]] = []

    for k in range(-order, order + 1):
        t = k * h
        sh = math.sinh(t)
        ch = math.cosh(t)
        u = math.tanh(_PI_OVER_2 * sh)                             # map to [-1,1]
        w = _PI_OVER_2 * ch / math.cosh(_PI_OVER_2 * sh) ** 2 * h  # dx/dt * h
        w *= 0.5                                                   # Jacobian [-∞,∞]→[-1,1]
        nodes.append((u, w))

    return nodes


# --------------------------------------------------------------------------- #
#  2. Adaptive integrators
# --------------------------------------------------------------------------- #

def integrate_a1(
    f: Callable[[float], float],
    xi: float,
    xj: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 20,
    max_terms: int = 1_000_000,
) -> float:
    """
    Tanh–sinh (double exponential) quadrature by Takahashi & Mori.

    Parameters
    ----------
    f : callable
        Scalar function to integrate.
    xi, xj : float
        Integration limits (finite, may be in either order).
    tol : float, optional
        Absolute tolerance for the integral (default 1e-12).
    max_iter : int, optional
        Maximum mesh-refinement steps (each halves the step size).
    max_terms : int, optional
        Safety cap on the number of positive mesh points evaluated per level.

    Returns
    -------
    float
        Approximation of ∫_xi^xj f(x) dx.

    Notes
    -----
    *   Handles integrable algebraic / logarithmic endpoint singularities well.
    *   If the integrand is extremely oscillatory or decays slowly, increase
        `max_terms` or loosen `tol`.
    """
    if xi == xj:
        return 0.0
    if xi > xj:                     # ensure xi < xj
        return -integrate(f, xj, xi, tol=tol,
                          max_iter=max_iter, max_terms=max_terms)

    a, b = xi, xj
    span_half = 0.5 * (b - a)
    mid       = 0.5 * (b + a)
    const     = math.pi / 2.0
    # Machine-epsilon–scaled buffer used to avoid evaluating exactly at the
    # singular endpoints (if any):
    eps = 10 * sys.float_info.epsilon * max(abs(a), abs(b), 1.0)

    def _node_and_weight(t: float):
        """x(t) and dx/dt for the Takahashi–Mori transform."""
        sh = math.sinh(t)
        ch = math.cosh(t)
        u  = math.tanh(const * sh)
        x  = mid + span_half * u
        dxdt = span_half * const * ch / (math.cosh(const * sh) ** 2)
        return x, dxdt

    # --- refinement loop --------------------------------------------------
    h = 1.0             # initial mesh spacing
    prev_sum = None
    for _ in range(max_iter):
        total = 0.0
        k = 0
        while True:
            if k > max_terms:
                raise RuntimeError("tanh-sinh: term cap exceeded; "
                                   "integrand decays too slowly for given tol.")

            t_pos = k * h                      # positive abscissa
            x_p, w_p = _node_and_weight(t_pos)
            if abs(x_p - a) < eps or abs(x_p - b) < eps:
                fx_p = 0.0                     # endpoint → weight already tiny
            else:
                fx_p = f(x_p) * w_p

            if k == 0:                         # centre node (t = 0)
                total += fx_p
            else:
                x_n, w_n = _node_and_weight(-t_pos)
                if abs(x_n - a) < eps or abs(x_n - b) < eps:
                    fx_n = 0.0
                else:
                    fx_n = f(x_n) * w_n
                total += fx_p + fx_n

            # Truncation criterion: once both symmetric contributions
            # fall below tol, the remaining tail is negligible.
            if abs(fx_p) < tol and k > 0 and abs(fx_n) < tol:
                break
            k += 1

        total *= h          # trapezoidal step factor

        if prev_sum is not None and abs(total - prev_sum) < tol:
            return total    # converged
        prev_sum = total
        h *= 0.5            # refine mesh (h → h/2)

    raise RuntimeError("tanh-sinh quadrature failed to converge "
                       "within max_iter refinements")

def integrate_a2(
    f: Callable[[float], float],
    xi: float,
    xj: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 20,
    max_terms: int = 1_000_000,
) -> float:
    """
    Adaptive tanh–sinh quadrature using step-size refinement (h → h/2).

    Parameters
    ----------
    f         : callable          integrand
    xi, xj    : float             finite integration limits
    tol       : float, optional   absolute tolerance (default 1e-12)
    max_iter  : int, optional     maximum h-refinements
    max_terms : int, optional     safety cap on K per level

    Returns
    -------
    float  approximation of ∫_xi^xj f(x) dx
    """
    if xi == xj:
        return 0.0
    if xi > xj:                                   # orient interval
        return -integrate(f, xj, xi, tol=tol,
                          max_iter=max_iter, max_terms=max_terms)

    a, b = xi, xj
    span_half = 0.5 * (b - a)
    mid = 0.5 * (b + a)

    # --- adaptive refinement loop ----------------------------------------- #
    h = 1.0
    prev_val = None

    for _ in range(max_iter):
        total = 0.0
        k = 0

        while True:
            if k > max_terms:
                raise RuntimeError("tanh-sinh: exceeded max_terms (tail too wide)")

            # Fetch nodes/weights for ±k (and k=0) via the stationary rule
            nodes = tanh_sinh_rule_h(k, h)

            if k == 0:
                u0, w0 = nodes[0]
                x0 = mid + span_half * u0
                contrib = w0 * f(x0)
                total += contrib
                # no tail check for k=0
            else:
                u_neg, w_neg = nodes[0]   # -k
                u_pos, w_pos = nodes[-1]  # +k

                x_neg = mid + span_half * u_neg
                x_pos = mid + span_half * u_pos

                contrib_neg = w_neg * f(x_neg)
                contrib_pos = w_pos * f(x_pos)

                total += contrib_neg + contrib_pos

                # truncation criterion for the t-domain tails
                if abs(contrib_neg) < tol and abs(contrib_pos) < tol:
                    break
            k += 1

        value = total * span_half          # map [-1,1] → [xi,xj]

        if prev_val is not None and abs(value - prev_val) < tol:
            return value                   # converged
        prev_val = value
        h *= 0.5                           # refine step size

    raise RuntimeError("tanh-sinh: failed to converge within max_iter")


if __name__ == "__main__":
    import math
    I1 = integrate_a2(math.sin, 0.0, math.pi)        # → 2.0
    I2 = integrate_a2(lambda x: 1 / math.sqrt(x), 0.0, 1.0, tol=1e-10)
    print(I1, I2)

