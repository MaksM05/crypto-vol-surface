"""Raw SVI per-expiry smile: model, fitter, and Gatheral–Durrleman no-butterfly check.

Pure numpy/scipy. No I/O. Reads nothing, writes nothing.

Parameterisation (Gatheral 2004)
--------------------------------

For log-moneyness ``k = ln(K/F)``, total variance ``w(k) = σ_BS² · T`` is

    w(k) = a + b · ( ρ·(k − m) + √((k − m)² + σ²) )

with ``a ≥ 0``, ``b ≥ 0``, ``ρ ∈ [-1, 1]``, ``m ∈ ℝ``, ``σ > 0``. The
parameter ``σ`` here is the SVI vertex curvature, **not** volatility — the
naming is unfortunate but conventional.

Closed-form derivatives used by the no-arb check (with ``u = k − m``,
``r = √(u² + σ²)``):

    w'(k)  = b · (ρ + u/r)
    w''(k) = b · σ² / r³

No-butterfly check (Gatheral–Durrleman, Roper 2010)
---------------------------------------------------

Define

    g(k) = (1 − k·w'(k)/(2·w(k)))²
           − (w'(k))² / 4 · (1/w(k) + 1/4)
           + w''(k) / 2

The smile is butterfly-arbitrage-free iff ``g(k) ≥ 0 ∀ k``. This is the
density-positivity / call-price-convexity condition; it is strictly
stronger than Lee's asymptotic wing bounds (``b·(1±ρ) ≤ 2``), so we check
only ``g`` — one source of truth, and a violation of Lee's bound shows up
as ``g < 0`` in the wings.

Fit
---

Unweighted least-squares on total variance ``w`` (not ``σ_BS``), via
``scipy.optimize.least_squares`` with box bounds. Single deterministic
seed — multistart is a tool we'll reach for if real fits become unstable.
Refuse to fit fewer than 6 points (5 parameters + ≥ 1 degree of freedom):
return ``SVIFitResult`` with ``success=False`` instead of raising.

The liquidity filter (``OI > 10``, ``spread/mid < 0.05``, OTM-only,
``|k| < 3``) is the orchestrator's job, not this module's — by the time
``k`` and ``w_market`` arrive here they're already clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

type FloatArray = npt.NDArray[np.float64]

_MIN_POINTS = 6


@dataclass(frozen=True, slots=True)
class SVIParams:
    """Raw SVI parameters. ``sigma`` is vertex curvature, NOT volatility."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass(frozen=True, slots=True)
class SVIFitResult:
    """Outcome of a single-smile SVI fit.

    Attributes
    ----------
    params
        Fitted parameters, or ``None`` if the fit could not run / converge.
    rmse
        Root-mean-square residual on total variance ``w``. ``nan`` when no
        fit was produced.
    n_points
        Number of liquid points that entered the fit.
    success
        Did the optimiser report convergence?
    is_butterfly_free
        Did the post-fit Durrleman check pass? Meaningful only if
        ``success`` is true.
    message
        Short diagnostic — "ok", "need >= 6 points, got N", optimiser
        message, or "fitted but butterfly arb".
    """

    params: SVIParams | None
    rmse: float
    n_points: int
    success: bool
    is_butterfly_free: bool
    message: str


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def svi_total_variance(k: npt.ArrayLike, params: SVIParams) -> FloatArray:
    """Evaluate the SVI total-variance function ``w(k)``."""
    k_arr = np.asarray(k, dtype=np.float64)
    u = k_arr - params.m
    r = np.sqrt(u * u + params.sigma * params.sigma)
    return np.asarray(params.a + params.b * (params.rho * u + r), dtype=np.float64)


def _svi_derivatives(k: FloatArray, params: SVIParams) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(w, w', w'')`` at each ``k``. Internal — assumes ``k`` already array."""
    u = k - params.m
    r = np.sqrt(u * u + params.sigma * params.sigma)
    w = params.a + params.b * (params.rho * u + r)
    w_prime = params.b * (params.rho + u / r)
    w_pp = params.b * params.sigma * params.sigma / (r * r * r)
    return w, w_prime, w_pp


def durrleman_g(k: npt.ArrayLike, params: SVIParams) -> FloatArray:
    """Gatheral–Durrleman ``g(k)``. Non-negative everywhere iff no butterfly arb."""
    k_arr = np.asarray(k, dtype=np.float64)
    w, w_prime, w_pp = _svi_derivatives(k_arr, params)
    term1 = (1.0 - k_arr * w_prime / (2.0 * w)) ** 2
    term2 = -0.25 * w_prime * w_prime * (1.0 / w + 0.25)
    term3 = 0.5 * w_pp
    return np.asarray(term1 + term2 + term3, dtype=np.float64)


# ---------------------------------------------------------------------------
# No-butterfly arbitrage check
# ---------------------------------------------------------------------------


def is_butterfly_arb_free(
    params: SVIParams,
    *,
    k_min: float = -3.0,
    k_max: float = 3.0,
    n_grid: int = 2000,
    tol: float = 1e-10,
) -> bool:
    """Return True iff ``g(k) ≥ -tol`` on a dense grid covering ``[k_min, k_max]``.

    Defaults check ``[-3, +3]`` with 2000 points — wider than typical Deribit
    BTC log-moneyness so wing-arbitrage is caught beyond the data range. Any
    non-finite ``g`` (caused by a degenerate ``w → 0``) returns ``False``.
    """
    if k_max <= k_min or n_grid < 10:
        raise ValueError("require k_max > k_min and n_grid >= 10")
    k_grid = np.linspace(k_min, k_max, n_grid, dtype=np.float64)
    g = durrleman_g(k_grid, params)
    if not bool(np.all(np.isfinite(g))):
        return False
    return bool(np.min(g) >= -tol)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------


def fit_svi(k: npt.ArrayLike, w_market: npt.ArrayLike) -> SVIFitResult:
    """Fit raw SVI to ``(k, w_market)`` points via bounded least-squares.

    Parameters
    ----------
    k
        Log-moneyness array, ``ln(K/F)``. Assumed already filtered (liquid,
        OTM-only, ``|k| < 3``) by the caller.
    w_market
        Observed total variance ``σ²·T`` at each ``k``.

    Returns
    -------
    SVIFitResult
        Always non-throwing for runtime conditions. The result's ``success``
        and ``is_butterfly_free`` flags carry the verdict.
    """
    k_arr = np.asarray(k, dtype=np.float64).reshape(-1)
    w_arr = np.asarray(w_market, dtype=np.float64).reshape(-1)
    if k_arr.shape != w_arr.shape:
        raise ValueError(f"shape mismatch: k {k_arr.shape} vs w {w_arr.shape}")
    n = int(k_arr.shape[0])
    if n < _MIN_POINTS:
        return SVIFitResult(
            params=None,
            rmse=math.nan,
            n_points=n,
            success=False,
            is_butterfly_free=False,
            message=f"need >= {_MIN_POINTS} points, got {n}",
        )

    x0 = _initial_guess(k_arr, w_arr)
    lo = np.array([0.0, 0.0, -1.0, -2.0, 1e-4])
    hi = np.array([np.inf, np.inf, 1.0, 2.0, 5.0])

    def residuals(theta: FloatArray) -> FloatArray:
        params = SVIParams(*theta.tolist())
        return svi_total_variance(k_arr, params) - w_arr

    try:
        result = least_squares(residuals, x0, bounds=(lo, hi), method="trf")
    except (ValueError, RuntimeError) as exc:
        return SVIFitResult(
            params=None,
            rmse=math.nan,
            n_points=n,
            success=False,
            is_butterfly_free=False,
            message=f"optimiser raised: {exc}",
        )

    if not bool(result.success):
        return SVIFitResult(
            params=None,
            rmse=math.nan,
            n_points=n,
            success=False,
            is_butterfly_free=False,
            message=f"optimiser failed: {result.message}",
        )

    params = SVIParams(*result.x.tolist())
    rmse = float(np.sqrt(np.mean(np.asarray(result.fun) ** 2)))
    arb_free = is_butterfly_arb_free(params)
    msg = "ok" if arb_free else "fitted but butterfly arb"
    return SVIFitResult(
        params=params,
        rmse=rmse,
        n_points=n,
        success=True,
        is_butterfly_free=arb_free,
        message=msg,
    )


def _initial_guess(k: FloatArray, w: FloatArray) -> FloatArray:
    """Deterministic SVI seed. Non-convex problem — seed matters but is not magical."""
    k_min, k_max = float(k.min()), float(k.max())
    w_min, w_max = float(w.min()), float(w.max())
    idx_min = int(np.argmin(w))
    m0 = float(k[idx_min])
    sigma0 = 0.1
    b0 = max((w_max - w_min) / max(0.5, k_max - k_min), 1e-3)
    # At the smile vertex w(m) ≈ a + b·σ, so seed a slightly below the empirical min.
    a0 = max(w_min - b0 * sigma0, 1e-6)
    rho0 = 0.0
    return np.array([a0, b0, rho0, m0, sigma0], dtype=np.float64)
