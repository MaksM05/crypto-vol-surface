"""SSVI (Surface SVI) joint calibration with both no-arb conditions.

Parameterisation (Gatheral–Jacquier 2014)
-----------------------------------------

For log-moneyness ``k = ln(K/F)`` and ATM total-variance backbone
``θ_T = w(0, T)``,

    w(k, θ) = (θ/2) · ( 1 + ρ·φ(θ)·k + √( (φ(θ)·k + ρ)² + 1 − ρ² ) )

with power-law shape function

    φ(θ) = η / θ^γ           η > 0,  γ ∈ [0, 1/2]

Three free parameters shared across the surface: ``(ρ, η, γ)``. The backbone
``{θ_T}`` is built by upstream code from per-expiry ATM total variance —
this module takes it as input.

Sanity at ATM: ``w(0, θ) = (θ/2) · (1 + √(ρ² + 1−ρ²)) = θ`` ✓.

No-butterfly conditions (Lemma 4.2)
-----------------------------------

For each ``θ`` in the backbone:

    (C1)   θ · φ(θ) · (1 + |ρ|)    <  4         strict
    (C2)   θ · φ(θ)² · (1 + |ρ|)   ≤  4         non-strict

These are enforced **AS HARD SLSQP INEQUALITY CONSTRAINTS DURING THE FIT** —
the optimiser stays inside the no-butterfly feasible region throughout
iteration. Butterfly arbitrage is excluded by construction.

No-calendar conditions (Theorem 4.1)
------------------------------------

For every fixed ``k``, ``w(k, T)`` must be non-decreasing in ``T``. For SSVI
this decomposes into:

1. **Backbone monotonicity**: ``θ_T`` non-decreasing in ``T``.
2. **No smile crossings**: ``w(k, θ_{T_{i+1}}) ≥ w(k, θ_{T_i})`` for every
   consecutive expiry pair and every ``k``.

These are enforced by **PARAMETER RESTRICTION AND PREPROCESSING**, then
**VERIFIED NUMERICALLY POST-FIT** — *not* hard constraints inside the
optimiser. Specifically: box-bound ``γ ∈ [0, 0.5]`` (the power-law range
where Gatheral–Jacquier's analytic conditions are well-behaved), refuse to
fit a non-monotone backbone in preprocessing, and after the fit walk a
dense ``k``-grid out to ``|k| = 3`` (wider than the data) to confirm no
smile crossing. **Do not describe the calendar dimension as "arb-free by
construction"** — it is "arb-free by restriction and post-hoc verification."
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

type FloatArray = npt.NDArray[np.float64]

_MIN_EXPIRIES = 3
_MIN_POINTS_PER_EXPIRY = 6
_C1_EPS = 1e-6  # strict-inequality slack for SLSQP


@dataclass(frozen=True, slots=True)
class SSVIParams:
    """The three SSVI free parameters."""

    rho: float
    eta: float
    gamma: float


@dataclass(frozen=True, slots=True)
class SSVIBackbone:
    """ATM total variance per expiry. Sorted by ``t_years`` ascending."""

    expiries: tuple[datetime, ...]
    t_years: tuple[float, ...]
    theta: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SSVIFitResult:
    """Outcome of an SSVI joint fit.

    ``is_butterfly_free`` is asserted by SLSQP feasibility at the optimum;
    ``is_calendar_free`` is asserted by the post-fit numerical check.
    """

    params: SSVIParams | None
    backbone: SSVIBackbone | None
    rmse: float
    n_points: int
    n_expiries: int
    success: bool
    is_butterfly_free: bool
    is_calendar_free: bool
    butterfly_message: str
    calendar_message: str
    message: str


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def ssvi_phi(theta: npt.ArrayLike, params: SSVIParams) -> FloatArray:
    """Power-law shape function ``φ(θ) = η / θ^γ``."""
    theta_arr = np.asarray(theta, dtype=np.float64)
    return np.asarray(params.eta / np.power(theta_arr, params.gamma), dtype=np.float64)


def ssvi_total_variance(
    k: npt.ArrayLike,
    theta: npt.ArrayLike,
    params: SSVIParams,
) -> FloatArray:
    """Evaluate the SSVI total variance ``w(k, θ)``.

    ``k`` and ``theta`` broadcast against each other — supply scalars, equal-length
    arrays (one ``θ`` per ``k``), or scalar/array combinations.
    """
    k_arr = np.asarray(k, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    rho = params.rho
    phi = ssvi_phi(theta_arr, params)
    phi_k_plus_rho = phi * k_arr + rho
    sqrt_term = np.sqrt(phi_k_plus_rho * phi_k_plus_rho + 1.0 - rho * rho)
    return np.asarray(
        0.5 * theta_arr * (1.0 + rho * phi * k_arr + sqrt_term),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# No-butterfly check — evaluated at every θ in the backbone.
# ---------------------------------------------------------------------------


def is_butterfly_arb_free(
    backbone: SSVIBackbone,
    params: SSVIParams,
    *,
    tol: float = 1e-9,
) -> tuple[bool, str]:
    """Check (C1) and (C2) at every backbone ``θ``.

    Returns ``(True, "ok")`` if both inequalities hold at every backbone point,
    else ``(False, message)`` identifying which inequality fails and at which
    ``θ``.
    """
    thetas = np.asarray(backbone.theta, dtype=np.float64)
    phi = ssvi_phi(thetas, params)
    abs_rho = abs(params.rho)
    c1_lhs = thetas * phi * (1.0 + abs_rho)
    c2_lhs = thetas * phi * phi * (1.0 + abs_rho)

    # (C1) strict: lhs < 4. Failure when lhs >= 4 - tol.
    if np.any(c1_lhs >= 4.0 - tol):
        i = int(np.argmax(c1_lhs))
        return False, (
            f"(C1) θ·φ·(1+|ρ|) ≥ 4 at θ={float(thetas[i]):.5f} "
            f"(lhs={float(c1_lhs[i]):.5f}, gap={float(4.0 - c1_lhs[i]):.3e})"
        )
    # (C2) non-strict: lhs <= 4. Failure when lhs > 4 + tol.
    if np.any(c2_lhs > 4.0 + tol):
        i = int(np.argmax(c2_lhs))
        return False, (
            f"(C2) θ·φ²·(1+|ρ|) > 4 at θ={float(thetas[i]):.5f} "
            f"(lhs={float(c2_lhs[i]):.5f}, gap={float(4.0 - c2_lhs[i]):.3e})"
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# No-calendar check — backbone monotonicity + numerical smile-crossing test.
# ---------------------------------------------------------------------------


def is_calendar_arb_free(
    backbone: SSVIBackbone,
    params: SSVIParams,
    *,
    k_min: float = -3.0,
    k_max: float = 3.0,
    n_grid: int = 400,
    tol: float = 1e-10,
) -> tuple[bool, str]:
    """Confirm backbone monotonicity AND no smile crossings on a wide k-grid.

    The default grid covers ``|k| ≤ 3`` — wider than the typical Deribit BTC
    observed-strike range, so any wing crossing beyond the data is still
    caught.
    """
    if k_max <= k_min or n_grid < 10:
        raise ValueError("require k_max > k_min and n_grid >= 10")
    thetas = np.asarray(backbone.theta, dtype=np.float64)
    if len(thetas) < 2:
        return True, "single expiry, nothing to check"

    # 1) Backbone monotonicity.
    diffs = np.diff(thetas)
    if bool(np.any(diffs < -tol)):
        i = int(np.argmin(diffs)) + 1
        return False, (
            f"backbone non-monotone at index {i}: θ={float(thetas[i]):.5f} "
            f"< previous θ={float(thetas[i - 1]):.5f}"
        )

    # 2) Smile crossings on a dense grid.
    k_grid = np.linspace(k_min, k_max, n_grid, dtype=np.float64)
    for i in range(1, len(thetas)):
        w_prev = ssvi_total_variance(k_grid, float(thetas[i - 1]), params)
        w_curr = ssvi_total_variance(k_grid, float(thetas[i]), params)
        gap = w_curr - w_prev
        if bool(np.any(gap < -tol)):
            j = int(np.argmin(gap))
            return False, (
                f"smile crossing between expiry index {i - 1} and {i} "
                f"at k={float(k_grid[j]):+.4f}: w_later={float(w_curr[j]):.5f} "
                f"< w_earlier={float(w_prev[j]):.5f} (gap={float(-gap[j]):.3e})"
            )
    return True, "ok"


# ---------------------------------------------------------------------------
# Joint fit — SLSQP with hard butterfly inequality constraints.
# ---------------------------------------------------------------------------


def fit_ssvi(
    backbone: SSVIBackbone,
    points: list[tuple[float, float, int]],
) -> SSVIFitResult:
    """Joint SLSQP fit of ``(ρ, η, γ)`` given the precomputed backbone.

    Parameters
    ----------
    backbone
        ``SSVIBackbone`` sorted by ``t_years`` ascending. Built by the
        orchestrator from per-expiry SVI ATM fits.
    points
        ``(k, w_market, expiry_index)`` tuples. ``expiry_index`` indexes
        into ``backbone.expiries``. Filter and IV-solving are upstream.

    Returns
    -------
    SSVIFitResult
        Always non-throwing for runtime conditions. ``success`` /
        ``is_butterfly_free`` / ``is_calendar_free`` flags carry the verdict.
    """
    n_expiries = len(backbone.theta)
    n_points = len(points)

    if n_expiries < _MIN_EXPIRIES:
        return _fail(
            backbone,
            n_points,
            n_expiries,
            message=f"need >= {_MIN_EXPIRIES} expiries, got {n_expiries}",
        )

    # Refuse non-monotone backbone — no silent isotonic.
    thetas = np.asarray(backbone.theta, dtype=np.float64)
    if bool(np.any(np.diff(thetas) < -1e-12)):
        idx_drop = int(np.argmin(np.diff(thetas))) + 1
        return _fail(
            backbone,
            n_points,
            n_expiries,
            is_calendar_free=False,
            calendar_message=(
                f"backbone non-monotone at index {idx_drop}: "
                f"θ={float(thetas[idx_drop]):.5f} < previous "
                f"θ={float(thetas[idx_drop - 1]):.5f}"
            ),
            message="refusing to fit non-monotone backbone",
        )

    # Per-expiry point counts.
    counts = [sum(1 for _, _, i in points if i == e_idx) for e_idx in range(n_expiries)]
    too_thin = [i for i, c in enumerate(counts) if c < _MIN_POINTS_PER_EXPIRY]
    if too_thin:
        return _fail(
            backbone,
            n_points,
            n_expiries,
            message=(
                f"expiry index {too_thin} have <{_MIN_POINTS_PER_EXPIRY} points (counts={counts})"
            ),
        )

    ks = np.array([p[0] for p in points], dtype=np.float64)
    ws = np.array([p[1] for p in points], dtype=np.float64)
    point_thetas = thetas[np.array([p[2] for p in points], dtype=np.int64)]

    def objective(x: FloatArray) -> float:
        params = SSVIParams(rho=float(x[0]), eta=float(x[1]), gamma=float(x[2]))
        w_pred = ssvi_total_variance(ks, point_thetas, params)
        diff = w_pred - ws
        return float(np.sum(diff * diff))

    # Butterfly constraints: SLSQP wants g(x) >= 0. Build (C1) and (C2) per θ.
    def make_c1(theta_val: float) -> Callable[[FloatArray], float]:
        def c(x: FloatArray) -> float:
            rho = float(x[0])
            eta = float(x[1])
            gamma = float(x[2])
            phi = eta / float(theta_val**gamma)
            return float(4.0 - theta_val * phi * (1.0 + abs(rho)) - _C1_EPS)

        return c

    def make_c2(theta_val: float) -> Callable[[FloatArray], float]:
        def c(x: FloatArray) -> float:
            rho = float(x[0])
            eta = float(x[1])
            gamma = float(x[2])
            phi = eta / float(theta_val**gamma)
            return float(4.0 - theta_val * phi * phi * (1.0 + abs(rho)))

        return c

    constraints: list[dict[str, object]] = []
    for theta_val in (float(t) for t in thetas):
        constraints.append({"type": "ineq", "fun": make_c1(theta_val)})
        constraints.append({"type": "ineq", "fun": make_c2(theta_val)})

    bounds = [(-0.999, 0.999), (1e-4, 10.0), (0.0, 0.5)]
    x0 = np.array([-0.3, 1.0, 0.3], dtype=np.float64)

    try:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-12},
        )
    except (ValueError, RuntimeError) as exc:
        return _fail(
            backbone,
            n_points,
            n_expiries,
            message=f"optimiser raised: {exc}",
        )

    if not bool(result.success):
        return _fail(
            backbone,
            n_points,
            n_expiries,
            message=f"optimiser failed: {result.message}",
        )

    params = SSVIParams(
        rho=float(result.x[0]),
        eta=float(result.x[1]),
        gamma=float(result.x[2]),
    )
    rmse = float(np.sqrt(result.fun / max(n_points, 1)))

    butterfly_ok, butterfly_msg = is_butterfly_arb_free(backbone, params)
    calendar_ok, calendar_msg = is_calendar_arb_free(backbone, params)

    if butterfly_ok and calendar_ok:
        overall = "ok"
    else:
        flags = []
        if not butterfly_ok:
            flags.append("butterfly")
        if not calendar_ok:
            flags.append("calendar")
        overall = f"fitted but arb in {','.join(flags)}"

    return SSVIFitResult(
        params=params,
        backbone=backbone,
        rmse=rmse,
        n_points=n_points,
        n_expiries=n_expiries,
        success=True,
        is_butterfly_free=butterfly_ok,
        is_calendar_free=calendar_ok,
        butterfly_message=butterfly_msg,
        calendar_message=calendar_msg,
        message=overall,
    )


def _fail(
    backbone: SSVIBackbone,
    n_points: int,
    n_expiries: int,
    *,
    is_calendar_free: bool = False,
    calendar_message: str = "",
    butterfly_message: str = "",
    message: str = "",
) -> SSVIFitResult:
    return SSVIFitResult(
        params=None,
        backbone=backbone,
        rmse=math.nan,
        n_points=n_points,
        n_expiries=n_expiries,
        success=False,
        is_butterfly_free=False,
        is_calendar_free=is_calendar_free,
        butterfly_message=butterfly_message,
        calendar_message=calendar_message,
        message=message,
    )
