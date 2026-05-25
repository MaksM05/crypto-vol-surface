"""Unit tests for the SVI calibrator. No DB, no network."""

from __future__ import annotations

import math

import numpy as np
import pytest

from volsurface.calibration.svi import (
    SVIParams,
    _svi_derivatives,
    durrleman_g,
    fit_svi,
    is_butterfly_arb_free,
    svi_total_variance,
)

# Typical BTC-like, well-behaved smile.
_GOOD = SVIParams(a=0.04, b=0.10, rho=-0.30, m=0.0, sigma=0.10)


# ----- model evaluation ----------------------------------------------------


def test_svi_total_variance_at_vertex() -> None:
    """At k = m: u = 0, r = σ, so w(m) = a + b·σ. Pure formula sanity."""
    p = _GOOD
    expected = p.a + p.b * p.sigma
    assert float(svi_total_variance(p.m, p)) == pytest.approx(expected, abs=1e-14)


def test_svi_first_derivative_matches_bump_revalue() -> None:
    """Analytic w' matches central FD with small h."""
    p = _GOOD
    k = np.array([-1.0, -0.3, 0.0, 0.2, 0.7])
    _, w_prime, _ = _svi_derivatives(k, p)
    h = 1e-6
    w_up = svi_total_variance(k + h, p)
    w_dn = svi_total_variance(k - h, p)
    w_prime_num = (w_up - w_dn) / (2.0 * h)
    assert np.allclose(w_prime, w_prime_num, rtol=1e-5)


def test_svi_second_derivative_matches_bump_revalue() -> None:
    """Analytic w'' matches central FD with a coarser h (second-order FD eats
    precision quadratically — h ~ 1e-3 is the sweet spot)."""
    p = _GOOD
    k = np.array([-1.0, -0.3, 0.0, 0.2, 0.7])
    _, _, w_pp = _svi_derivatives(k, p)
    h = 1e-3
    w_up = svi_total_variance(k + h, p)
    w_dn = svi_total_variance(k - h, p)
    w_mid = svi_total_variance(k, p)
    w_pp_num = (w_up - 2.0 * w_mid + w_dn) / (h * h)
    assert np.allclose(w_pp, w_pp_num, rtol=1e-4)


def test_svi_vectorises() -> None:
    k = np.linspace(-2.0, 2.0, 50)
    w = svi_total_variance(k, _GOOD)
    assert w.shape == (50,)
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


# ----- recovery of known parameters ----------------------------------------


def test_svi_recovers_known_params_noise_free() -> None:
    """Sample w from known params on a strike grid, fit, recover within 1e-6."""
    true_params = SVIParams(a=0.035, b=0.12, rho=-0.25, m=0.02, sigma=0.08)
    k = np.linspace(-1.0, 1.0, 30)
    w = svi_total_variance(k, true_params)
    fit = fit_svi(k, w)
    assert fit.success
    assert fit.params is not None
    assert fit.params.a == pytest.approx(true_params.a, abs=1e-6)
    assert fit.params.b == pytest.approx(true_params.b, abs=1e-6)
    assert fit.params.rho == pytest.approx(true_params.rho, abs=1e-6)
    assert fit.params.m == pytest.approx(true_params.m, abs=1e-6)
    assert fit.params.sigma == pytest.approx(true_params.sigma, abs=1e-6)
    assert fit.rmse < 1e-9  # least_squares' natural floor on a perfectly clean problem


def test_svi_recovers_known_params_with_small_noise() -> None:
    """Add 0.5% relative noise to w; fit should recover params within a sensible tolerance."""
    rng = np.random.default_rng(seed=42)
    true_params = SVIParams(a=0.040, b=0.10, rho=-0.30, m=0.0, sigma=0.10)
    k = np.linspace(-1.0, 1.0, 40)
    w_clean = svi_total_variance(k, true_params)
    w_noisy = w_clean * (1.0 + 0.005 * rng.standard_normal(k.shape))
    fit = fit_svi(k, w_noisy)
    assert fit.success
    assert fit.params is not None
    # 0.5% noise on w propagates to ~1% on params in well-conditioned smiles.
    assert fit.params.rho == pytest.approx(true_params.rho, abs=0.05)
    assert fit.params.m == pytest.approx(true_params.m, abs=0.05)
    assert fit.is_butterfly_free


def test_fit_returns_failure_on_too_few_points() -> None:
    """4 points < 6: refuse to fit. No exception."""
    k = np.linspace(-0.5, 0.5, 4)
    w = np.array([0.06, 0.05, 0.05, 0.06])
    fit = fit_svi(k, w)
    assert not fit.success
    assert fit.params is None
    assert math.isnan(fit.rmse)
    assert fit.n_points == 4
    assert "6" in fit.message


def test_fit_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_svi(np.array([0.0, 0.5]), np.array([0.05, 0.06, 0.07]))


# ----- the no-butterfly check ----------------------------------------------


def test_butterfly_check_passes_on_arbfree_params() -> None:
    """Typical BTC-like smile: every g(k) >= 0."""
    assert is_butterfly_arb_free(_GOOD)
    # Spot-check the grid: min(g) is comfortably positive, not borderline.
    g = durrleman_g(np.linspace(-3.0, 3.0, 2000), _GOOD)
    assert np.all(np.isfinite(g))
    assert float(np.min(g)) > 0.0


def test_butterfly_check_fails_on_arb_params() -> None:
    """THE KEY TEST: a hand-constructed pathological SVI must be rejected.

    Construction: ``b = 2.5, ρ = 0.95`` gives a right-wing slope
    ``b·(1+ρ) = 4.875`` — far above Lee's bound of 2. The very steep ρ
    on top of a sharp vertex (``σ = 0.02``) produces ``g(k) < 0`` in the
    right wing where call prices become non-convex in strike.

    Proves the check actually rejects arbitrage rather than rubber-stamping.
    """
    bad = SVIParams(a=0.04, b=2.5, rho=0.95, m=0.0, sigma=0.02)
    assert not is_butterfly_arb_free(bad)

    # Be specific about WHERE the violation lives, so a future regression
    # of the formula can't accidentally make this pass by erasing g entirely.
    k_grid = np.linspace(-3.0, 3.0, 2000)
    g = durrleman_g(k_grid, bad)
    assert float(np.min(g)) < 0.0, "g must dip below zero somewhere for an arb-y smile"
    # Violation should be in the right wing (positive ρ → right-side arb).
    k_at_min = float(k_grid[int(np.argmin(g))])
    assert k_at_min > 0.0, f"expected right-wing violation, min was at k={k_at_min:.3f}"


def test_butterfly_check_fails_on_extreme_lee_violation() -> None:
    """Second pathological case: huge ``b`` with mild ρ still violates."""
    bad = SVIParams(a=0.01, b=3.5, rho=0.0, m=0.0, sigma=0.05)
    # b·(1+|ρ|) = 3.5 >> 2 — Lee bound violated symmetrically.
    assert not is_butterfly_arb_free(bad)


def test_butterfly_check_invalid_grid_raises() -> None:
    with pytest.raises(ValueError):
        is_butterfly_arb_free(_GOOD, k_min=1.0, k_max=-1.0)
    with pytest.raises(ValueError):
        is_butterfly_arb_free(_GOOD, n_grid=5)


def test_durrleman_g_continuous_on_grid() -> None:
    """No NaNs, no jumps on a typical smile."""
    g = durrleman_g(np.linspace(-3.0, 3.0, 1000), _GOOD)
    assert np.all(np.isfinite(g))
    # Continuous → bounded variation locally. Loose check: no spike > 10.
    assert np.max(np.abs(np.diff(g))) < 10.0
