"""Unit tests for the SSVI surface calibrator. No DB, no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from volsurface.calibration.ssvi import (
    SSVIBackbone,
    SSVIParams,
    fit_ssvi,
    is_butterfly_arb_free,
    is_calendar_arb_free,
    ssvi_phi,
    ssvi_total_variance,
)

# Typical BTC-like, well-behaved SSVI params.
_GOOD_PARAMS = SSVIParams(rho=-0.30, eta=1.0, gamma=0.30)


def _make_backbone(t_years: list[float], theta: list[float]) -> SSVIBackbone:
    """Synthetic backbone with arbitrary expiry datetimes (only t_years matters here)."""
    base = datetime(2026, 5, 25, tzinfo=UTC)
    expiries = tuple(base + timedelta(days=int(t * 365.0)) for t in t_years)
    return SSVIBackbone(
        expiries=expiries,
        t_years=tuple(t_years),
        theta=tuple(theta),
    )


# ----- model evaluation ----------------------------------------------------


def test_ssvi_total_variance_at_atm_equals_theta() -> None:
    """w(0, θ) = θ for any params (formula sanity)."""
    for theta in (0.01, 0.05, 0.20):
        for params in (
            SSVIParams(rho=-0.5, eta=1.0, gamma=0.0),
            SSVIParams(rho=0.0, eta=2.0, gamma=0.5),
            SSVIParams(rho=+0.8, eta=0.5, gamma=0.3),
        ):
            assert float(ssvi_total_variance(0.0, theta, params)) == pytest.approx(theta, abs=1e-14)


def test_ssvi_phi_power_law() -> None:
    """φ(θ) = η/θ^γ, broadcasts over θ arrays."""
    p = SSVIParams(rho=0.0, eta=1.5, gamma=0.4)
    theta = np.array([0.01, 0.05, 0.1])
    phi = ssvi_phi(theta, p)
    assert np.allclose(phi, 1.5 / theta**0.4)


def test_ssvi_vectorises_over_k() -> None:
    k = np.linspace(-2.0, 2.0, 50)
    w = ssvi_total_variance(k, 0.05, _GOOD_PARAMS)
    assert w.shape == (50,)
    assert np.all(w > 0)


def test_ssvi_vectorises_over_paired_k_and_theta() -> None:
    """Element-wise broadcasting when k and theta are same-length arrays."""
    k = np.array([-0.5, 0.0, 0.5])
    theta = np.array([0.04, 0.05, 0.06])
    w = ssvi_total_variance(k, theta, _GOOD_PARAMS)
    assert w.shape == (3,)
    # ATM element collapses to θ:
    assert w[1] == pytest.approx(0.05, abs=1e-14)


# ----- recovery of known parameters ----------------------------------------


def test_ssvi_recovers_known_params() -> None:
    """Generate a surface from known params + backbone, fit, recover all three."""
    true_params = SSVIParams(rho=-0.40, eta=1.20, gamma=0.35)
    backbone = _make_backbone(
        t_years=[0.10, 0.25, 0.50, 1.00],
        theta=[0.04, 0.08, 0.12, 0.20],
    )
    # Synthetic noise-free observations: 15 k-points per expiry.
    rng = np.random.default_rng(seed=7)
    points: list[tuple[float, float, int]] = []
    for i, theta in enumerate(backbone.theta):
        ks = np.sort(rng.uniform(-1.0, 1.0, size=15))
        for k in ks:
            w = float(ssvi_total_variance(float(k), theta, true_params))
            points.append((float(k), w, i))

    fit = fit_ssvi(backbone, points)
    assert fit.success, f"fit did not converge: {fit.message}"
    assert fit.params is not None
    assert fit.params.rho == pytest.approx(true_params.rho, abs=1e-3)
    assert fit.params.eta == pytest.approx(true_params.eta, abs=1e-3)
    assert fit.params.gamma == pytest.approx(true_params.gamma, abs=1e-3)
    assert fit.is_butterfly_free
    assert fit.is_calendar_free


# ----- butterfly: passes on arb-free params --------------------------------


def test_butterfly_check_passes_on_arbfree_params() -> None:
    backbone = _make_backbone(
        t_years=[0.10, 0.25, 0.50, 1.00],
        theta=[0.04, 0.08, 0.12, 0.20],
    )
    ok, msg = is_butterfly_arb_free(backbone, _GOOD_PARAMS)
    assert ok, f"unexpected butterfly violation: {msg}"


# ----- KEY TEST: butterfly fails on hand-bad params ------------------------


def test_butterfly_check_fails_on_arb_params() -> None:
    """Hand-bad SSVI must be rejected with a message identifying which inequality.

    Construction: ρ=0.95, η=8.0, γ=0.5 against a backbone reaching θ=0.10.

        φ(0.10) = 8 / √0.10 = 25.298
        (C1) θ·φ·(1+|ρ|) = 0.10 · 25.298 · 1.95 = 4.933  (must be < 4)
        (C2) θ·φ²·(1+|ρ|) = 0.10 · 640.0  · 1.95 = 124.8 (must be ≤ 4)

    Both (C1) and (C2) are violated; the check reports (C1) first (it's the
    earlier failure in the standard ordering). Proves the check has teeth —
    the message must name the offending inequality AND identify the θ.
    """
    backbone = _make_backbone(t_years=[0.10, 0.25], theta=[0.04, 0.10])
    bad = SSVIParams(rho=0.95, eta=8.0, gamma=0.5)
    ok, msg = is_butterfly_arb_free(backbone, bad)
    assert not ok
    assert "(C1)" in msg or "(C2)" in msg, f"message must name the violated inequality, got: {msg}"
    assert "θ=" in msg or "theta=" in msg.lower(), (
        f"message must identify the offending θ, got: {msg}"
    )

    # Independently verify the LHS values match the analytic prediction at θ=0.10.
    phi = 8.0 / (0.10**0.5)
    c1_lhs = 0.10 * phi * (1.0 + 0.95)
    c2_lhs = 0.10 * phi * phi * (1.0 + 0.95)
    assert c1_lhs > 4.0, "fixture: (C1) must actually be violated at θ=0.10"
    assert c2_lhs > 4.0, "fixture: (C2) must actually be violated at θ=0.10"


# ----- calendar: passes on monotonic backbone with sensible params ---------


def test_calendar_check_passes_on_monotonic_backbone() -> None:
    """Well-ordered backbone + typical params → no smile crossings on |k|≤3."""
    backbone = _make_backbone(
        t_years=[0.10, 0.25, 0.50, 1.00],
        theta=[0.04, 0.08, 0.12, 0.20],
    )
    ok, msg = is_calendar_arb_free(backbone, _GOOD_PARAMS)
    assert ok, f"unexpected calendar violation: {msg}"


# ----- KEY TEST: calendar fails on non-monotonic backbone ------------------


def test_calendar_check_fails_on_nonmonotonic_backbone() -> None:
    """Backbone that decreases in T must be rejected with a teeth-having message.

    Constructed: θ = [0.05, 0.04, 0.06] dips at index 1. The check must
    return False and the message must name the offending index AND the
    decreasing θ values. Proves the calendar check is not just rubber-stamping.
    """
    backbone = _make_backbone(t_years=[0.10, 0.25, 0.50], theta=[0.05, 0.04, 0.06])
    ok, msg = is_calendar_arb_free(backbone, _GOOD_PARAMS)
    assert not ok
    assert "non-monotone" in msg or "monoton" in msg, (
        f"message must say the backbone is non-monotone, got: {msg}"
    )
    assert "index 1" in msg, f"message must identify the dipping index, got: {msg}"
    # Verify both adjacent θ values appear in the diagnostic so a future
    # regression can't hide the violation behind a generic "non-monotone"
    # string.
    assert "0.04" in msg and "0.05" in msg, (
        f"message must surface the offending θ values, got: {msg}"
    )


def test_calendar_check_handles_single_expiry() -> None:
    """One expiry has no calendar dimension — trivially passes."""
    backbone = _make_backbone(t_years=[0.25], theta=[0.05])
    ok, msg = is_calendar_arb_free(backbone, _GOOD_PARAMS)
    assert ok
    assert "single" in msg


def test_calendar_check_invalid_grid_raises() -> None:
    bb = _make_backbone(t_years=[0.1, 0.2], theta=[0.04, 0.05])
    with pytest.raises(ValueError):
        is_calendar_arb_free(bb, _GOOD_PARAMS, k_min=1.0, k_max=-1.0)
    with pytest.raises(ValueError):
        is_calendar_arb_free(bb, _GOOD_PARAMS, n_grid=5)


# ----- fit refusals --------------------------------------------------------


def test_fit_refuses_too_few_expiries() -> None:
    """Joint SSVI needs ≥ 3 expiries to disentangle the (ρ, η, γ) term structure."""
    backbone = _make_backbone(t_years=[0.10, 0.25], theta=[0.04, 0.08])
    points = [(0.0, 0.04, 0), (0.1, 0.04, 0), (0.0, 0.08, 1), (0.1, 0.08, 1)]
    fit = fit_ssvi(backbone, points)
    assert not fit.success
    assert "3 expiries" in fit.message


def test_fit_refuses_non_monotone_backbone() -> None:
    """Non-monotone backbone → fit refuses immediately, no SLSQP call."""
    backbone = _make_backbone(t_years=[0.10, 0.25, 0.50], theta=[0.05, 0.04, 0.06])
    # Enough points per expiry so it's the backbone, not point counts, that trips it.
    rng = np.random.default_rng(0)
    points = []
    for i, theta in enumerate(backbone.theta):
        for k in rng.uniform(-0.5, 0.5, size=10):
            points.append((float(k), theta + 0.001, i))
    fit = fit_ssvi(backbone, points)
    assert not fit.success
    assert not fit.is_calendar_free
    assert "non-monotone" in fit.calendar_message
