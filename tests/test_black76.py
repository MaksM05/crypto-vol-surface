"""Unit tests for the Black-76 pricer. No DB, no network."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.pricer import black76
from volsurface.pricer.black76 import (
    black76_gamma,
    black76_price,
    black76_theta,
    black76_vega,
    standard_delta,
)

# ----- naming guard --------------------------------------------------------


def test_standard_delta_not_named_delta() -> None:
    """Lock in the export name: textbook delta is `standard_delta`, never `delta`.

    Future PRs cannot grab a `delta` import and accidentally ship USD risk
    built on the un-adjusted textbook formula.
    """
    assert hasattr(black76, "standard_delta")
    assert not hasattr(black76, "delta")
    assert not hasattr(black76, "black76_delta")


# ----- put-call parity -----------------------------------------------------


@pytest.mark.parametrize("forward", [50_000.0, 75_000.0, 100_000.0])
@pytest.mark.parametrize("strike", [40_000.0, 70_000.0, 80_000.0, 120_000.0])
@pytest.mark.parametrize("t", [0.01, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("sigma", [0.3, 0.6, 1.2])
def test_put_call_parity(forward: float, strike: float, t: float, sigma: float) -> None:
    """c − p == (F − K) · DF within 1e-6."""
    df = 1.0
    c = float(black76_price(forward, strike, t, sigma, df, True))
    p = float(black76_price(forward, strike, t, sigma, df, False))
    assert c - p == pytest.approx((forward - strike) * df, abs=1e-6)


# ----- intrinsic edges -----------------------------------------------------


def test_intrinsic_at_zero_time() -> None:
    """At T = 0, price collapses to intrinsic · DF."""
    df = 0.9
    # ITM call.
    assert float(black76_price(80_000.0, 70_000.0, 0.0, 0.5, df, True)) == pytest.approx(
        10_000.0 * df
    )
    # OTM call.
    assert float(black76_price(70_000.0, 80_000.0, 0.0, 0.5, df, True)) == pytest.approx(0.0)
    # ITM put.
    assert float(black76_price(70_000.0, 80_000.0, 0.0, 0.5, df, False)) == pytest.approx(
        10_000.0 * df
    )


def test_intrinsic_at_zero_sigma() -> None:
    """sigma = 0 → intrinsic."""
    assert float(black76_price(80_000.0, 70_000.0, 0.5, 0.0, 1.0, True)) == pytest.approx(10_000.0)


def test_negative_time_raises() -> None:
    with pytest.raises(ValueError, match="time-to-expiry"):
        black76_price(75_000.0, 75_000.0, -0.1, 0.5, 1.0, True)


def test_negative_sigma_raises() -> None:
    with pytest.raises(ValueError, match="sigma"):
        black76_price(75_000.0, 75_000.0, 0.5, -0.1, 1.0, True)


# ----- Greeks: bump-and-revalue --------------------------------------------

# Choose ATM-ish inputs where Greeks are well-conditioned for finite differences.
_F = 75_000.0
_K = 76_000.0
_T = 0.4
_SIGMA = 0.6
_DF = 1.0


def test_vega_matches_bump_revalue() -> None:
    h = 1e-5
    analytic = float(black76_vega(_F, _K, _T, _SIGMA, _DF))
    numeric = (
        float(black76_price(_F, _K, _T, _SIGMA + h, _DF, True))
        - float(black76_price(_F, _K, _T, _SIGMA - h, _DF, True))
    ) / (2.0 * h)
    assert analytic == pytest.approx(numeric, rel=1e-5)


def test_vega_same_for_call_and_put() -> None:
    """Vega is identical for calls and puts (put-call parity is linear in σ)."""
    h = 1e-5
    call_vega_num = (
        float(black76_price(_F, _K, _T, _SIGMA + h, _DF, True))
        - float(black76_price(_F, _K, _T, _SIGMA - h, _DF, True))
    ) / (2.0 * h)
    put_vega_num = (
        float(black76_price(_F, _K, _T, _SIGMA + h, _DF, False))
        - float(black76_price(_F, _K, _T, _SIGMA - h, _DF, False))
    ) / (2.0 * h)
    assert call_vega_num == pytest.approx(put_vega_num, rel=1e-5)


def test_gamma_matches_bump_revalue() -> None:
    h = 1.0  # bump F by $1; gamma is ∂²price/∂F²
    p_up = float(black76_price(_F + h, _K, _T, _SIGMA, _DF, True))
    p_mid = float(black76_price(_F, _K, _T, _SIGMA, _DF, True))
    p_dn = float(black76_price(_F - h, _K, _T, _SIGMA, _DF, True))
    numeric = (p_up - 2.0 * p_mid + p_dn) / (h * h)
    analytic = float(black76_gamma(_F, _K, _T, _SIGMA, _DF))
    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_theta_matches_bump_revalue() -> None:
    """Analytic theta = ∂price/∂T should match central finite diff in T."""
    h = 1e-5
    numeric = (
        float(black76_price(_F, _K, _T + h, _SIGMA, _DF, True))
        - float(black76_price(_F, _K, _T - h, _SIGMA, _DF, True))
    ) / (2.0 * h)
    analytic = float(black76_theta(_F, _K, _T, _SIGMA, _DF))
    assert analytic == pytest.approx(numeric, rel=1e-4)


def test_standard_delta_matches_bump_revalue() -> None:
    """∂price/∂F (textbook). Tests the function works correctly even though
    it's NOT the right number to use for Deribit USD risk."""
    h = 0.5
    numeric_call = (
        float(black76_price(_F + h, _K, _T, _SIGMA, _DF, True))
        - float(black76_price(_F - h, _K, _T, _SIGMA, _DF, True))
    ) / (2.0 * h)
    analytic_call = float(standard_delta(_F, _K, _T, _SIGMA, _DF, True))
    assert analytic_call == pytest.approx(numeric_call, rel=1e-5)

    numeric_put = (
        float(black76_price(_F + h, _K, _T, _SIGMA, _DF, False))
        - float(black76_price(_F - h, _K, _T, _SIGMA, _DF, False))
    ) / (2.0 * h)
    analytic_put = float(standard_delta(_F, _K, _T, _SIGMA, _DF, False))
    assert analytic_put == pytest.approx(numeric_put, rel=1e-5)


# ----- vectorisation -------------------------------------------------------


def test_price_vectorises_over_strike() -> None:
    strikes = np.array([70_000.0, 75_000.0, 80_000.0, 90_000.0])
    prices = black76_price(_F, strikes, _T, _SIGMA, _DF, True)
    assert prices.shape == (4,)
    assert np.all(np.diff(prices) < 0), "call price must be strictly decreasing in K"


def test_price_vectorises_over_is_call() -> None:
    is_call = np.array([True, False, True, False])
    strikes = np.array([70_000.0, 70_000.0, 80_000.0, 80_000.0])
    prices = black76_price(_F, strikes, _T, _SIGMA, _DF, is_call)
    assert prices.shape == (4,)
