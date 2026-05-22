"""Unit tests for the IV solver. No DB, no network."""

from __future__ import annotations

import math

import pytest

from volsurface.pricer.black76 import black76_price
from volsurface.pricer.iv_solver import implied_vol

# ----- price -> sigma -> price roundtrip -----------------------------------


@pytest.mark.parametrize("sigma_true", [0.2, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("forward", [50_000.0, 75_000.0, 100_000.0])
@pytest.mark.parametrize("k_rel", [0.7, 0.9, 1.0, 1.1, 1.3])
@pytest.mark.parametrize("t", [0.05, 0.25, 1.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_iv_roundtrip(
    sigma_true: float, forward: float, k_rel: float, t: float, is_call: bool
) -> None:
    """price(sigma_true) -> solve -> price within 1e-8 (the spec'd tolerance).

    σ itself round-trips to ~1e-6 in well-conditioned cases and ~1e-7 in
    deep-wing low-vol cases where Brent's xtol bottoms out — but the PRICE
    residual is what calibration and validation care about, and that meets
    1e-8 everywhere.
    """
    strike = forward * k_rel
    df = 1.0
    price = float(black76_price(forward, strike, t, sigma_true, df, is_call))
    if price < 1e-10:  # numerical underflow on extreme OTM
        pytest.skip("price underflows to zero")
    # Deep ITM low-vol: F·N(d1)-K·N(d2) cancels and our computed price can sit
    # within float noise of intrinsic, leaving no recoverable σ. Skip those —
    # they're a pricer-formula limitation, not a solver failure.
    intrinsic = max(forward - strike, 0.0) * df if is_call else max(strike - forward, 0.0) * df
    if price - intrinsic < 1e-6:
        pytest.skip("time value below numerical floor; no recoverable IV")
    sigma_solved = implied_vol(price, forward, strike, t, df, is_call)
    assert not math.isnan(sigma_solved)
    price_solved = float(black76_price(forward, strike, t, sigma_solved, df, is_call))
    assert price_solved == pytest.approx(price, abs=1e-8)
    # σ recovered to within a generous tolerance — proves we found the right root.
    assert sigma_solved == pytest.approx(sigma_true, abs=1e-4, rel=1e-4)


# ----- no-arb violations return nan, never crash ---------------------------


def test_solver_returns_nan_on_sub_intrinsic_price() -> None:
    """Call priced below intrinsic = no valid IV. Return nan, no exception."""
    f, k, t, df = 75_000.0, 70_000.0, 0.5, 1.0
    intrinsic = (f - k) * df  # 5000
    bad_price = intrinsic - 100.0  # 4900, sub-intrinsic
    result = implied_vol(bad_price, f, k, t, df, is_call=True)
    assert math.isnan(result)


def test_solver_returns_nan_on_above_upper_bound() -> None:
    """Call priced above F·DF cannot be solved."""
    f, k, t, df = 75_000.0, 80_000.0, 0.5, 1.0
    bad_price = f * df + 1.0  # > upper bound
    result = implied_vol(bad_price, f, k, t, df, is_call=True)
    assert math.isnan(result)


def test_solver_returns_nan_on_put_below_intrinsic() -> None:
    f, k, t, df = 75_000.0, 80_000.0, 0.5, 1.0
    intrinsic = (k - f) * df  # 5000
    bad_price = intrinsic - 50.0
    result = implied_vol(bad_price, f, k, t, df, is_call=False)
    assert math.isnan(result)


def test_solver_returns_nan_on_zero_time() -> None:
    """T = 0 leaves no sigma to recover; return nan."""
    result = implied_vol(100.0, 75_000.0, 75_000.0, 0.0, 1.0, is_call=True)
    assert math.isnan(result)


def test_solver_returns_nan_on_non_finite_inputs() -> None:
    assert math.isnan(implied_vol(float("nan"), 75_000.0, 75_000.0, 0.5, 1.0, True))
    assert math.isnan(implied_vol(100.0, float("inf"), 75_000.0, 0.5, 1.0, True))


# ----- robustness ----------------------------------------------------------


def test_solver_handles_atm_well() -> None:
    """ATM call should solve cleanly."""
    f, k, t, df = 75_000.0, 75_000.0, 0.5, 1.0
    for sigma_true in (0.3, 0.6, 1.0):
        price = float(black76_price(f, k, t, sigma_true, df, True))
        assert implied_vol(price, f, k, t, df, True) == pytest.approx(sigma_true, abs=1e-8)


def test_solver_handles_deep_wing_small_premium() -> None:
    """Far-OTM call with tiny but non-zero premium — solver converges or returns nan,
    never throws."""
    f, k, t, df = 75_000.0, 120_000.0, 0.1, 1.0
    sigma_true = 0.6
    price = float(black76_price(f, k, t, sigma_true, df, True))
    result = implied_vol(price, f, k, t, df, True)
    if not math.isnan(result):
        assert result == pytest.approx(sigma_true, rel=1e-4)
