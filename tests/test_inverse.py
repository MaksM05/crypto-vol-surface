"""Unit tests for the inverse-contract (Deribit-convention) delta.

Validates the formula ``inverse_delta = standard_delta − price/spot`` against:

- the closed-form formula itself (units-consistency hand-check),
- first-principles limits (OTM convergence, ITM divergence, call saturation
  at zero, put unbounded below),
- edge cases (zero price, zero spot).

Does not yet validate against Deribit's published ticker.delta — that
hookup is deferred (see ``inverse.py`` docstring "Known limitation").
"""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.pricer.black76 import black76_price, standard_delta
from volsurface.pricer.inverse import inverse_delta

_F = 75_000.0  # USD/BTC, also S since DF = 1
_S = 75_000.0
_T = 0.5
_SIGMA = 0.6
_DF = 1.0


# ----- units-consistency hand check ----------------------------------------


def test_inverse_delta_units_consistent() -> None:
    """Pin the formula against a hand computation at ATM (F = S = K = 75k).

    Hand computation:
        d1 = 0.5·σ²·T / (σ·√T) = 0.09 / 0.42426 ≈ 0.21213
        d2 = −0.21213
        N(d1) ≈ 0.5840  → standard_delta ≈ 0.5840
        N(d2) ≈ 0.4160
        C_usd = 75000·(0.5840 − 0.4160) ≈ 12 600
        inverse_delta = 0.5840 − 12600/75000 ≈ 0.4160
    """
    sd = float(standard_delta(_F, _F, _T, _SIGMA, _DF, True))
    px = float(black76_price(_F, _F, _T, _SIGMA, _DF, True))

    # The function under test must reproduce the closed form exactly (modulo
    # float arithmetic).
    result = float(inverse_delta(sd, px, _S))
    assert result == pytest.approx(sd - px / _S, abs=1e-15)

    # And the absolute value matches the hand-computed 0.4160 within slide-rule
    # precision. Catches any sign / algebra slip in the formula itself.
    assert result == pytest.approx(0.4160, abs=1e-3)
    # Sanity: the correction is materially smaller than std delta at ATM.
    assert 0.40 < result < 0.45
    assert sd == pytest.approx(0.584, abs=1e-3)


# ----- THE KEY TEST: OTM convergence ---------------------------------------


def test_inverse_converges_to_standard_in_deep_otm() -> None:
    """Deep OTM: C/S → 0, so inverse_delta ≈ standard_delta.

    This is the load-bearing limit — it encodes that the inverse correction
    is driven purely by C/S and vanishes where there is no premium.
    """
    strike = 5.0 * _F  # extreme OTM call: K = $375k vs F = $75k
    sd = float(standard_delta(_F, strike, _T, _SIGMA, _DF, True))
    px = float(black76_price(_F, strike, _T, _SIGMA, _DF, True))
    assert px / _S < 1e-3, "fixture: deep-OTM premium should be << 1 BTC"
    result = float(inverse_delta(sd, px, _S))
    assert result == pytest.approx(sd, abs=1e-3)


# ----- ITM divergence ------------------------------------------------------


def test_inverse_diverges_materially_in_itm() -> None:
    """Deep ITM: C/S is large; inverse_delta differs from standard by >> noise."""
    strike = 0.4 * _F  # deep ITM call: K = $30k vs F = $75k
    sd = float(standard_delta(_F, strike, _T, _SIGMA, _DF, True))
    px = float(black76_price(_F, strike, _T, _SIGMA, _DF, True))
    result = float(inverse_delta(sd, px, _S))
    assert abs(result - sd) > 0.3, (
        f"deep-ITM correction should be material; got |inv-std|={abs(result - sd):.4f}"
    )


# ----- limit: call delta saturates at zero deep ITM ------------------------


def test_inverse_call_saturates_at_zero_for_deep_itm() -> None:
    """Famous property: a deep-ITM coin-settled call has inverse delta → 0
    because the BTC payoff is bounded above by 1 BTC."""
    strike = 0.05 * _F  # extreme ITM: K=$3.75k vs F=$75k, pushes N(d1) to ~1
    sd = float(standard_delta(_F, strike, _T, _SIGMA, _DF, True))
    px = float(black76_price(_F, strike, _T, _SIGMA, _DF, True))
    assert sd > 0.999, "fixture: extreme-ITM standard delta should be ≈ 1"
    result = float(inverse_delta(sd, px, _S))
    # At this strike inverse_delta ≈ 1 − (F−K)/S = K/S = 0.05 — clear saturation.
    assert abs(result) < 0.1, (
        f"extreme-ITM inverse call delta should saturate near 0; got {result:.4f}"
    )


# ----- limit: put delta unbounded below for deep ITM -----------------------


def test_inverse_put_can_exceed_minus_one_for_deep_itm() -> None:
    """A deep-ITM coin-settled put has inverse |delta| > 1 — BTC payoff K/S
    grows without bound as S → 0."""
    strike = 3.0 * _F  # very deep ITM put: K = $225k vs F = $75k
    sd = float(standard_delta(_F, strike, _T, _SIGMA, _DF, False))
    px = float(black76_price(_F, strike, _T, _SIGMA, _DF, False))
    assert sd < -0.95, "fixture: very-deep-ITM put standard delta should be ≈ -1"
    result = float(inverse_delta(sd, px, _S))
    assert result < -1.0, f"deep-ITM inverse put delta should be < -1; got {result:.4f}"


# ----- signs at moderate strikes -------------------------------------------


def test_inverse_signs_match_standard_at_moderate_strikes() -> None:
    """Slight OTM call > 0; slight OTM put < 0."""
    k_call, k_put = 1.05 * _F, 0.95 * _F
    sd_c = float(standard_delta(_F, k_call, _T, _SIGMA, _DF, True))
    px_c = float(black76_price(_F, k_call, _T, _SIGMA, _DF, True))
    sd_p = float(standard_delta(_F, k_put, _T, _SIGMA, _DF, False))
    px_p = float(black76_price(_F, k_put, _T, _SIGMA, _DF, False))
    assert float(inverse_delta(sd_c, px_c, _S)) > 0
    assert float(inverse_delta(sd_p, px_p, _S)) < 0


# ----- edge cases ----------------------------------------------------------


def test_inverse_with_zero_price_returns_standard() -> None:
    """price = 0 → no correction → exactly standard_delta."""
    assert float(inverse_delta(0.3, 0.0, _S)) == 0.3
    assert float(inverse_delta(-0.4, 0.0, _S)) == -0.4


def test_inverse_raises_on_zero_spot() -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        inverse_delta(0.5, 100.0, 0.0)


def test_inverse_raises_on_negative_spot() -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        inverse_delta(0.5, 100.0, -75_000.0)


# ----- vectorisation -------------------------------------------------------


def test_inverse_vectorises_across_strikes() -> None:
    """Array inputs broadcast and return matching shape."""
    strikes = np.array([60_000.0, 70_000.0, 75_000.0, 80_000.0, 90_000.0])
    sd = standard_delta(_F, strikes, _T, _SIGMA, _DF, True)
    px = black76_price(_F, strikes, _T, _SIGMA, _DF, True)
    result = inverse_delta(sd, px, _S)
    assert result.shape == (5,)
    # The correction must be strictly monotone in K for calls at fixed F:
    # higher K → smaller premium → smaller |correction| → result closer to sd.
    correction = sd - result
    assert np.all(np.diff(correction) < 0)
