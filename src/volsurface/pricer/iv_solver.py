"""Scalar implied-vol solver: Newton-Raphson with a Brent fallback.

Returns ``nan`` (not an exception, not a garbage root) when the market price
violates the no-arbitrage bounds and no valid IV exists. That ``nan`` is the
correct signal for the wing-noise filter — never silently return a number
that solves nothing.

Algorithm:

1. **No-arb bounds.** For a call: ``intrinsic = max(F - K, 0) · DF``,
   ``upper = F · DF``. For a put: ``intrinsic = max(K - F, 0) · DF``,
   ``upper = K · DF``. A market price strictly outside ``(intrinsic, upper)``
   returns ``nan``.
2. **Newton-Raphson** with vega as derivative. Seed: Brenner-Subrahmanyam
   approximation clamped to ``[0.05, 3.0]``. Falls back to Brent if a step
   leaves the bracket or vega collapses.
3. **Brent bisection** on ``[sigma_lo, sigma_hi]`` via ``scipy.optimize.brentq``.
   If Newton has not converged after ``max_iter`` iterations, or stepped out
   of bounds, or hit a near-zero vega, the residual is solved by Brent.
4. If Brent also fails (no sign change in bracket, etc.), returns ``nan``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from volsurface.pricer.black76 import black76_price, black76_vega

_VEGA_FLOOR = 1e-10  # below this, Newton step is unreliable; fall back to Brent.


def implied_vol(
    market_price: float,
    forward: float,
    strike: float,
    t: float,
    df: float,
    is_call: bool,
    *,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
    tol: float = 1e-10,
    max_iter: int = 64,
) -> float:
    """Recover the Black-76 implied volatility from a single observed price.

    Parameters
    ----------
    market_price
        Observed option price in the same unit as ``forward`` and ``strike``
        (USD for this project — the BTC→USD conversion happens in the
        validation layer, not here).
    forward, strike, t, df
        Black-76 inputs; ``t`` in years (Act/365), ``df`` typically ``1.0``
        for the zero-rate Deribit convention.
    is_call
        ``True`` for call, ``False`` for put.
    sigma_lo, sigma_hi
        Bracket for the solver. Defaults span 0.0001% to 500% vol.
    tol
        Newton convergence tolerance on ``|price(sigma) - market_price|``.
    max_iter
        Newton iteration cap before falling back to Brent.

    Returns
    -------
    float
        The implied volatility (decimal, not percent). ``float('nan')`` if
        the price is outside the no-arb bracket or the solver fails to
        converge inside ``[sigma_lo, sigma_hi]``.
    """
    if not math.isfinite(market_price) or not math.isfinite(forward) or not math.isfinite(strike):
        return math.nan
    if t <= 0.0 or df <= 0.0 or forward <= 0.0 or strike <= 0.0:
        return math.nan

    intrinsic = max(forward - strike, 0.0) * df if is_call else max(strike - forward, 0.0) * df
    upper = forward * df if is_call else strike * df

    # Strict inequality: a market price equal to intrinsic or upper has only a
    # degenerate solution (sigma = 0 or sigma = +inf) outside our bracket.
    if not (intrinsic < market_price < upper):
        return math.nan

    sigma = _initial_guess(market_price, intrinsic, forward, t)

    # Newton-Raphson.
    for _ in range(max_iter):
        price = float(black76_price(forward, strike, t, sigma, df, is_call))
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        vega = float(black76_vega(forward, strike, t, sigma, df))
        if vega < _VEGA_FLOOR:
            break  # vega-flat region; let Brent handle it
        step = diff / vega
        sigma_new = sigma - step
        if not (sigma_lo < sigma_new < sigma_hi):
            break  # stepped out of bracket; hand off to Brent
        sigma = sigma_new

    # Brent fallback.
    def objective(s: float) -> float:
        return float(black76_price(forward, strike, t, s, df, is_call)) - market_price

    try:
        return float(brentq(objective, sigma_lo, sigma_hi, xtol=1e-12, maxiter=128))
    except (ValueError, RuntimeError):
        return math.nan


def _initial_guess(
    market_price: float,
    intrinsic: float,
    forward: float,
    t: float,
) -> float:
    """Brenner-Subrahmanyam seed on the time-value portion, clamped to [0.05, 3.0]."""
    time_value = max(market_price - intrinsic, 1e-12)
    sigma_0 = math.sqrt(2.0 * math.pi / t) * time_value / forward
    return min(max(sigma_0, 0.05), 3.0)
