"""Forward-based Black-76 pricer with analytic Greeks. Pure numpy, vectorised.

The functions in this module are unit-agnostic: pass USD-denominated forward,
strike, and discount factor and you get a USD price out. Conversion from the
BTC premium (Deribit's quote unit) to USD happens at the data boundary in
``validation/``, not here.

Formulas (forward-based Black-76):

    d1 = (ln(F/K) + 0.5 σ² T) / (σ √T)
    d2 = d1 − σ √T
    call_price = DF · (F · N(d1) − K · N(d2))
    put_price  = DF · (K · N(−d2) − F · N(−d1))
    vega   = DF · F · √T · n(d1)
    gamma  = DF · n(d1) / (F · σ · √T)
    theta  = DF · F · σ · n(d1) / (2 · √T)              # ∂price/∂T, same call/put
    standard_delta_call = DF · N(d1)
    standard_delta_put  = -DF · N(-d1)

Where ``N`` is the standard normal CDF and ``n`` is its pdf.

All functions broadcast across their numeric inputs via numpy. ``is_call`` is
broadcastable boolean. Inputs with ``T <= 0`` or ``sigma <= 0`` return the
intrinsic-value limit; ``T < 0`` raises ``ValueError`` because a negative
time-to-expiry is a caller bug, not market data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

type FloatArray = npt.NDArray[np.float64]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _norm_pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: FloatArray) -> FloatArray:
    # scipy.special.ndtr is faster than scipy.stats.norm.cdf and vectorises.
    return np.asarray(ndtr(x), dtype=np.float64)


def _d1_d2(
    forward: FloatArray,
    strike: FloatArray,
    t: FloatArray,
    sigma: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return (d1, d2). Caller guarantees t > 0 and sigma > 0 element-wise."""
    sqrt_t = np.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def _validate_inputs(t: FloatArray, sigma: FloatArray) -> None:
    if np.any(t < 0):
        raise ValueError("time-to-expiry must be non-negative")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative")


def black76_price(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    sigma: npt.ArrayLike,
    df: npt.ArrayLike,
    is_call: npt.ArrayLike,
) -> FloatArray:
    """Forward-based Black-76 call or put price.

    Parameters
    ----------
    forward, strike, t, sigma, df
        Broadcastable numeric arrays. ``forward`` and ``strike`` are USD/BTC,
        ``t`` is years (Act/365), ``sigma`` is decimal vol, ``df`` is the
        discount factor (use ``1.0`` to match Deribit's zero-rate convention).
    is_call
        Broadcastable boolean. ``True`` for call, ``False`` for put.

    Returns
    -------
    ndarray
        Option price in the forward's unit (USD).
    """
    f = np.asarray(forward, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    d = np.asarray(df, dtype=np.float64)
    call = np.asarray(is_call, dtype=bool)
    _validate_inputs(t_arr, s)

    # Intrinsic path for degenerate inputs (T = 0 or sigma = 0).
    intrinsic_call = np.maximum(f - k, 0.0) * d
    intrinsic_put = np.maximum(k - f, 0.0) * d
    intrinsic = np.where(call, intrinsic_call, intrinsic_put)
    degenerate = (t_arr == 0.0) | (s == 0.0)

    # Regular path. Use np.where on inputs first to avoid div-by-zero warnings.
    safe_t = np.where(degenerate, 1.0, t_arr)
    safe_s = np.where(degenerate, 1.0, s)
    d1, d2 = _d1_d2(f, k, safe_t, safe_s)
    call_price = d * (f * _norm_cdf(d1) - k * _norm_cdf(d2))
    put_price = d * (k * _norm_cdf(-d2) - f * _norm_cdf(-d1))
    price = np.where(call, call_price, put_price)

    return np.where(degenerate, intrinsic, price)


def black76_vega(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    sigma: npt.ArrayLike,
    df: npt.ArrayLike,
) -> FloatArray:
    """∂price/∂sigma. Same for calls and puts. Returns 0 at T = 0 or sigma = 0."""
    f = np.asarray(forward, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    d = np.asarray(df, dtype=np.float64)
    _validate_inputs(t_arr, s)

    degenerate = (t_arr == 0.0) | (s == 0.0)
    safe_t = np.where(degenerate, 1.0, t_arr)
    safe_s = np.where(degenerate, 1.0, s)
    d1, _ = _d1_d2(f, k, safe_t, safe_s)
    vega = d * f * np.sqrt(safe_t) * _norm_pdf(d1)
    return np.where(degenerate, 0.0, vega)


def black76_gamma(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    sigma: npt.ArrayLike,
    df: npt.ArrayLike,
) -> FloatArray:
    """∂²price/∂F². Same for calls and puts. Returns 0 at T = 0 or sigma = 0."""
    f = np.asarray(forward, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    d = np.asarray(df, dtype=np.float64)
    _validate_inputs(t_arr, s)

    degenerate = (t_arr == 0.0) | (s == 0.0)
    safe_t = np.where(degenerate, 1.0, t_arr)
    safe_s = np.where(degenerate, 1.0, s)
    d1, _ = _d1_d2(f, k, safe_t, safe_s)
    gamma = d * _norm_pdf(d1) / (f * safe_s * np.sqrt(safe_t))
    return np.where(degenerate, 0.0, gamma)


def black76_theta(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    sigma: npt.ArrayLike,
    df: npt.ArrayLike,
) -> FloatArray:
    """∂price/∂T (positive for normal options).

    Same for calls and puts under the Black-76 / zero-rate convention because
    the discount factor is treated as constant in T. The textbook "time decay"
    is ``-theta`` with this sign convention.
    """
    f = np.asarray(forward, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    d = np.asarray(df, dtype=np.float64)
    _validate_inputs(t_arr, s)

    degenerate = (t_arr == 0.0) | (s == 0.0)
    safe_t = np.where(degenerate, 1.0, t_arr)
    safe_s = np.where(degenerate, 1.0, s)
    d1, _ = _d1_d2(f, k, safe_t, safe_s)
    theta = d * f * safe_s * _norm_pdf(d1) / (2.0 * np.sqrt(safe_t))
    return np.where(degenerate, 0.0, theta)


def standard_delta(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    t: npt.ArrayLike,
    sigma: npt.ArrayLike,
    df: npt.ArrayLike,
    is_call: npt.ArrayLike,
) -> FloatArray:
    """Textbook Black-76 delta, ∂price/∂F.

    .. warning::

        This is NOT the correct delta for Deribit's coin-settled (inverse)
        BTC/ETH options. The Deribit-convention (inverse) delta is
        :func:`volsurface.pricer.inverse.inverse_delta`, which subtracts the
        ``C / S`` correction this function does not apply. Do not persist
        this value to ``computed_iv.my_delta`` and never read it as the
        hedge delta for a Deribit option. The name ``standard_delta``
        (rather than ``delta``) is deliberate — there is no top-level
        ``delta`` in this module so nothing can import the wrong number by
        accident.

    Returns
    -------
    ndarray
        ``DF · N(d1)`` for calls, ``-DF · N(-d1)`` for puts. Zero at T = 0
        out-of-the-money; intrinsic-sign at T = 0 in-the-money.
    """
    f = np.asarray(forward, dtype=np.float64)
    k = np.asarray(strike, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    d = np.asarray(df, dtype=np.float64)
    call = np.asarray(is_call, dtype=bool)
    _validate_inputs(t_arr, s)

    degenerate = (t_arr == 0.0) | (s == 0.0)
    safe_t = np.where(degenerate, 1.0, t_arr)
    safe_s = np.where(degenerate, 1.0, s)
    d1, _ = _d1_d2(f, k, safe_t, safe_s)
    delta_call = d * _norm_cdf(d1)
    delta_put = -d * _norm_cdf(-d1)
    delta = np.where(call, delta_call, delta_put)

    # Intrinsic-limit delta at degenerate inputs.
    in_the_money = np.where(call, f > k, f < k)
    degen_delta = np.where(in_the_money, np.where(call, d, -d), 0.0)
    return np.where(degenerate, degen_delta, delta)
