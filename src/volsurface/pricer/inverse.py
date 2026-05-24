"""Inverse-contract (Deribit-convention) delta for coin-settled BTC/ETH options.

This is the correct delta for hedging on Deribit. The textbook Black-76
``standard_delta`` lives next door in ``black76.py`` and is loudly documented
as the wrong thing to ship for these contracts.

Derivation
----------

Setup: ``C_usd`` is the Black-76 USD price (per 1 BTC notional, what our
``black76_price`` returns). ``C_btc = C_usd / S`` is the BTC-denominated
quote Deribit publishes (``mark_price``). ``S`` is the BTC index, USD/BTC.

The hedge ratio used in coin-collateralised books is ``dC_btc/dS · S``
(BTC moved per BTC-of-spot moved). With the project-wide zero-rate
convention ``DF = 1`` (so ``F = S``), apply the quotient rule to
``C_btc = C_usd / S``:

    dC_btc/dS  =  (δ_bs · S − C_usd) / S²

and multiply by ``S``:

    inverse_delta  =  dC_btc/dS · S  =  δ_bs − C_usd / S  =  δ_bs − C_btc

For any long position ``C_usd ≥ 0`` so the correction is always subtractive —
not because of a directional sign rule, but because ``d(1/S)/dS = −1/S²``
introduces the minus sign mechanically.

Sanity (recovered by the unit tests):

- Deep-ITM call: ``δ_bs → 1``, ``C_usd → S``, so ``inverse_delta → 0``.
  Inverse-call delta saturates at zero — a coin-settled call's BTC payoff is
  bounded above by 1 BTC.
- Deep-OTM call: ``C_usd → 0``, so ``inverse_delta → δ_bs``. Correction
  vanishes where there is no premium to re-price.
- Deep-ITM put: ``δ_bs → −1``, ``C_usd → K > S``, so ``inverse_delta < −1``.
  Inverse-put delta is unbounded below as ``S → 0``.

LOUD ASSUMPTION — DF = 1
------------------------

This formula assumes the zero-rate / ``F = S`` convention used everywhere
else in this project (the same convention that makes our IV harness match
Deribit's published ``mark_iv``). There is no ``df`` parameter on purpose:
a non-trivial discount factor introduces a ``dF/dS = 1/DF`` chain-rule
factor on ``δ_bs`` that this function does not apply. If a future caller
ever needs ``DF ≠ 1``, this is an explicit boundary to extend at — not a
silent assumption to dig out later.

Known limitation
----------------

Inverse delta is currently validated only against first-principles limits
(saturation, OTM convergence, |δ| > 1 for deep-ITM puts) and the closed-form
formula itself. It is NOT yet validated against Deribit's published
``ticker.delta`` because our ingester leaves ``deribit_delta`` ``NULL``
(per the REST-poller Q3 decision). A future slice will fetch the ticker
greeks for a small fixture and add a like-for-like comparison.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


def inverse_delta(
    standard_delta: npt.ArrayLike,
    price: npt.ArrayLike,
    spot: npt.ArrayLike,
) -> FloatArray:
    """Deribit-convention delta for coin-settled BTC/ETH options.

    ``inverse_delta = standard_delta − price / spot``

    Parameters
    ----------
    standard_delta
        Textbook Black-76 delta, broadcastable. Use
        :func:`volsurface.pricer.black76.standard_delta`.
    price
        Option price in **USD**, broadcastable. Use
        :func:`volsurface.pricer.black76.black76_price` (the conversion
        from Deribit's BTC ``mark_price`` to USD is the harness's job).
    spot
        BTC index ``S`` in USD/BTC, broadcastable. Must be the same ``S``
        the IV harness uses for the BTC→USD conversion
        (``forwards.index_price``), not the forward.

    Returns
    -------
    ndarray
        Dimensionless inverse delta (BTC per BTC of spot move, the
        Deribit / coin-book convention).

    Raises
    ------
    ValueError
        If any element of ``spot`` is non-positive — a zero or negative
        BTC index is a caller bug, not market data.
    """
    delta_arr = np.asarray(standard_delta, dtype=np.float64)
    price_arr = np.asarray(price, dtype=np.float64)
    spot_arr = np.asarray(spot, dtype=np.float64)
    if np.any(spot_arr <= 0.0):
        raise ValueError("spot must be positive")
    return delta_arr - price_arr / spot_arr
