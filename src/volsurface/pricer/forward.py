"""Forward-curve primitives. Pure types and helpers — no SQL, no I/O.

The actual ``forwards`` table read lives in ``storage/db.py`` per the
project's "DB access only through storage/" rule. This module exposes the
domain dataclass and the year-fraction helper the solver needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_SECONDS_PER_YEAR = 365.0 * 86400.0  # Act/365 — confirmed convention for the project


@dataclass(frozen=True, slots=True)
class ForwardSnapshot:
    """The forward + index pair for one (expiry, time).

    ``forward_price`` is the dated future's mark price, in USD/BTC; it is the
    F used inside Black-76. ``index_price`` is the BTC index Deribit uses for
    delivery and for its own ``mark_iv`` computation — it is the S used to
    convert a BTC-denominated option premium to USD before the IV solve.
    """

    time: datetime
    expiry: datetime
    forward_price: float
    index_price: float


def time_to_expiry_years(quote_time: datetime, expiry: datetime) -> float:
    """Act/365 year fraction between ``quote_time`` and ``expiry``.

    Negative if ``expiry`` has passed. The solver treats ``t <= 0`` as NaN —
    this helper does not clip; callers decide what to do with a non-positive
    value.
    """
    seconds = (expiry - quote_time).total_seconds()
    return seconds / _SECONDS_PER_YEAR
