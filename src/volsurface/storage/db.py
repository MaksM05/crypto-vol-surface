"""Async storage layer for raw market observations.

This module owns all SQL access for the project. It exposes:

- An ``asyncpg`` connection pool factory, cached process-wide.
- Typed row containers that mirror ``storage/schema.sql`` column-for-column.
- One bulk upsert per ingestion table. All upserts are idempotent.

Conflict semantics
------------------

- ``instruments``        — ``ON CONFLICT DO UPDATE``. Static-ish metadata; rare
  corrections (contract size, creation timestamp) must land. ``last_seen`` is
  refreshed on every upsert so the poller's most recent cycle timestamp wins.
- ``option_quotes``      — ``ON CONFLICT DO NOTHING``. Raw observation, source
  of truth; replays must never overwrite a captured tick.
- ``forwards``           — ``ON CONFLICT DO NOTHING``. Same reasoning.
- ``funding_rates``      — ``ON CONFLICT DO NOTHING``. Same reasoning.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import astuple, dataclass
from datetime import datetime
from typing import Any, Literal

import asyncpg
from asyncpg.pool import PoolConnectionProxy

from volsurface.config import Settings

# Either a raw Connection (used directly) or a proxy returned by ``pool.acquire()``.
# Both implement the same execute/executemany/transaction surface used here.
# PEP 695 ``type`` keeps the RHS lazy — asyncpg.Connection isn't subscriptable at runtime.
type DbConn = asyncpg.Connection[asyncpg.Record] | PoolConnectionProxy[asyncpg.Record]

# ---------------------------------------------------------------------------
# Row types — mirror storage/schema.sql column-for-column.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstrumentRow:
    """One row of the ``instruments`` table.

    ``last_seen`` is the cycle timestamp of the most recent poll that observed
    this instrument. It is refreshed on every upsert (insert AND DO UPDATE) so
    a stalling ``last_seen`` flags a delisted / expired contract.
    """

    instrument_name: str
    currency: str
    strike: float
    option_type: Literal["C", "P"]
    expiry: datetime
    contract_size: float | None
    creation_ts: datetime | None
    last_seen: datetime


@dataclass(frozen=True, slots=True)
class OptionQuoteRow:
    """One row of the ``option_quotes`` table."""

    time: datetime
    instrument_name: str
    mark_price: float | None
    best_bid: float | None
    best_ask: float | None
    open_interest: float | None
    deribit_mark_iv: float | None
    deribit_delta: float | None


@dataclass(frozen=True, slots=True)
class ForwardRow:
    """One row of the ``forwards`` table."""

    time: datetime
    expiry: datetime
    forward_price: float
    index_price: float


@dataclass(frozen=True, slots=True)
class FundingRateRow:
    """One row of the ``funding_rates`` table."""

    time: datetime
    instrument_name: str
    funding_rate_8h: float | None
    index_price: float | None


@dataclass(frozen=True, slots=True)
class OptionQuoteWithMeta:
    """An ``option_quotes`` row joined to its ``instruments`` metadata.

    Returned by :func:`fetch_option_quotes_at` for read-side consumers
    (validation harness, analytics) that need strike/expiry/option_type
    alongside the quote.
    """

    time: datetime
    instrument_name: str
    mark_price: float | None
    best_bid: float | None
    best_ask: float | None
    open_interest: float | None
    deribit_mark_iv: float | None
    deribit_delta: float | None
    strike: float
    option_type: Literal["C", "P"]
    expiry: datetime


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------


_pool: asyncpg.Pool[asyncpg.Record] | None = None


async def get_pool(settings: Settings) -> asyncpg.Pool[asyncpg.Record]:
    """Return a process-wide ``asyncpg`` pool, creating it on first call.

    Parameters
    ----------
    settings
        Loaded :class:`volsurface.config.Settings`. The pool DSN is derived
        from ``settings.database_url`` and bounded by
        ``db_pool_min`` / ``db_pool_max``.
    """
    global _pool
    if _pool is None:
        pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
        )
        assert pool is not None  # asyncpg.create_pool returns None only when used as ctx mgr
        _pool = pool
    return _pool


async def close_pool() -> None:
    """Close the cached pool, if any. Safe to call when no pool exists."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Upserts — one per ingestion table.
# ---------------------------------------------------------------------------


_INSTRUMENT_UPSERT = """
INSERT INTO instruments (
    instrument_name, currency, strike, option_type,
    expiry, contract_size, creation_ts, last_seen
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (instrument_name) DO UPDATE SET
    currency      = EXCLUDED.currency,
    strike        = EXCLUDED.strike,
    option_type   = EXCLUDED.option_type,
    expiry        = EXCLUDED.expiry,
    contract_size = EXCLUDED.contract_size,
    creation_ts   = EXCLUDED.creation_ts,
    last_seen     = EXCLUDED.last_seen
"""

_OPTION_QUOTE_INSERT = """
INSERT INTO option_quotes (
    time, instrument_name, mark_price, best_bid, best_ask,
    open_interest, deribit_mark_iv, deribit_delta
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (instrument_name, time) DO NOTHING
"""

_FORWARD_INSERT = """
INSERT INTO forwards (time, expiry, forward_price, index_price)
VALUES ($1, $2, $3, $4)
ON CONFLICT (expiry, time) DO NOTHING
"""

_FUNDING_RATE_INSERT = """
INSERT INTO funding_rates (time, instrument_name, funding_rate_8h, index_price)
VALUES ($1, $2, $3, $4)
ON CONFLICT (instrument_name, time) DO NOTHING
"""


async def upsert_instruments(
    conn: DbConn,
    rows: Iterable[InstrumentRow],
) -> int:
    """Upsert rows into ``instruments``.

    Existing rows are overwritten field-by-field so metadata corrections land.
    Returns the number of rows submitted (zero if ``rows`` is empty).
    """
    batch: list[tuple[Any, ...]] = [astuple(r) for r in rows]
    if not batch:
        return 0
    async with conn.transaction():
        await conn.executemany(_INSTRUMENT_UPSERT, batch)
    return len(batch)


async def insert_option_quotes(
    conn: DbConn,
    rows: Iterable[OptionQuoteRow],
) -> int:
    """Insert rows into ``option_quotes``.

    Duplicates on ``(instrument_name, time)`` are silently skipped — raw
    observations are immutable. Returns the number of rows submitted.
    """
    batch: list[tuple[Any, ...]] = [astuple(r) for r in rows]
    if not batch:
        return 0
    async with conn.transaction():
        await conn.executemany(_OPTION_QUOTE_INSERT, batch)
    return len(batch)


async def insert_forwards(
    conn: DbConn,
    rows: Iterable[ForwardRow],
) -> int:
    """Insert rows into ``forwards``.

    Duplicates on ``(expiry, time)`` are silently skipped. Returns the number
    of rows submitted.
    """
    batch: list[tuple[Any, ...]] = [astuple(r) for r in rows]
    if not batch:
        return 0
    async with conn.transaction():
        await conn.executemany(_FORWARD_INSERT, batch)
    return len(batch)


async def insert_funding_rates(
    conn: DbConn,
    rows: Iterable[FundingRateRow],
) -> int:
    """Insert rows into ``funding_rates``.

    Duplicates on ``(instrument_name, time)`` are silently skipped. Returns
    the number of rows submitted.
    """
    batch: list[tuple[Any, ...]] = [astuple(r) for r in rows]
    if not batch:
        return 0
    async with conn.transaction():
        await conn.executemany(_FUNDING_RATE_INSERT, batch)
    return len(batch)


# ---------------------------------------------------------------------------
# Read helpers — used by the validation harness and future analytics.
# ---------------------------------------------------------------------------


async def get_forward(
    conn: DbConn,
    expiry: datetime,
    at_or_before: datetime,
) -> ForwardRow | None:
    """Return the most recent ``forwards`` row for ``expiry`` at or before ``at_or_before``.

    ``None`` if no forward has been recorded for that expiry up to the cutoff.
    """
    row = await conn.fetchrow(
        """
        SELECT time, expiry, forward_price, index_price
        FROM forwards
        WHERE expiry = $1 AND time <= $2
        ORDER BY time DESC
        LIMIT 1
        """,
        expiry,
        at_or_before,
    )
    if row is None:
        return None
    return ForwardRow(
        time=row["time"],
        expiry=row["expiry"],
        forward_price=row["forward_price"],
        index_price=row["index_price"],
    )


async def fetch_option_quotes_at(
    conn: DbConn,
    snapshot_time: datetime,
) -> list[OptionQuoteWithMeta]:
    """Return every ``option_quotes`` row at ``snapshot_time`` joined to ``instruments``.

    No filtering — callers decide what to drop (null prices, illiquid, etc.).
    """
    rows = await conn.fetch(
        """
        SELECT q.time, q.instrument_name, q.mark_price, q.best_bid, q.best_ask,
               q.open_interest, q.deribit_mark_iv, q.deribit_delta,
               i.strike, i.option_type, i.expiry
        FROM option_quotes q
        JOIN instruments i ON i.instrument_name = q.instrument_name
        WHERE q.time = $1
        """,
        snapshot_time,
    )
    return [
        OptionQuoteWithMeta(
            time=r["time"],
            instrument_name=r["instrument_name"],
            mark_price=r["mark_price"],
            best_bid=r["best_bid"],
            best_ask=r["best_ask"],
            open_interest=r["open_interest"],
            deribit_mark_iv=r["deribit_mark_iv"],
            deribit_delta=r["deribit_delta"],
            strike=r["strike"],
            option_type=r["option_type"],
            expiry=r["expiry"],
        )
        for r in rows
    ]
