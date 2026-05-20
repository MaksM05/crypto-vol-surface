"""Async storage layer for raw market observations.

This module owns all SQL access for the project. It exposes:

- An ``asyncpg`` connection pool factory, cached process-wide.
- Typed row containers that mirror ``storage/schema.sql`` column-for-column.
- One bulk upsert per ingestion table. All upserts are idempotent.

Conflict semantics
------------------

- ``instruments``        — ``ON CONFLICT DO UPDATE``. Static-ish metadata; rare
  corrections (contract size, creation timestamp) must land.
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

from volsurface.config import Settings

# ---------------------------------------------------------------------------
# Row types — mirror storage/schema.sql column-for-column.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InstrumentRow:
    """One row of the ``instruments`` table."""

    instrument_name: str
    currency: str
    strike: float
    option_type: Literal["C", "P"]
    expiry: datetime
    contract_size: float | None
    creation_ts: datetime | None


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
    expiry, contract_size, creation_ts
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (instrument_name) DO UPDATE SET
    currency      = EXCLUDED.currency,
    strike        = EXCLUDED.strike,
    option_type   = EXCLUDED.option_type,
    expiry        = EXCLUDED.expiry,
    contract_size = EXCLUDED.contract_size,
    creation_ts   = EXCLUDED.creation_ts
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
    conn: asyncpg.Connection[asyncpg.Record],
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
    conn: asyncpg.Connection[asyncpg.Record],
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
    conn: asyncpg.Connection[asyncpg.Record],
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
    conn: asyncpg.Connection[asyncpg.Record],
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
