"""Integration tests for the storage layer.

Require a running TimescaleDB (see ``docker-compose.yml``). Marked
``integration`` so they can be filtered out where a database is unavailable.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg
import pytest

from volsurface.storage import (
    ForwardRow,
    FundingRateRow,
    InstrumentRow,
    OptionQuoteRow,
    insert_forwards,
    insert_funding_rates,
    insert_option_quotes,
    upsert_instruments,
)

pytestmark = pytest.mark.integration


def _instrument(
    name: str = "BTC-26JUN26-77000-C",
    strike: float = 77000.0,
    option_type: str = "C",
) -> InstrumentRow:
    return InstrumentRow(
        instrument_name=name,
        currency="BTC",
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        expiry=datetime(2026, 6, 26, 8, 0, tzinfo=UTC),
        contract_size=1.0,
        creation_ts=datetime(2026, 1, 1, tzinfo=UTC),
    )


# ----- instruments ---------------------------------------------------------


async def test_upsert_instruments_roundtrip(pool: asyncpg.Pool, db_cleanup: None) -> None:
    rows = [
        _instrument("BTC-26JUN26-77000-C", 77000.0, "C"),
        _instrument("BTC-26JUN26-78000-C", 78000.0, "C"),
        _instrument("BTC-26JUN26-79000-P", 79000.0, "P"),
    ]
    async with pool.acquire() as conn:
        written = await upsert_instruments(conn, rows)
        fetched = await conn.fetch(
            "SELECT instrument_name, strike, option_type FROM instruments ORDER BY strike"
        )
    assert written == 3
    assert [(r["instrument_name"], r["strike"], r["option_type"]) for r in fetched] == [
        ("BTC-26JUN26-77000-C", 77000.0, "C"),
        ("BTC-26JUN26-78000-C", 78000.0, "C"),
        ("BTC-26JUN26-79000-P", 79000.0, "P"),
    ]


async def test_upsert_instruments_idempotent_updates_metadata(
    pool: asyncpg.Pool, db_cleanup: None
) -> None:
    first = _instrument()
    corrected = InstrumentRow(
        instrument_name=first.instrument_name,
        currency=first.currency,
        strike=first.strike,
        option_type=first.option_type,
        expiry=first.expiry,
        contract_size=2.5,  # the correction
        creation_ts=first.creation_ts,
    )
    async with pool.acquire() as conn:
        await upsert_instruments(conn, [first])
        await upsert_instruments(conn, [corrected])
        rows = await conn.fetch("SELECT instrument_name, contract_size FROM instruments")
    assert len(rows) == 1
    assert rows[0]["contract_size"] == 2.5


async def test_upsert_instruments_empty_is_noop(pool: asyncpg.Pool, db_cleanup: None) -> None:
    async with pool.acquire() as conn:
        written = await upsert_instruments(conn, [])
        count = await conn.fetchval("SELECT count(*) FROM instruments")
    assert written == 0
    assert count == 0


# ----- option_quotes -------------------------------------------------------


async def test_insert_option_quotes_roundtrip(pool: asyncpg.Pool, db_cleanup: None) -> None:
    async with pool.acquire() as conn:
        await upsert_instruments(conn, [_instrument()])
        quotes = [
            OptionQuoteRow(
                time=datetime(2026, 5, 20, 10, i, tzinfo=UTC),
                instrument_name="BTC-26JUN26-77000-C",
                mark_price=0.05 + 0.001 * i,
                best_bid=0.049,
                best_ask=0.051,
                open_interest=42.0,
                deribit_mark_iv=0.65,
                deribit_delta=0.42,
            )
            for i in range(5)
        ]
        written = await insert_option_quotes(conn, quotes)
        fetched = await conn.fetch("SELECT time, mark_price FROM option_quotes ORDER BY time")
    assert written == 5
    assert [r["mark_price"] for r in fetched] == [q.mark_price for q in quotes]


async def test_insert_option_quotes_idempotent_does_not_overwrite(
    pool: asyncpg.Pool, db_cleanup: None
) -> None:
    t = datetime(2026, 5, 20, 10, 0, tzinfo=UTC)
    original = OptionQuoteRow(
        time=t,
        instrument_name="BTC-26JUN26-77000-C",
        mark_price=0.05,
        best_bid=0.049,
        best_ask=0.051,
        open_interest=10.0,
        deribit_mark_iv=0.6,
        deribit_delta=0.4,
    )
    overwrite_attempt = OptionQuoteRow(
        time=t,
        instrument_name="BTC-26JUN26-77000-C",
        mark_price=99.0,  # would be a data-integrity disaster
        best_bid=99.0,
        best_ask=99.0,
        open_interest=99.0,
        deribit_mark_iv=99.0,
        deribit_delta=99.0,
    )
    async with pool.acquire() as conn:
        await upsert_instruments(conn, [_instrument()])
        await insert_option_quotes(conn, [original])
        await insert_option_quotes(conn, [overwrite_attempt])
        rows = await conn.fetch("SELECT mark_price FROM option_quotes")
    assert len(rows) == 1, "duplicate (instrument, time) must not create a second row"
    assert rows[0]["mark_price"] == 0.05, "raw observation must not be overwritten on replay"


async def test_insert_option_quotes_requires_instrument_fk(
    pool: asyncpg.Pool, db_cleanup: None
) -> None:
    orphan = OptionQuoteRow(
        time=datetime(2026, 5, 20, 10, 0, tzinfo=UTC),
        instrument_name="BTC-DOES-NOT-EXIST",
        mark_price=0.05,
        best_bid=None,
        best_ask=None,
        open_interest=None,
        deribit_mark_iv=None,
        deribit_delta=None,
    )
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await insert_option_quotes(conn, [orphan])


# ----- forwards ------------------------------------------------------------


async def test_insert_forwards_roundtrip(pool: asyncpg.Pool, db_cleanup: None) -> None:
    expiry = datetime(2026, 6, 26, 8, 0, tzinfo=UTC)
    rows = [
        ForwardRow(
            time=datetime(2026, 5, 20, 10, i, tzinfo=UTC),
            expiry=expiry,
            forward_price=70000.0 + i,
            index_price=69500.0 + i,
        )
        for i in range(3)
    ]
    async with pool.acquire() as conn:
        written = await insert_forwards(conn, rows)
        fetched = await conn.fetch("SELECT time, forward_price FROM forwards ORDER BY time")
    assert written == 3
    assert [r["forward_price"] for r in fetched] == [70000.0, 70001.0, 70002.0]


async def test_insert_forwards_idempotent(pool: asyncpg.Pool, db_cleanup: None) -> None:
    row = ForwardRow(
        time=datetime(2026, 5, 20, 10, 0, tzinfo=UTC),
        expiry=datetime(2026, 6, 26, 8, 0, tzinfo=UTC),
        forward_price=70000.0,
        index_price=69500.0,
    )
    overwrite = ForwardRow(
        time=row.time,
        expiry=row.expiry,
        forward_price=99999.0,
        index_price=99999.0,
    )
    async with pool.acquire() as conn:
        await insert_forwards(conn, [row])
        await insert_forwards(conn, [overwrite])
        fetched = await conn.fetch("SELECT forward_price FROM forwards")
    assert len(fetched) == 1
    assert fetched[0]["forward_price"] == 70000.0


# ----- funding_rates -------------------------------------------------------


async def test_insert_funding_rates_roundtrip(pool: asyncpg.Pool, db_cleanup: None) -> None:
    rows = [
        FundingRateRow(
            time=datetime(2026, 5, 20, h, 0, tzinfo=UTC),
            instrument_name="BTC-PERPETUAL",
            funding_rate_8h=0.0001 * h,
            index_price=69500.0,
        )
        for h in range(3)
    ]
    async with pool.acquire() as conn:
        written = await insert_funding_rates(conn, rows)
        fetched = await conn.fetch("SELECT funding_rate_8h FROM funding_rates ORDER BY time")
    assert written == 3
    assert [r["funding_rate_8h"] for r in fetched] == [
        pytest.approx(0.0),
        pytest.approx(0.0001),
        pytest.approx(0.0002),
    ]


async def test_insert_funding_rates_idempotent(pool: asyncpg.Pool, db_cleanup: None) -> None:
    row = FundingRateRow(
        time=datetime(2026, 5, 20, 10, 0, tzinfo=UTC),
        instrument_name="BTC-PERPETUAL",
        funding_rate_8h=0.0001,
        index_price=69500.0,
    )
    overwrite = FundingRateRow(
        time=row.time,
        instrument_name=row.instrument_name,
        funding_rate_8h=0.999,
        index_price=99999.0,
    )
    async with pool.acquire() as conn:
        await insert_funding_rates(conn, [row])
        await insert_funding_rates(conn, [overwrite])
        fetched = await conn.fetch("SELECT funding_rate_8h FROM funding_rates")
    assert len(fetched) == 1
    assert fetched[0]["funding_rate_8h"] == pytest.approx(0.0001)
