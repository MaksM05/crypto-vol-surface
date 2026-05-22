"""Integration tests for the REST poller.

Mocks Deribit via ``httpx.MockTransport`` backed by captured fixtures, runs the
poller against the real TimescaleDB, and verifies what landed in each table.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest

from volsurface.config import Settings
from volsurface.ingestion.rest_poller import run_one_cycle

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def _default_payloads() -> dict[str, dict[str, Any]]:
    return {
        "instruments_option": _load("deribit_get_instruments_btc_option.json"),
        "instruments_future": _load("deribit_get_instruments_btc_future.json"),
        "book_option": _load("deribit_book_summary_btc_option.json"),
        "book_future": _load("deribit_book_summary_btc_future.json"),
        "ticker_perp": _load("deribit_ticker_btc_perpetual.json"),
    }


def _make_transport(payloads: dict[str, dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        kind = request.url.params.get("kind")
        if path.endswith("/get_instruments") and kind == "option":
            return httpx.Response(200, json=payloads["instruments_option"])
        if path.endswith("/get_instruments") and kind == "future":
            return httpx.Response(200, json=payloads["instruments_future"])
        if path.endswith("/get_book_summary_by_currency") and kind == "option":
            return httpx.Response(200, json=payloads["book_option"])
        if path.endswith("/get_book_summary_by_currency") and kind == "future":
            return httpx.Response(200, json=payloads["book_future"])
        if path.endswith("/ticker"):
            return httpx.Response(200, json=payloads["ticker_perp"])
        return httpx.Response(404, json={"error": f"no fixture for {path}"})

    return httpx.MockTransport(handler)


async def _run_one(
    settings: Settings,
    pool: asyncpg.Pool,
    mutate: Callable[[dict[str, dict[str, Any]]], None] | None = None,
):
    payloads = _default_payloads()
    if mutate is not None:
        mutate(payloads)
    transport = _make_transport(payloads)
    async with httpx.AsyncClient(base_url=settings.deribit_rest_url, transport=transport) as http:
        return await run_one_cycle(http, pool)


# ---------------------------------------------------------------------------


async def test_run_one_cycle_populates_all_tables(
    pool: asyncpg.Pool, db_cleanup: None, settings: Settings
) -> None:
    stats = await _run_one(settings, pool)

    assert stats.instruments == 5  # five option instruments in the fixture
    assert stats.option_quotes == 5
    assert stats.quotes_dropped_fk == 0
    assert stats.forwards >= 3  # several dated futures in the fixture
    assert stats.funding_rates == 1

    async with pool.acquire() as conn:
        n_instr = await conn.fetchval("SELECT count(*) FROM instruments")
        n_q = await conn.fetchval("SELECT count(*) FROM option_quotes")
        n_f = await conn.fetchval("SELECT count(*) FROM forwards")
        n_fund = await conn.fetchval("SELECT count(*) FROM funding_rates")
        deltas_null = await conn.fetchval(
            "SELECT bool_and(deribit_delta IS NULL) FROM option_quotes"
        )
        last_seen_set = await conn.fetchval(
            "SELECT bool_and(last_seen IS NOT NULL) FROM instruments"
        )
        # forwards.index_price == every future's estimated_delivery_price,
        # which is the BTC index — all dated futures share it within a snapshot.
        distinct_idx = await conn.fetchval("SELECT count(DISTINCT index_price) FROM forwards")

    assert n_instr == 5
    assert n_q == 5
    assert n_f == stats.forwards
    assert n_fund == 1
    assert deltas_null is True, "Q3: deribit_delta must be NULL in v1 REST cycles"
    assert last_seen_set is True
    assert distinct_idx == 1, (
        "estimated_delivery_price is the same BTC index across all futures in a snapshot"
    )


async def test_run_one_cycle_drops_quote_for_unknown_instrument(
    pool: asyncpg.Pool, db_cleanup: None, settings: Settings
) -> None:
    def add_zombie(p: dict[str, dict[str, Any]]) -> None:
        p["book_option"]["result"].append(
            {
                "instrument_name": "BTC-ZOMBIE-99999-C",
                "mark_price": 0.1,
                "bid_price": 0.09,
                "ask_price": 0.11,
                "open_interest": 0.0,
                "mark_iv": 50.0,
            }
        )

    stats = await _run_one(settings, pool, mutate=add_zombie)
    assert stats.quotes_dropped_fk == 1
    assert stats.option_quotes == 5  # zombie not written

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT instrument_name FROM option_quotes WHERE instrument_name = 'BTC-ZOMBIE-99999-C'"
        )
    assert rows == []


async def test_run_one_cycle_handles_null_iv_and_quotes(
    pool: asyncpg.Pool, db_cleanup: None, settings: Settings
) -> None:
    target_name = _default_payloads()["book_option"]["result"][0]["instrument_name"]

    def null_out_illiquid(p: dict[str, dict[str, Any]]) -> None:
        first = p["book_option"]["result"][0]
        first["bid_price"] = None
        first["ask_price"] = None
        first["mark_iv"] = None

    stats = await _run_one(settings, pool, mutate=null_out_illiquid)
    assert stats.option_quotes == 5  # NOT dropped — illiquid rows still land

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT best_bid, best_ask, deribit_mark_iv FROM option_quotes "
            "WHERE instrument_name = $1",
            target_name,
        )
    assert row is not None
    assert row["best_bid"] is None
    assert row["best_ask"] is None
    assert row["deribit_mark_iv"] is None


async def test_run_one_cycle_refreshes_last_seen(
    pool: asyncpg.Pool, db_cleanup: None, settings: Settings
) -> None:
    first = await _run_one(settings, pool)
    await asyncio.sleep(1.1)  # force a measurable cycle_time delta
    second = await _run_one(settings, pool)

    async with pool.acquire() as conn:
        max_last = await conn.fetchval("SELECT max(last_seen) FROM instruments")
        min_last = await conn.fetchval("SELECT min(last_seen) FROM instruments")

    assert second.cycle_time > first.cycle_time
    assert max_last == second.cycle_time
    assert min_last == second.cycle_time, (
        "every instrument seen this cycle must have last_seen refreshed"
    )


async def test_run_one_cycle_forward_uses_authoritative_expiry(
    pool: asyncpg.Pool, db_cleanup: None, settings: Settings
) -> None:
    """Forward expiry must come from get_instruments.expiration_timestamp,
    NOT parsed from the instrument_name string."""
    stats = await _run_one(settings, pool)

    fut_meta = {
        r["instrument_name"]: r["expiration_timestamp"]
        for r in _default_payloads()["instruments_future"]["result"]
    }

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT expiry, forward_price FROM forwards ORDER BY expiry")

    assert len(rows) == stats.forwards
    # Every forward's expiry must match exactly one dated future's
    # expiration_timestamp from the fixture (millisecond precision).
    expected_expiries_ms = {ts for name, ts in fut_meta.items() if not name.endswith("-PERPETUAL")}
    actual_expiries_ms = {int(r["expiry"].timestamp() * 1000) for r in rows}
    assert actual_expiries_ms <= expected_expiries_ms
