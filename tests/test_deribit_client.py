"""Unit tests for the Deribit REST client (no DB, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from volsurface.config import Settings
from volsurface.ingestion.deribit_client import (
    DeribitApiError,
    _parse_strike_and_type,
    get_future_book_summary,
    get_future_instruments,
    get_option_book_summary,
    get_option_instruments,
    get_perpetual_ticker,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# ----- name parsing --------------------------------------------------------


def test_parse_strike_and_type_call() -> None:
    strike, opt = _parse_strike_and_type("BTC-23MAY26-68000-C")
    assert strike == 68000.0
    assert opt == "C"


def test_parse_strike_and_type_put() -> None:
    strike, opt = _parse_strike_and_type("BTC-23MAY26-69000-P")
    assert strike == 69000.0
    assert opt == "P"


def test_parse_strike_and_type_rejects_bad_count() -> None:
    with pytest.raises(DeribitApiError):
        _parse_strike_and_type("BTC-23MAY26-68000")


def test_parse_strike_and_type_rejects_bad_type() -> None:
    with pytest.raises(DeribitApiError):
        _parse_strike_and_type("BTC-23MAY26-68000-X")


def test_parse_strike_and_type_rejects_non_numeric_strike() -> None:
    with pytest.raises(DeribitApiError):
        _parse_strike_and_type("BTC-23MAY26-ABCDE-C")


# ----- fixture-backed REST calls -------------------------------------------


def _client_with(
    payloads: dict[tuple[str, str | None], dict],
) -> httpx.AsyncClient:
    """Build an AsyncClient whose transport returns fixture payloads.

    Routing key: (path_suffix, kind-param or None).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        kind = request.url.params.get("kind")
        for (suffix, k), body in payloads.items():
            if path.endswith(suffix) and (k is None or k == kind):
                return httpx.Response(200, json=body)
        return httpx.Response(404, json={"error": f"no fixture for {path} kind={kind}"})

    return httpx.AsyncClient(
        base_url=Settings().deribit_rest_url,
        transport=httpx.MockTransport(handler),
    )


async def test_get_option_instruments_parses_fixture() -> None:
    payloads = {
        ("/get_instruments", "option"): _load("deribit_get_instruments_btc_option.json"),
    }
    async with _client_with(payloads) as http:
        rows = await get_option_instruments(http)
    assert len(rows) == 5
    first = rows[0]
    assert first.instrument_name.startswith("BTC-")
    assert first.base_currency == "BTC"
    assert first.option_type in ("C", "P")
    assert first.strike > 0
    assert first.expiration_ts.tzinfo is not None


async def test_get_future_instruments_marks_perpetual() -> None:
    payloads = {
        ("/get_instruments", "future"): _load("deribit_get_instruments_btc_future.json"),
    }
    async with _client_with(payloads) as http:
        rows = await get_future_instruments(http)
    perps = [r for r in rows if r.is_perpetual]
    assert len(perps) == 1
    assert perps[0].instrument_name == "BTC-PERPETUAL"
    dated = [r for r in rows if not r.is_perpetual]
    assert len(dated) >= 3
    assert all(d.expiration_ts.tzinfo is not None for d in dated)


async def test_get_option_book_summary_passes_through_nullable_fields() -> None:
    raw = _load("deribit_book_summary_btc_option.json")
    # Force a null on the first entry to confirm pass-through.
    raw["result"][0]["bid_price"] = None
    raw["result"][0]["mark_iv"] = None
    payloads = {("/get_book_summary_by_currency", "option"): raw}
    async with _client_with(payloads) as http:
        rows = await get_option_book_summary(http)
    assert rows[0].bid_price is None
    assert rows[0].mark_iv is None


async def test_get_future_book_summary_drops_rows_missing_required_fields() -> None:
    raw = _load("deribit_book_summary_btc_future.json")
    n_complete = sum(
        1
        for r in raw["result"]
        if r.get("mark_price") is not None and r.get("estimated_delivery_price") is not None
    )
    # Inject a future with a null mark_price; it should be dropped.
    raw["result"].append(
        {
            "instrument_name": "BTC-FAKE-FUTURE",
            "mark_price": None,
            "estimated_delivery_price": 70000.0,
        }
    )
    payloads = {("/get_book_summary_by_currency", "future"): raw}
    async with _client_with(payloads) as http:
        rows = await get_future_book_summary(http)
    assert len(rows) == n_complete
    assert all(r.instrument_name != "BTC-FAKE-FUTURE" for r in rows)


async def test_get_perpetual_ticker_extracts_funding_and_index() -> None:
    payloads = {("/ticker", None): _load("deribit_ticker_btc_perpetual.json")}
    async with _client_with(payloads) as http:
        tkr = await get_perpetual_ticker(http)
    assert tkr.index_price > 0
    # funding_8h may legitimately be None; just check the field is reachable.
    assert tkr.funding_8h is None or isinstance(tkr.funding_8h, float)


async def test_raises_on_deribit_error_envelope() -> None:
    payloads = {
        ("/get_instruments", "option"): {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": 10000, "message": "boom"},
        },
    }
    async with _client_with(payloads) as http:
        with pytest.raises(DeribitApiError):
            await get_option_instruments(http)
