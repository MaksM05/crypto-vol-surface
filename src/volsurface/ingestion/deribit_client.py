"""Thin async wrapper around the Deribit public v2 REST endpoints used by ingestion.

Five calls cover the entire ingestion cycle:

- ``get_option_instruments``  — authoritative metadata for active BTC options.
- ``get_future_instruments``  — authoritative metadata for active BTC futures.
- ``get_option_book_summary`` — bulk quotes + ``mark_iv`` for all options.
- ``get_future_book_summary`` — bulk quotes + delivery-index price for all futures.
- ``get_perpetual_ticker``    — single ticker fetch for ``BTC-PERPETUAL`` funding.

The forward-curve expiry is always taken from ``expiration_timestamp`` returned
by ``get_instruments`` — never parsed from the instrument name (the name's
``DDMMMYY`` format and 08:00 UTC delivery convention are not contractual; a
wrong expiry silently corrupts time-to-expiry and the whole vol surface).

Strike and option-type *are* parsed from the name. Those fields are part of
the name by construction and parsing failure is a hard error, not a silent
default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

import httpx

from volsurface.config import Settings


class DeribitApiError(RuntimeError):
    """Raised on a non-success Deribit response or malformed payload."""


@dataclass(frozen=True, slots=True)
class InstrumentMeta:
    """Metadata for one option instrument, sourced from ``get_instruments``."""

    instrument_name: str
    base_currency: str
    strike: float
    option_type: Literal["C", "P"]
    expiration_ts: datetime
    contract_size: float
    creation_ts: datetime


@dataclass(frozen=True, slots=True)
class FutureMeta:
    """Metadata for one future instrument (dated or perpetual)."""

    instrument_name: str
    expiration_ts: datetime
    is_perpetual: bool


@dataclass(frozen=True, slots=True)
class OptionBookSummary:
    """Quote-side fields from ``get_book_summary_by_currency?kind=option``.

    All numeric fields are nullable — Deribit returns ``null`` for illiquid
    contracts (no bid, no last, no IV computable).
    """

    instrument_name: str
    mark_price: float | None
    bid_price: float | None
    ask_price: float | None
    open_interest: float | None
    mark_iv: float | None


@dataclass(frozen=True, slots=True)
class FutureBookSummary:
    """Forward-side fields from ``get_book_summary_by_currency?kind=future``.

    ``estimated_delivery_price`` is the BTC index value Deribit uses for
    delivery — same value across every future in one snapshot. Stored as the
    ``forwards.index_price`` column.
    """

    instrument_name: str
    mark_price: float
    estimated_delivery_price: float


@dataclass(frozen=True, slots=True)
class PerpetualTicker:
    """Subset of ``public/ticker`` for ``BTC-PERPETUAL`` used for funding."""

    funding_8h: float | None
    index_price: float


def _parse_strike_and_type(instrument_name: str) -> tuple[float, Literal["C", "P"]]:
    """Extract strike and option type from ``BTC-23MAY26-68000-C``.

    The name is the only source for these two fields, and Deribit's format is
    stable. Parsing failure raises ``DeribitApiError`` rather than guessing.
    """
    parts = instrument_name.split("-")
    if len(parts) != 4:
        raise DeribitApiError(
            f"unexpected option instrument name {instrument_name!r}: "
            f"want 4 hyphen-separated parts, got {len(parts)}"
        )
    _currency, _expiry_str, strike_str, type_str = parts
    if type_str not in ("C", "P"):
        raise DeribitApiError(f"unexpected option type {type_str!r} in {instrument_name!r}")
    try:
        strike = float(strike_str)
    except ValueError as exc:
        raise DeribitApiError(f"non-numeric strike {strike_str!r} in {instrument_name!r}") from exc
    return strike, cast(Literal["C", "P"], type_str)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


async def _get(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET ``path`` on the Deribit base URL, return the ``result`` field."""
    resp = await client.get(path, params=params)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        raise DeribitApiError(f"deribit error on {path}: {body['error']!r}")
    if "result" not in body:
        raise DeribitApiError(f"deribit response missing 'result' on {path}: {body!r}")
    return body["result"]


async def get_option_instruments(client: httpx.AsyncClient) -> list[InstrumentMeta]:
    """Fetch active BTC option instruments. One bulk call."""
    raw = await _get(
        client,
        "/public/get_instruments",
        params={"currency": "BTC", "kind": "option", "expired": "false"},
    )
    out: list[InstrumentMeta] = []
    for r in raw:
        name = r["instrument_name"]
        strike, opt_type = _parse_strike_and_type(name)
        out.append(
            InstrumentMeta(
                instrument_name=name,
                base_currency=r["base_currency"],
                strike=strike,
                option_type=opt_type,
                expiration_ts=_ms_to_dt(r["expiration_timestamp"]),
                contract_size=float(r["contract_size"]),
                creation_ts=_ms_to_dt(r["creation_timestamp"]),
            )
        )
    return out


async def get_future_instruments(client: httpx.AsyncClient) -> list[FutureMeta]:
    """Fetch active BTC futures (dated and perpetual). One bulk call.

    Used to obtain authoritative ``expiration_timestamp`` for every dated
    future, which is the forward-curve expiry written to the ``forwards`` table.
    """
    raw = await _get(
        client,
        "/public/get_instruments",
        params={"currency": "BTC", "kind": "future", "expired": "false"},
    )
    out: list[FutureMeta] = []
    for r in raw:
        name = r["instrument_name"]
        out.append(
            FutureMeta(
                instrument_name=name,
                expiration_ts=_ms_to_dt(r["expiration_timestamp"]),
                is_perpetual=name.endswith("-PERPETUAL"),
            )
        )
    return out


async def get_option_book_summary(
    client: httpx.AsyncClient,
) -> list[OptionBookSummary]:
    """Bulk quotes + ``mark_iv`` for every active BTC option. One call.

    Greeks/delta are *not* returned by this endpoint — Deribit only exposes
    them on the per-instrument ``ticker``. ``deribit_delta`` therefore stays
    ``NULL`` in v1 (the pricer module computes delta itself).
    """
    raw = await _get(
        client,
        "/public/get_book_summary_by_currency",
        params={"currency": "BTC", "kind": "option"},
    )
    return [
        OptionBookSummary(
            instrument_name=r["instrument_name"],
            mark_price=r.get("mark_price"),
            bid_price=r.get("bid_price"),
            ask_price=r.get("ask_price"),
            open_interest=r.get("open_interest"),
            mark_iv=r.get("mark_iv"),
        )
        for r in raw
    ]


async def get_future_book_summary(
    client: httpx.AsyncClient,
) -> list[FutureBookSummary]:
    """Bulk forwards + delivery-index for every active BTC future. One call.

    Entries with missing ``mark_price`` or ``estimated_delivery_price`` are
    silently dropped — a future without those two values is unusable for the
    forward curve.
    """
    raw = await _get(
        client,
        "/public/get_book_summary_by_currency",
        params={"currency": "BTC", "kind": "future"},
    )
    out: list[FutureBookSummary] = []
    for r in raw:
        mark = r.get("mark_price")
        edp = r.get("estimated_delivery_price")
        if mark is None or edp is None:
            continue
        out.append(
            FutureBookSummary(
                instrument_name=r["instrument_name"],
                mark_price=float(mark),
                estimated_delivery_price=float(edp),
            )
        )
    return out


async def get_perpetual_ticker(
    client: httpx.AsyncClient,
    instrument_name: str = "BTC-PERPETUAL",
) -> PerpetualTicker:
    """Fetch ``funding_8h`` and ``index_price`` for the perpetual."""
    r = await _get(client, "/public/ticker", params={"instrument_name": instrument_name})
    return PerpetualTicker(
        funding_8h=r.get("funding_8h"),
        index_price=float(r["index_price"]),
    )


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    """Construct an ``httpx.AsyncClient`` configured for Deribit REST.

    Caller is responsible for the ``async with`` lifecycle.
    """
    return httpx.AsyncClient(
        base_url=settings.deribit_rest_url,
        timeout=settings.deribit_http_timeout_s,
    )
