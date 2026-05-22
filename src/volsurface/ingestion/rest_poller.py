"""5-minute REST poller — sole DB writer for the ingestion tables.

Per-cycle work, in order:

1. ``get_instruments?kind=option``   → upsert ``instruments`` with ``last_seen = cycle_time``.
2. ``get_instruments?kind=future``   → map ``{future_name → expiration_ts}`` for forwards.
3. ``get_book_summary?kind=option``  → insert ``option_quotes`` (drop any quote whose
   instrument is not in step 1 to avoid FK violations during a midnight listing roll).
   ``deribit_delta`` is left ``NULL`` — book_summary doesn't carry greeks.
4. ``get_book_summary?kind=future``  → insert ``forwards`` for every dated future,
   using ``mark_price`` as forward and ``estimated_delivery_price`` as index.
5. ``ticker?instrument=BTC-PERPETUAL`` → insert one ``funding_rates`` row.

WebSocket subscriber is in-memory only; the REST poller is the only writer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import asyncpg
import httpx

from volsurface.config import Settings
from volsurface.ingestion.deribit_client import (
    get_future_book_summary,
    get_future_instruments,
    get_option_book_summary,
    get_option_instruments,
    get_perpetual_ticker,
)
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

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CycleStats:
    """Per-cycle row counts and the instrument universe written this cycle."""

    cycle_time: datetime
    instruments: int
    option_quotes: int
    quotes_dropped_fk: int
    forwards: int
    funding_rates: int
    instrument_universe: frozenset[str]


async def run_one_cycle(
    http: httpx.AsyncClient,
    pool: asyncpg.Pool[asyncpg.Record],
) -> CycleStats:
    """Run a single poll cycle. Called by ``run_forever`` and by tests.

    Parameters
    ----------
    http
        An open ``httpx.AsyncClient`` pointing at the Deribit base URL.
        Tests inject one backed by ``httpx.MockTransport``.
    pool
        Asyncpg pool. The cycle acquires a single connection for all four
        bulk writes after every HTTP call has completed.
    """
    cycle_time = datetime.now(UTC).replace(microsecond=0)

    # 1) Option instruments — authoritative universe for this cycle.
    option_meta = await get_option_instruments(http)
    instrument_rows = [
        InstrumentRow(
            instrument_name=m.instrument_name,
            currency=m.base_currency,
            strike=m.strike,
            option_type=m.option_type,
            expiry=m.expiration_ts,
            contract_size=m.contract_size,
            creation_ts=m.creation_ts,
            last_seen=cycle_time,
        )
        for m in option_meta
    ]
    known_instruments: frozenset[str] = frozenset(m.instrument_name for m in option_meta)

    # 2) Future instruments — authoritative expiry source for dated futures.
    future_meta = await get_future_instruments(http)
    expiry_by_future: dict[str, datetime] = {
        fm.instrument_name: fm.expiration_ts for fm in future_meta if not fm.is_perpetual
    }

    # 3) Option book summary → option_quotes (deribit_delta stays NULL in v1).
    option_quotes_raw = await get_option_book_summary(http)
    quote_rows: list[OptionQuoteRow] = []
    dropped_fk = 0
    for q in option_quotes_raw:
        if q.instrument_name not in known_instruments:
            dropped_fk += 1
            continue
        quote_rows.append(
            OptionQuoteRow(
                time=cycle_time,
                instrument_name=q.instrument_name,
                mark_price=q.mark_price,
                best_bid=q.bid_price,
                best_ask=q.ask_price,
                open_interest=q.open_interest,
                deribit_mark_iv=q.mark_iv,
                deribit_delta=None,
            )
        )
    if dropped_fk:
        log.warning(
            "cycle %s: dropped %d option quotes for unknown instruments",
            cycle_time.isoformat(),
            dropped_fk,
        )

    # 4) Future book summary → forwards (skip perpetual; expiry from step 2).
    future_quotes = await get_future_book_summary(http)
    forward_rows: list[ForwardRow] = []
    for f in future_quotes:
        if f.instrument_name.endswith("-PERPETUAL"):
            continue
        expiry = expiry_by_future.get(f.instrument_name)
        if expiry is None:
            log.warning(
                "cycle %s: future %s present in book_summary but missing from "
                "get_instruments; skipping",
                cycle_time.isoformat(),
                f.instrument_name,
            )
            continue
        forward_rows.append(
            ForwardRow(
                time=cycle_time,
                expiry=expiry,
                forward_price=f.mark_price,
                index_price=f.estimated_delivery_price,
            )
        )

    # 5) Perp ticker → funding_rates (single row).
    perp = await get_perpetual_ticker(http)
    funding_rows = [
        FundingRateRow(
            time=cycle_time,
            instrument_name="BTC-PERPETUAL",
            funding_rate_8h=perp.funding_8h,
            index_price=perp.index_price,
        )
    ]

    # Bulk writes: instruments first (FK target for option_quotes), then the rest.
    async with pool.acquire() as conn:
        n_instr = await upsert_instruments(conn, instrument_rows)
        n_quotes = await insert_option_quotes(conn, quote_rows)
        n_forwards = await insert_forwards(conn, forward_rows)
        n_funding = await insert_funding_rates(conn, funding_rows)

    stats = CycleStats(
        cycle_time=cycle_time,
        instruments=n_instr,
        option_quotes=n_quotes,
        quotes_dropped_fk=dropped_fk,
        forwards=n_forwards,
        funding_rates=n_funding,
        instrument_universe=known_instruments,
    )
    log.info(
        "cycle %s: instruments=%d quotes=%d dropped=%d forwards=%d funding=%d",
        cycle_time.isoformat(),
        n_instr,
        n_quotes,
        dropped_fk,
        n_forwards,
        n_funding,
    )
    return stats


async def run_forever(
    settings: Settings,
    http: httpx.AsyncClient,
    pool: asyncpg.Pool[asyncpg.Record],
    on_cycle: Callable[[CycleStats], Awaitable[None]] | None = None,
) -> None:
    """Run cycles forever at ``settings.poll_interval_s`` cadence.

    Sleeps ``poll_interval_s - elapsed`` between cycles. Exceptions inside a
    cycle propagate out — the caller's ``TaskGroup`` (see ``__main__``) cancels
    the WebSocket task and the process fails loudly.

    Parameters
    ----------
    on_cycle
        Optional callback invoked after each successful cycle. Used by
        ``__main__`` to hand the latest instrument universe to the WebSocket
        subscriber so it can reconcile its subscription set.
    """
    loop = asyncio.get_running_loop()
    while True:
        started = loop.time()
        stats = await run_one_cycle(http, pool)
        if on_cycle is not None:
            await on_cycle(stats)
        elapsed = loop.time() - started
        sleep_for = max(0.0, settings.poll_interval_s - elapsed)
        log.debug("cycle finished in %.2fs, sleeping %.2fs", elapsed, sleep_for)
        await asyncio.sleep(sleep_for)
