"""Tests for the WebSocket subscriber. Uses an in-process fake WS server."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator

import pytest
import websockets
from websockets.asyncio.server import ServerConnection, serve

from volsurface.config import Settings
from volsurface.ingestion.deribit_ws import DeribitBookSubscriber

pytestmark = pytest.mark.integration  # exercises real sockets even if not the DB


# ---------------------------------------------------------------------------
# Test server
# ---------------------------------------------------------------------------


class FakeDeribit:
    """A tiny WS server that records every frame it receives and lets the test
    push a book update at will."""

    def __init__(self) -> None:
        self.received: list[dict] = []
        self.conn: ServerConnection | None = None
        self._connected = asyncio.Event()

    async def handler(self, ws: ServerConnection) -> None:
        self.conn = ws
        self._connected.set()
        try:
            async for raw in ws:
                self.received.append(json.loads(raw))
        except websockets.exceptions.ConnectionClosed:
            return

    async def wait_connected(self, timeout: float = 2.0) -> None:
        await asyncio.wait_for(self._connected.wait(), timeout)

    async def push_book_update(
        self,
        instrument_name: str,
        bid: float,
        ask: float,
        ts_ms: int,
    ) -> None:
        assert self.conn is not None
        msg = {
            "jsonrpc": "2.0",
            "method": "subscription",
            "params": {
                "channel": f"book.{instrument_name}.none.1.100ms",
                "data": {
                    "instrument_name": instrument_name,
                    "bids": [[bid, 1.0]],
                    "asks": [[ask, 1.0]],
                    "timestamp": ts_ms,
                },
            },
        }
        await self.conn.send(json.dumps(msg))


@contextlib.asynccontextmanager
async def fake_server() -> AsyncIterator[tuple[FakeDeribit, str]]:
    fake = FakeDeribit()
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        host, port = server.sockets[0].getsockname()[:2]
        yield fake, f"ws://{host}:{port}"


def _settings_for(url: str) -> Settings:
    s = Settings()
    return s.model_copy(
        update={
            "deribit_ws_url": url,
            "ws_backoff_initial_s": 0.05,
            "ws_backoff_max_s": 0.2,
        }
    )


async def _wait_for(predicate, timeout: float = 2.0, step: float = 0.02) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(step)
    raise AssertionError("condition not reached within timeout")


# ---------------------------------------------------------------------------


async def test_subscribes_on_connect() -> None:
    async with fake_server() as (fake, url):
        sub = DeribitBookSubscriber(_settings_for(url))
        await sub.set_universe({"BTC-23MAY26-68000-C", "BTC-23MAY26-69000-P"})
        task = asyncio.create_task(sub.run())
        try:
            await fake.wait_connected()
            await _wait_for(
                lambda: any(m.get("method") == "public/subscribe" for m in fake.received)
            )
            subscribe = next(m for m in fake.received if m.get("method") == "public/subscribe")
            channels = set(subscribe["params"]["channels"])
            assert channels == {
                "book.BTC-23MAY26-68000-C.none.1.100ms",
                "book.BTC-23MAY26-69000-P.none.1.100ms",
            }
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_updates_in_memory_book_on_message() -> None:
    async with fake_server() as (fake, url):
        sub = DeribitBookSubscriber(_settings_for(url))
        await sub.set_universe({"BTC-23MAY26-68000-C"})
        task = asyncio.create_task(sub.run())
        try:
            await fake.wait_connected()
            await _wait_for(lambda: fake.received != [])
            await fake.push_book_update(
                "BTC-23MAY26-68000-C", bid=0.041, ask=0.043, ts_ms=1700000000000
            )
            await _wait_for(lambda: sub.latest("BTC-23MAY26-68000-C") is not None)
            top = sub.latest("BTC-23MAY26-68000-C")
            assert top is not None
            assert top.best_bid == 0.041
            assert top.best_ask == 0.043
            assert top.time.year >= 2023
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_reconnects_and_resubscribes_after_disconnect() -> None:
    async with fake_server() as (fake, url):
        sub = DeribitBookSubscriber(_settings_for(url))
        await sub.set_universe({"BTC-23MAY26-68000-C"})
        task = asyncio.create_task(sub.run())
        try:
            await fake.wait_connected()
            await _wait_for(
                lambda: any(m.get("method") == "public/subscribe" for m in fake.received)
            )
            # Force a disconnect from the server side.
            assert fake.conn is not None
            await fake.conn.close()
            # Subscriber should reconnect and resubscribe.
            await _wait_for(
                lambda: sum(1 for m in fake.received if m.get("method") == "public/subscribe") >= 2,
                timeout=3.0,
            )
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_set_universe_diff_sends_subscribe_and_unsubscribe() -> None:
    async with fake_server() as (fake, url):
        sub = DeribitBookSubscriber(_settings_for(url))
        await sub.set_universe({"A", "B"})
        task = asyncio.create_task(sub.run())
        try:
            await fake.wait_connected()
            await _wait_for(
                lambda: any(m.get("method") == "public/subscribe" for m in fake.received)
            )
            initial_subs = sum(1 for m in fake.received if m.get("method") == "public/subscribe")

            # Shift universe: drop B, add C.
            await sub.set_universe({"A", "C"})
            await _wait_for(
                lambda: any(m.get("method") == "public/unsubscribe" for m in fake.received)
            )
            unsub = next(m for m in fake.received if m.get("method") == "public/unsubscribe")
            assert unsub["params"]["channels"] == ["book.B.none.1.100ms"]
            new_subs = [m for m in fake.received if m.get("method") == "public/subscribe"][
                initial_subs:
            ]
            assert any("book.C.none.1.100ms" in m["params"]["channels"] for m in new_subs)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
