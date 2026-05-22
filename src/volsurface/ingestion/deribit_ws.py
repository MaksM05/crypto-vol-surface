"""Deribit WebSocket subscriber for live top-of-book — in-memory only in v1.

DOES NOT WRITE TO POSTGRES. The REST poller is the sole writer to every
ingestion table (see ``ingestion/README.md``). This module subscribes to
``book.<instrument>.none.1.100ms`` channels and maintains a
``dict[instrument_name, BookTop]`` of the latest top-of-book, ready to be
consumed by the FastAPI dashboard layer in a later milestone.

Connection-management responsibilities:

- Connect, subscribe to the desired instrument set on each (re)connect.
- Exponential backoff with jitter on disconnect, capped at
  ``settings.ws_backoff_max_s``.
- React to a changing instrument universe (new expiries listed, old ones
  expired) by sending incremental subscribe/unsubscribe messages — no full
  reconnect needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from volsurface.config import Settings

log = logging.getLogger(__name__)

_BOOK_CHANNEL_TEMPLATE = "book.{name}.none.1.100ms"


@dataclass(frozen=True, slots=True)
class BookTop:
    """Latest top-of-book observation for one instrument."""

    instrument_name: str
    time: datetime
    best_bid: float | None
    best_ask: float | None


def _channel_for(name: str) -> str:
    return _BOOK_CHANNEL_TEMPLATE.format(name=name)


class DeribitBookSubscriber:
    """Maintains live top-of-book state for a dynamic instrument set.

    The poller drives the instrument set via :meth:`set_universe`. The
    subscriber reconciles its subscribed channels to match.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._desired: set[str] = set()
        self._subscribed: set[str] = set()
        self._state: dict[str, BookTop] = {}
        self._ws: ClientConnection | None = None
        self._lock = asyncio.Lock()
        self._next_id = 1

    def latest(self, instrument_name: str) -> BookTop | None:
        """Return the most recent top-of-book seen for ``instrument_name``, or ``None``."""
        return self._state.get(instrument_name)

    def snapshot(self) -> dict[str, BookTop]:
        """Return a shallow copy of the current top-of-book state."""
        return dict(self._state)

    async def set_universe(self, instruments: Iterable[str]) -> None:
        """Update the desired subscription set and reconcile with the live socket.

        Safe to call repeatedly. If the websocket is not connected, the new
        desired set is stored and applied on the next (re)connect.
        """
        async with self._lock:
            self._desired = set(instruments)
            if self._ws is not None:
                await self._reconcile_subscriptions()

    async def _reconcile_subscriptions(self) -> None:
        """Send subscribe/unsubscribe for the diff. Caller must hold ``self._lock``."""
        to_add = self._desired - self._subscribed
        to_remove = self._subscribed - self._desired
        if to_add:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": self._claim_id(),
                    "method": "public/subscribe",
                    "params": {"channels": [_channel_for(n) for n in sorted(to_add)]},
                }
            )
            self._subscribed |= to_add
        if to_remove:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": self._claim_id(),
                    "method": "public/unsubscribe",
                    "params": {"channels": [_channel_for(n) for n in sorted(to_remove)]},
                }
            )
            self._subscribed -= to_remove

    def _claim_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    async def _send(self, msg: dict[str, Any]) -> None:
        assert self._ws is not None, "_send called with no open socket"
        await self._ws.send(json.dumps(msg))

    async def run(self) -> None:
        """Connect-subscribe-dispatch loop with exponential backoff on disconnect.

        Returns only via cancellation. Other exceptions propagate to the
        caller's ``TaskGroup`` so failure is loud.
        """
        backoff = self._settings.ws_backoff_initial_s
        while True:
            try:
                async with websockets.connect(self._settings.deribit_ws_url) as ws:
                    log.info("ws connected to %s", self._settings.deribit_ws_url)
                    self._ws = ws
                    backoff = self._settings.ws_backoff_initial_s
                    async with self._lock:
                        self._subscribed.clear()
                        await self._reconcile_subscriptions()
                    async for raw in ws:
                        self._handle_message(raw)
            except (ConnectionClosed, OSError) as exc:
                self._ws = None
                jitter = random.uniform(0.0, 0.25 * backoff)
                wait = backoff + jitter
                log.warning("ws disconnected (%s); reconnecting in %.2fs", exc, wait)
                await asyncio.sleep(wait)
                backoff = min(backoff * 2.0, self._settings.ws_backoff_max_s)

    def _handle_message(self, raw: str | bytes) -> None:
        """Parse a single message and update in-memory state if it's a book update."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("ws: non-JSON message dropped")
            return
        if not isinstance(msg, dict):
            return
        params = msg.get("params")
        if not isinstance(params, dict):
            return  # subscribe/unsubscribe ack or heartbeat
        channel = params.get("channel")
        data = params.get("data")
        if not isinstance(channel, str) or not channel.startswith("book."):
            return
        if not isinstance(data, dict):
            return
        instrument = data.get("instrument_name")
        if not isinstance(instrument, str):
            return
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None
        ts_ms = data.get("timestamp")
        ts = (
            datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
            if isinstance(ts_ms, int)
            else datetime.now(UTC)
        )
        self._state[instrument] = BookTop(
            instrument_name=instrument,
            time=ts,
            best_bid=best_bid,
            best_ask=best_ask,
        )
