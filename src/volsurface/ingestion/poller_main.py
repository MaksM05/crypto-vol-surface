"""Poller-only entry point for unattended research capture.

Runs ONLY the REST poller (the sole DB writer) with no WebSocket
subscriber. The subscriber holds in-memory state for a future dashboard,
writes nothing, and over a multi-week capture is pure liability: a
disconnect escaping its handler would take the TaskGroup (and the poller)
down with it. For the research capture we want exactly one supervised
thing: the 5-minute poller.

Run under systemd with Restart=always. run_forever fails loudly on a bad
cycle (exception propagates, process exits non-zero); systemd revives it.
A missed cycle or two across a restart is invisible at 5-min cadence.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from volsurface.config import Settings
from volsurface.ingestion.deribit_client import build_http_client
from volsurface.ingestion.rest_poller import run_forever
from volsurface.storage import close_pool, get_pool

log = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    pool = await get_pool(settings)
    try:
        async with build_http_client(settings) as http:
            await run_forever(settings, http, pool)
    finally:
        await close_pool()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
