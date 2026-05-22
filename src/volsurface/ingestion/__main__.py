"""Run the REST poller and WS subscriber together.

Both tasks are managed by an ``asyncio.TaskGroup``: if either one raises,
the other is cancelled and the exception is logged at ``CRITICAL`` and
re-raised, so the process exits with a non-zero status. Supervised restart
is the deploy layer's job (systemd / Docker restart policy).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from volsurface.config import Settings
from volsurface.ingestion.deribit_client import build_http_client
from volsurface.ingestion.deribit_ws import DeribitBookSubscriber
from volsurface.ingestion.rest_poller import CycleStats, run_forever
from volsurface.storage import close_pool, get_pool

log = logging.getLogger(__name__)


async def main() -> None:
    """Wire the poller + subscriber into a single supervised event loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    pool = await get_pool(settings)
    subscriber = DeribitBookSubscriber(settings)

    async def on_cycle(stats: CycleStats) -> None:
        await subscriber.set_universe(stats.instrument_universe)

    try:
        async with build_http_client(settings) as http:
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        run_forever(settings, http, pool, on_cycle=on_cycle),
                        name="rest_poller",
                    )
                    tg.create_task(subscriber.run(), name="deribit_ws")
            except* Exception as eg:
                for exc in eg.exceptions:
                    log.critical("ingestion task crashed: %r", exc, exc_info=exc)
                raise
    finally:
        await close_pool()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
