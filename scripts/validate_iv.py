"""CLI: run the IV-error validation harness against a stored snapshot.

Usage
-----
    uv run python scripts/validate_iv.py [--time ISO8601] [--version v0.1.0]

If ``--time`` is omitted, picks the most recent ``option_quotes.time``.
Histogram PNG + summary JSON land in ``tests/validation/`` by default
(per CLAUDE.md: "An error histogram lives in `tests/validation/`").
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from volsurface.config import Settings
from volsurface.storage import close_pool, get_pool
from volsurface.validation.iv_error import compute_iv_errors

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "validation"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--time",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO 8601 snapshot timestamp; defaults to most recent in option_quotes",
    )
    p.add_argument("--version", default="v0.1.0", help="pricer version tag")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where to write PNG + JSON (default {DEFAULT_OUTPUT_DIR})",
    )
    return p.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _parse_args()
    settings = Settings()
    pool = await get_pool(settings)
    try:
        async with pool.acquire() as conn:
            snapshot_time = args.time
            if snapshot_time is None:
                snapshot_time = await conn.fetchval("SELECT max(time) FROM option_quotes")
                if snapshot_time is None:
                    raise SystemExit("no option_quotes in DB; run the poller first")
            report = await compute_iv_errors(
                conn,
                snapshot_time,
                pricer_version=args.version,
                output_dir=args.output_dir,
            )
        print(
            f"snapshot={report.snapshot_time.isoformat()}  "
            f"n_total={report.n_total}  n_priced={report.n_priced}  "
            f"n_liquid={report.n_liquid}"
        )
        print(
            f"  |error| on liquid: mean={report.abs_error_mean_liquid:.4f}  "
            f"median={report.abs_error_median_liquid:.4f}  "
            f"p95={report.abs_error_p95_liquid:.4f}  "
            f"max={report.abs_error_max_liquid:.4f} vol pts"
        )
        print(f"histogram + summary written to {args.output_dir}/")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
