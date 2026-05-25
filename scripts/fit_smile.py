"""CLI: fit SVI smile(s) from a stored snapshot.

Usage
-----
    uv run python scripts/fit_smile.py
        [--time ISO8601]    # default: most recent option_quotes.time
        [--expiry ISO8601]  # default: fit every distinct expiry at that snapshot
        [--output-dir PATH] # default: tests/analytics/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from volsurface.analytics.smile_fit import fit_one_smile
from volsurface.config import Settings
from volsurface.storage import close_pool, get_pool

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "analytics"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--time",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO 8601 snapshot timestamp; defaults to most recent",
    )
    p.add_argument(
        "--expiry",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO 8601 expiry; defaults to every distinct expiry at the snapshot",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
            if args.expiry is not None:
                expiries = [args.expiry]
            else:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT i.expiry
                    FROM option_quotes q
                    JOIN instruments i ON i.instrument_name = q.instrument_name
                    WHERE q.time = $1
                    ORDER BY i.expiry
                    """,
                    snapshot_time,
                )
                expiries = [r["expiry"] for r in rows]
                if not expiries:
                    raise SystemExit(f"no quotes found at {snapshot_time.isoformat()}")

            for expiry in expiries:
                report = await fit_one_smile(
                    conn, snapshot_time, expiry, output_dir=args.output_dir
                )
                params = report.fit.params
                p_str = (
                    f"a={params.a:.4f} b={params.b:.4f} ρ={params.rho:+.3f} "
                    f"m={params.m:+.3f} σ={params.sigma:.4f}"
                    if params is not None
                    else "—"
                )
                print(
                    f"expiry={report.expiry.date()} T={report.t_years:.3f}y  "
                    f"n_liquid={report.n_quotes_liquid:3d}  "
                    f"success={report.fit.success}  arb_free={report.fit.is_butterfly_free}  "
                    f"RMSE(w)={report.fit.rmse:.5f}  [{p_str}]"
                )
        print(f"plots written to {args.output_dir}/")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
