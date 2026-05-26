"""CLI: fit a full SSVI surface from a stored snapshot.

Usage
-----
    uv run python scripts/fit_surface.py
        [--time ISO8601]    # default: most recent option_quotes.time
        [--output-dir PATH] # default: tests/analytics/
        [--no-plot]         # default: render the 3D plot
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from volsurface.analytics.surface_fit import fit_surface
from volsurface.analytics.surface_plot import render_surface
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
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="skip rendering the 3D HTML+PNG plot (fit + summary JSON only)",
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
            report = await fit_surface(conn, snapshot_time, output_dir=args.output_dir)
        print(
            f"snapshot={report.snapshot_time.isoformat()}  "
            f"expiries seen={report.n_expiries_total}  "
            f"fitted={report.n_expiries_fitted}  "
            f"n_points={report.n_points_total}"
        )
        print(
            f"  success={report.fit.success}  "
            f"butterfly_free={report.fit.is_butterfly_free}  "
            f"calendar_free={report.fit.is_calendar_free}  "
            f"RMSE(w)={report.fit.rmse:.5f}"
        )
        if report.fit.params is not None:
            p = report.fit.params
            print(f"  params: ρ={p.rho:+.4f}  η={p.eta:.4f}  γ={p.gamma:.4f}")
        if not report.fit.is_butterfly_free:
            print(f"  butterfly: {report.fit.butterfly_message}")
        if not report.fit.is_calendar_free:
            print(f"  calendar: {report.fit.calendar_message}")
        for exp, rmse in sorted(report.per_expiry_rmse.items()):
            print(f"    {exp.date()}: RMSE(w)={rmse:.5f}")
        if report.skipped_expiries:
            print("  skipped:")
            for exp, why in sorted(report.skipped_expiries.items()):
                print(f"    {exp.date()}: {why}")
        print(f"summary written to {args.output_dir}/")

        if not args.no_plot:
            if report.fit.success and report.fit.params is not None:
                paths = render_surface(report, args.output_dir)
                print(f"  HTML: {paths.html}")
                print(f"  PNG:  {paths.png}")
            else:
                print(f"  skipping plot: {report.fit.message}")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
