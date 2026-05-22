"""IV-error validation harness — graders our pricer against Deribit's mark_iv.

For a stored snapshot ``(snapshot_time)``: pull every option_quote at that
timestamp joined to its instrument metadata, look up the per-expiry forward
from the same snapshot, solve our Black-76 IV from the USD-converted market
premium, compare to ``deribit_mark_iv``. Report per-row + summary stats +
write a histogram PNG and a summary JSON to disk for the README.

The unit convention is documented in ``validation/README.md``: the harness
uses ``S`` (BTC index) for the BTC→USD price conversion and ``DF = 1`` to
match Deribit's published mark-IV convention. Using ``F`` instead of ``S``
here would introduce a basis-sized bias and the 0.1 vol-pt gate would fail.

Deribit returns ``mark_iv`` in **percentage points** (e.g. ``47.5`` means
47.5% vol). Our solver returns decimal vol. Conversion: ``our_iv_pct =
sigma × 100``.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from volsurface.pricer.forward import time_to_expiry_years
from volsurface.pricer.iv_solver import implied_vol
from volsurface.storage import fetch_option_quotes_at, get_forward
from volsurface.storage.db import DbConn

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IVErrorRow:
    """Per-option result of the validation pass."""

    instrument_name: str
    strike: float
    expiry: datetime
    is_call: bool
    open_interest: float | None
    best_bid: float | None
    best_ask: float | None
    mark_price_btc: float
    market_price_usd: float
    forward: float
    index_price: float
    t_years: float
    deribit_mark_iv_pct: float
    our_iv_pct: float | None  # None when the solver returned nan
    iv_error_pct: float | None  # ours - Deribit, None when solver failed
    is_liquid: bool


@dataclass(frozen=True, slots=True)
class IVErrorReport:
    """Aggregate of an IV-validation pass against one snapshot."""

    pricer_version: str
    snapshot_time: datetime
    rows: list[IVErrorRow]
    n_total: int  # quotes considered (had a forward + non-null mark + non-null mark_iv)
    n_priced: int  # solver returned a finite IV
    n_liquid: int  # priced AND liquid
    abs_error_mean_liquid: float
    abs_error_median_liquid: float
    abs_error_p95_liquid: float
    abs_error_max_liquid: float


async def compute_iv_errors(
    conn: DbConn,
    snapshot_time: datetime,
    *,
    pricer_version: str = "v0.1.0",
    output_dir: Path | None = None,
) -> IVErrorReport:
    """Validate our IV solver against Deribit ``mark_iv`` for one snapshot.

    Parameters
    ----------
    conn
        asyncpg connection or pool-proxy.
    snapshot_time
        Exact ``option_quotes.time`` value to validate. Use a value produced
        by the REST poller (``datetime.now(UTC).replace(microsecond=0)``).
    pricer_version
        Tag used in the histogram filename and the report (for side-by-side
        comparison of pricer revisions).
    output_dir
        If given, writes ``error_histogram_<version>.png`` and
        ``error_summary_<version>.json`` into this directory.

    Returns
    -------
    IVErrorReport
        Per-row results plus summary stats on the liquid subset.
    """
    quotes = await fetch_option_quotes_at(conn, snapshot_time)
    # One forward per distinct expiry — fetch once, share across rows.
    expiries = {q.expiry for q in quotes}
    forwards_by_expiry = {}
    for expiry in expiries:
        fwd = await get_forward(conn, expiry, snapshot_time)
        if fwd is not None:
            forwards_by_expiry[expiry] = fwd

    rows: list[IVErrorRow] = []
    for q in quotes:
        if q.mark_price is None or q.deribit_mark_iv is None:
            continue
        fwd = forwards_by_expiry.get(q.expiry)
        if fwd is None:
            log.warning("no forward for %s at %s", q.instrument_name, snapshot_time)
            continue

        # --- the unit conversion, the load-bearing decision -----------------
        market_price_usd = q.mark_price * fwd.index_price
        # --------------------------------------------------------------------

        t = time_to_expiry_years(snapshot_time, q.expiry)
        is_call = q.option_type == "C"
        sigma = implied_vol(
            market_price_usd,
            fwd.forward_price,
            q.strike,
            t,
            df=1.0,
            is_call=is_call,
        )
        our_iv_pct = sigma * 100.0 if math.isfinite(sigma) else None
        iv_error_pct = our_iv_pct - q.deribit_mark_iv if our_iv_pct is not None else None

        is_liquid = (
            q.open_interest is not None
            and q.open_interest > 10.0
            and q.best_bid is not None
            and q.best_ask is not None
            and q.mark_price > 0.0
            and (q.best_ask - q.best_bid) / q.mark_price < 0.05
        )

        rows.append(
            IVErrorRow(
                instrument_name=q.instrument_name,
                strike=q.strike,
                expiry=q.expiry,
                is_call=is_call,
                open_interest=q.open_interest,
                best_bid=q.best_bid,
                best_ask=q.best_ask,
                mark_price_btc=q.mark_price,
                market_price_usd=market_price_usd,
                forward=fwd.forward_price,
                index_price=fwd.index_price,
                t_years=t,
                deribit_mark_iv_pct=q.deribit_mark_iv,
                our_iv_pct=our_iv_pct,
                iv_error_pct=iv_error_pct,
                is_liquid=is_liquid,
            )
        )

    liquid_errors = [
        abs(r.iv_error_pct) for r in rows if r.is_liquid and r.iv_error_pct is not None
    ]
    n_priced = sum(1 for r in rows if r.our_iv_pct is not None)

    if liquid_errors:
        abs_error_mean = statistics.fmean(liquid_errors)
        abs_error_median = statistics.median(liquid_errors)
        sorted_errs = sorted(liquid_errors)
        p95_idx = max(0, math.ceil(0.95 * len(sorted_errs)) - 1)
        abs_error_p95 = sorted_errs[p95_idx]
        abs_error_max = max(sorted_errs)
    else:
        abs_error_mean = abs_error_median = abs_error_p95 = abs_error_max = math.nan

    report = IVErrorReport(
        pricer_version=pricer_version,
        snapshot_time=snapshot_time,
        rows=rows,
        n_total=len(rows),
        n_priced=n_priced,
        n_liquid=len(liquid_errors),
        abs_error_mean_liquid=abs_error_mean,
        abs_error_median_liquid=abs_error_median,
        abs_error_p95_liquid=abs_error_p95,
        abs_error_max_liquid=abs_error_max,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_histogram(
            [r.iv_error_pct for r in rows if r.is_liquid and r.iv_error_pct is not None],
            output_dir / f"error_histogram_{pricer_version}.png",
            pricer_version,
        )
        _write_summary(report, output_dir / f"error_summary_{pricer_version}.json")

    return report


def _write_histogram(errors_pct: list[float], path: Path, pricer_version: str) -> None:
    """Render the error histogram PNG. Matplotlib is imported lazily."""
    import matplotlib

    matplotlib.use("Agg")  # headless backend; works in CI and on servers
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if errors_pct:
        ax.hist(errors_pct, bins=40, edgecolor="black")
    ax.axvline(0.0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(0.1, color="red", linestyle=":", alpha=0.6, label="±0.1 vol-pt budget")
    ax.axvline(-0.1, color="red", linestyle=":", alpha=0.6)
    ax.set_xlabel("IV error (vol points): ours − Deribit")
    ax.set_ylabel("count")
    ax.set_title(f"IV validation vs Deribit mark_iv  ({pricer_version})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_summary(report: IVErrorReport, path: Path) -> None:
    """Persist the report as JSON. Datetimes serialised as ISO 8601."""
    payload = {
        "pricer_version": report.pricer_version,
        "snapshot_time": report.snapshot_time.isoformat(),
        "n_total": report.n_total,
        "n_priced": report.n_priced,
        "n_liquid": report.n_liquid,
        "abs_error_mean_liquid": report.abs_error_mean_liquid,
        "abs_error_median_liquid": report.abs_error_median_liquid,
        "abs_error_p95_liquid": report.abs_error_p95_liquid,
        "abs_error_max_liquid": report.abs_error_max_liquid,
        "rows": [
            {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(r).items()}
            for r in report.rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
