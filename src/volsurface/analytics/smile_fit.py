"""Fit a single-expiry SVI smile from a stored snapshot.

Pulls liquid OTM quotes for one ``(snapshot_time, expiry)``, solves IV with
the existing pricer, fits raw SVI in total-variance space, runs the
Gatheral–Durrleman no-butterfly check, and renders a vol-space plot.

Liquidity filter (applied BEFORE fitting, per CLAUDE.md):

- ``open_interest > 10``
- ``best_bid`` and ``best_ask`` non-null, ``mark_price > 0``
- ``(best_ask − best_bid) / mark_price < 0.05``
- IV solver returned a finite value
- OTM-side only: puts for ``K < F``, calls for ``K >= F``
- ``|k| = |ln(K/F)| < 3``

Unit convention: same as ``validation/iv_error`` — ``market_usd =
mark_price_btc × index_price`` (S, not F), ``DF = 1``. The plot displays
implied vol on the y-axis but the fit itself is in total-variance space
(``w = σ²·T``), which is where the SVI parameterisation and the no-arb
condition live.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from volsurface.calibration.svi import (
    SVIFitResult,
    fit_svi,
    svi_total_variance,
)
from volsurface.pricer.forward import time_to_expiry_years
from volsurface.pricer.iv_solver import implied_vol
from volsurface.storage import fetch_option_quotes_at, get_forward
from volsurface.storage.db import DbConn

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SmilePoint:
    """One liquid (k, σ_BS) datum that entered the fit."""

    k: float
    sigma_bs: float  # decimal vol
    strike: float
    is_call: bool


@dataclass(frozen=True, slots=True)
class SmileFitReport:
    """Result of fitting one expiry's smile."""

    snapshot_time: datetime
    expiry: datetime
    forward: float
    index_price: float
    t_years: float
    n_quotes_total: int  # raw quotes pulled for this expiry
    n_quotes_liquid: int  # survived the filter and entered the fit
    fit: SVIFitResult
    points: list[SmilePoint]


async def fit_one_smile(
    conn: DbConn,
    snapshot_time: datetime,
    expiry: datetime,
    *,
    output_dir: Path | None = None,
) -> SmileFitReport:
    """Pull, filter, solve, fit, and (optionally) plot one smile.

    Parameters
    ----------
    conn
        asyncpg connection or pool-proxy.
    snapshot_time
        Exact ``option_quotes.time`` to fit.
    expiry
        Exact ``instruments.expiry`` to restrict to.
    output_dir
        If provided, writes
        ``smile_<expiry-date>_<snapshot-time>.png`` into the directory.

    Returns
    -------
    SmileFitReport
        Diagnostics + the SVI fit result. Caller decides what to do on
        ``fit.success == False`` or ``fit.is_butterfly_free == False``.
    """
    fwd = await get_forward(conn, expiry, snapshot_time)
    if fwd is None:
        raise ValueError(f"no forward recorded for expiry {expiry} at {snapshot_time}")

    all_quotes = await fetch_option_quotes_at(conn, snapshot_time)
    expiry_quotes = [q for q in all_quotes if q.expiry == expiry]
    t = time_to_expiry_years(snapshot_time, expiry)

    points: list[SmilePoint] = []
    for q in expiry_quotes:
        if q.mark_price is None or q.mark_price <= 0:
            continue
        if q.open_interest is None or q.open_interest <= 10.0:
            continue
        if q.best_bid is None or q.best_ask is None:
            continue
        if (q.best_ask - q.best_bid) / q.mark_price >= 0.05:
            continue
        is_call = q.option_type == "C"
        # OTM-only: puts for K<F, calls for K>=F.
        if is_call and q.strike < fwd.forward_price:
            continue
        if (not is_call) and q.strike >= fwd.forward_price:
            continue
        market_usd = q.mark_price * fwd.index_price
        sigma = implied_vol(
            market_usd,
            fwd.forward_price,
            q.strike,
            t,
            df=1.0,
            is_call=is_call,
        )
        if not math.isfinite(sigma):
            continue
        k = math.log(q.strike / fwd.forward_price)
        if abs(k) >= 3.0:
            continue
        points.append(SmilePoint(k=k, sigma_bs=sigma, strike=q.strike, is_call=is_call))

    points.sort(key=lambda p: p.k)
    n_liquid = len(points)
    log.info(
        "smile %s expiry %s: %d quotes -> %d liquid OTM points",
        snapshot_time.isoformat(),
        expiry.isoformat(),
        len(expiry_quotes),
        n_liquid,
    )

    if n_liquid < 6:
        fit = SVIFitResult(
            params=None,
            rmse=math.nan,
            n_points=n_liquid,
            success=False,
            is_butterfly_free=False,
            message=f"only {n_liquid} liquid OTM points",
        )
    else:
        ks = [p.k for p in points]
        ws = [p.sigma_bs * p.sigma_bs * t for p in points]
        fit = fit_svi(ks, ws)

    report = SmileFitReport(
        snapshot_time=snapshot_time,
        expiry=expiry,
        forward=fwd.forward_price,
        index_price=fwd.index_price,
        t_years=t,
        n_quotes_total=len(expiry_quotes),
        n_quotes_liquid=n_liquid,
        fit=fit,
        points=points,
    )

    if output_dir is not None:
        _render_plot(report, output_dir)

    return report


def _render_plot(report: SmileFitReport, output_dir: Path) -> Path:
    """Render σ_BS vs k: market dots + fitted curve + ATM marker."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    if report.points:
        ks = np.array([p.k for p in report.points])
        sigmas_pct = np.array([p.sigma_bs * 100.0 for p in report.points])
        ax.scatter(ks, sigmas_pct, color="black", s=35, zorder=3, label="market (liquid OTM)")
    else:
        ks = np.array([])
        sigmas_pct = np.array([])

    if report.fit.params is not None and report.t_years > 0.0 and report.points:
        k_min, k_max = float(ks.min()), float(ks.max())
        pad = max(0.1, 0.15 * (k_max - k_min))
        k_grid = np.linspace(k_min - pad, k_max + pad, 300)
        w_grid = svi_total_variance(k_grid, report.fit.params)
        sigma_grid_pct = np.sqrt(np.maximum(w_grid, 0.0) / report.t_years) * 100.0
        ax.plot(k_grid, sigma_grid_pct, color="C0", linewidth=1.6, label="SVI fit")

    ax.axvline(0.0, color="red", linestyle="--", alpha=0.5, label="ATM (k = 0)")
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("Black-76 implied vol (%)")
    title = (
        f"Smile fit  expiry={report.expiry.date()}  "
        f"T={report.t_years:.3f}y  "
        f"n_liquid={report.n_quotes_liquid}  "
        f"RMSE(w)={report.fit.rmse:.5f}  "
        f"arb_free={report.fit.is_butterfly_free}"
    )
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    filename = (
        f"smile_{report.expiry.strftime('%Y%m%d')}"
        f"_{report.snapshot_time.strftime('%Y%m%dT%H%M%S')}.png"
    )
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
