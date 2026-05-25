"""Fit a full SSVI surface from a stored snapshot.

For one ``snapshot_time``:

1. Pull all option_quotes joined to instruments + per-expiry forwards.
2. Per expiry: liquidity filter (same rules as ``smile_fit.py``), IV solve.
3. Per expiry: per-expiry SVI fit, take ``θ_T = w_svi(0, T)`` for the backbone.
4. Build the sorted backbone, drop expiries that couldn't produce a robust θ_T.
5. Joint SSVI fit of ``(ρ, η, γ)`` with the butterfly inequalities as hard
   SLSQP constraints (butterfly arb-free by construction).
6. Post-fit numerical calendar check on a wide ``k``-grid (calendar arb-free
   by parameter restriction and verification — NOT by construction; see
   ``calibration/ssvi.py`` docstring).

No 3D plot in this slice (that lands as a separate follow-on). An optional
``surface_summary_<snapshot>.json`` is written when ``output_dir`` is given.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from volsurface.calibration.ssvi import (
    SSVIBackbone,
    SSVIFitResult,
    fit_ssvi,
    ssvi_total_variance,
)
from volsurface.calibration.svi import fit_svi, svi_total_variance
from volsurface.pricer.forward import time_to_expiry_years
from volsurface.pricer.iv_solver import implied_vol
from volsurface.storage import fetch_option_quotes_at, get_forward
from volsurface.storage.db import DbConn

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SurfaceFitReport:
    """Result of fitting one full SSVI surface."""

    snapshot_time: datetime
    n_expiries_total: int  # distinct expiries seen at this snapshot
    n_expiries_fitted: int  # entered the joint fit (had robust backbone θ_T)
    n_points_total: int  # liquid OTM points across all fitted expiries
    fit: SSVIFitResult
    per_expiry_rmse: dict[datetime, float]
    forwards: dict[datetime, float]
    skipped_expiries: dict[datetime, str]  # expiry → why skipped


async def fit_surface(
    conn: DbConn,
    snapshot_time: datetime,
    *,
    output_dir: Path | None = None,
) -> SurfaceFitReport:
    """Pull, filter, build backbone, jointly fit SSVI, verify both arb conditions.

    Parameters
    ----------
    conn
        asyncpg connection or pool-proxy.
    snapshot_time
        Exact ``option_quotes.time`` to fit.
    output_dir
        If provided, writes
        ``surface_summary_<snapshot-time>.json`` into the directory. No PNG;
        the 3D viz is a follow-on slice.

    Returns
    -------
    SurfaceFitReport
        Diagnostics + the SSVI fit result. Caller decides what to do on
        ``fit.success == False`` or either arb flag being false.
    """
    all_quotes = await fetch_option_quotes_at(conn, snapshot_time)
    distinct_expiries = sorted({q.expiry for q in all_quotes})

    backbone_entries: list[tuple[datetime, float, float]] = []  # (expiry, T, θ_T)
    backbone_points: dict[datetime, list[tuple[float, float]]] = {}  # expiry → [(k, w)]
    forwards: dict[datetime, float] = {}
    skipped: dict[datetime, str] = {}

    for exp in distinct_expiries:
        fwd = await get_forward(conn, exp, snapshot_time)
        if fwd is None:
            skipped[exp] = "no forward recorded"
            continue
        forwards[exp] = fwd.forward_price
        t = time_to_expiry_years(snapshot_time, exp)
        if t <= 0.0:
            skipped[exp] = f"non-positive T={t:.4f}"
            continue

        expiry_quotes = [q for q in all_quotes if q.expiry == exp]
        points: list[tuple[float, float]] = []  # (k, w_market = σ²·T)
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
            points.append((k, sigma * sigma * t))

        if len(points) < 6:
            skipped[exp] = f"only {len(points)} liquid OTM points (need >= 6)"
            continue

        # Per-expiry SVI fit → robust θ_T = w_svi(0, T).
        ks_arr = np.array([p[0] for p in points], dtype=np.float64)
        ws_arr = np.array([p[1] for p in points], dtype=np.float64)
        svi_fit = fit_svi(ks_arr, ws_arr)
        if not svi_fit.success or svi_fit.params is None:
            skipped[exp] = f"per-expiry SVI failed: {svi_fit.message}"
            continue
        theta_t = float(svi_total_variance(0.0, svi_fit.params))
        if not math.isfinite(theta_t) or theta_t <= 0.0:
            skipped[exp] = f"non-positive θ_T={theta_t}"
            continue
        backbone_entries.append((exp, t, theta_t))
        backbone_points[exp] = points

    # Sort by T ascending.
    backbone_entries.sort(key=lambda x: x[1])
    if not backbone_entries:
        empty_backbone = SSVIBackbone(expiries=(), t_years=(), theta=())
        empty_fit = fit_ssvi(empty_backbone, [])
        return SurfaceFitReport(
            snapshot_time=snapshot_time,
            n_expiries_total=len(distinct_expiries),
            n_expiries_fitted=0,
            n_points_total=0,
            fit=empty_fit,
            per_expiry_rmse={},
            forwards=forwards,
            skipped_expiries=skipped,
        )

    backbone = SSVIBackbone(
        expiries=tuple(e for e, _, _ in backbone_entries),
        t_years=tuple(t for _, t, _ in backbone_entries),
        theta=tuple(theta for _, _, theta in backbone_entries),
    )
    exp_to_idx = {e: i for i, e in enumerate(backbone.expiries)}
    points_with_idx: list[tuple[float, float, int]] = []
    for exp in backbone.expiries:
        for k, w in backbone_points[exp]:
            points_with_idx.append((k, w, exp_to_idx[exp]))

    log.info(
        "surface %s: %d expiries seen, %d enter joint fit, %d points",
        snapshot_time.isoformat(),
        len(distinct_expiries),
        len(backbone.expiries),
        len(points_with_idx),
    )

    fit = fit_ssvi(backbone, points_with_idx)

    per_expiry_rmse: dict[datetime, float] = {}
    if fit.success and fit.params is not None:
        for i, exp in enumerate(backbone.expiries):
            pts = [(k, w) for k, w, idx in points_with_idx if idx == i]
            if not pts:
                per_expiry_rmse[exp] = math.nan
                continue
            ks = np.array([p[0] for p in pts])
            ws = np.array([p[1] for p in pts])
            w_pred = ssvi_total_variance(ks, backbone.theta[i], fit.params)
            per_expiry_rmse[exp] = float(np.sqrt(np.mean((w_pred - ws) ** 2)))

    report = SurfaceFitReport(
        snapshot_time=snapshot_time,
        n_expiries_total=len(distinct_expiries),
        n_expiries_fitted=len(backbone.expiries),
        n_points_total=len(points_with_idx),
        fit=fit,
        per_expiry_rmse=per_expiry_rmse,
        forwards=forwards,
        skipped_expiries=skipped,
    )

    if output_dir is not None:
        _write_summary(report, output_dir)

    return report


def _write_summary(report: SurfaceFitReport, output_dir: Path) -> Path:
    """Persist a JSON snapshot of the fit for later plotting / diff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "snapshot_time": report.snapshot_time.isoformat(),
        "n_expiries_total": report.n_expiries_total,
        "n_expiries_fitted": report.n_expiries_fitted,
        "n_points_total": report.n_points_total,
        "fit": {
            "success": report.fit.success,
            "is_butterfly_free": report.fit.is_butterfly_free,
            "is_calendar_free": report.fit.is_calendar_free,
            "butterfly_message": report.fit.butterfly_message,
            "calendar_message": report.fit.calendar_message,
            "message": report.fit.message,
            "rmse": report.fit.rmse,
            "n_points": report.fit.n_points,
            "n_expiries": report.fit.n_expiries,
        },
        "params": (
            None
            if report.fit.params is None
            else {
                "rho": report.fit.params.rho,
                "eta": report.fit.params.eta,
                "gamma": report.fit.params.gamma,
            }
        ),
        "backbone": (
            None
            if report.fit.backbone is None
            else [
                {"expiry": e.isoformat(), "t_years": t, "theta": th}
                for e, t, th in zip(
                    report.fit.backbone.expiries,
                    report.fit.backbone.t_years,
                    report.fit.backbone.theta,
                    strict=True,
                )
            ]
        ),
        "per_expiry_rmse": {e.isoformat(): rmse for e, rmse in report.per_expiry_rmse.items()},
        "forwards": {e.isoformat(): f for e, f in report.forwards.items()},
        "skipped_expiries": {e.isoformat(): why for e, why in report.skipped_expiries.items()},
    }
    filename = f"surface_summary_{report.snapshot_time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path = output_dir / filename
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
