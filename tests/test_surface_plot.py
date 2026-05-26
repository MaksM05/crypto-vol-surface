"""Tests for the 3D SSVI surface renderer.

Integration where DB is needed (loads the multi-expiry fixture, runs the
fit, then renders), pure-unit where it is not (failed-fit error path).
Render tests check trace shapes and substring claims, not pixels.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import plotly.graph_objects as go
import pytest

from volsurface.analytics.surface_fit import (
    SurfaceFitReport,
    SurfaceMarketPoint,
    fit_surface,
)
from volsurface.analytics.surface_plot import build_figure, render_surface
from volsurface.calibration.ssvi import SSVIBackbone, SSVIFitResult
from volsurface.storage import (
    ForwardRow,
    InstrumentRow,
    OptionQuoteRow,
    insert_forwards,
    insert_option_quotes,
    upsert_instruments,
)

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_btc_surface.json"


async def _load_fixture_into_db(
    pool: asyncpg.Pool, *, drop_forward_for_expiry: str | None = None
) -> datetime:
    """Insert the captured 4-expiry surface fixture.

    ``drop_forward_for_expiry`` lets a test simulate a missing forwards row
    for one expiry so we exercise the dropped-expiry render path.
    """
    payload = json.loads(FIXTURE.read_text())
    snapshot_time = datetime.fromisoformat(payload["snapshot_time"])
    async with pool.acquire() as conn:
        await upsert_instruments(
            conn,
            [
                InstrumentRow(
                    instrument_name=i["instrument_name"],
                    currency=i["currency"],
                    strike=i["strike"],
                    option_type=i["option_type"],
                    expiry=datetime.fromisoformat(i["expiry"]),
                    contract_size=i["contract_size"],
                    creation_ts=datetime.fromisoformat(i["creation_ts"]),
                    last_seen=snapshot_time,
                )
                for i in payload["instruments"]
            ],
        )
        await insert_forwards(
            conn,
            [
                ForwardRow(
                    time=snapshot_time,
                    expiry=datetime.fromisoformat(f["expiry"]),
                    forward_price=f["forward_price"],
                    index_price=f["index_price"],
                )
                for f in payload["forwards"]
                if f["expiry"] != drop_forward_for_expiry
            ],
        )
        await insert_option_quotes(
            conn,
            [
                OptionQuoteRow(
                    time=snapshot_time,
                    instrument_name=q["instrument_name"],
                    mark_price=q["mark_price"],
                    best_bid=q["best_bid"],
                    best_ask=q["best_ask"],
                    open_interest=q["open_interest"],
                    deribit_mark_iv=q["deribit_mark_iv"],
                    deribit_delta=q.get("deribit_delta"),
                )
                for q in payload["option_quotes"]
            ],
        )
    return snapshot_time


# ----- file-writing path ---------------------------------------------------


async def test_writes_html_and_png(pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path) -> None:
    """Both files land on disk, are non-empty, and use the expected filename stem."""
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time)
    paths = render_surface(report, tmp_path)

    assert paths.html.exists() and paths.html.stat().st_size > 0
    assert paths.png.exists() and paths.png.stat().st_size > 0
    stem = f"surface_{snapshot_time.strftime('%Y%m%dT%H%M%S')}"
    assert paths.html.name == f"{stem}.html"
    assert paths.png.name == f"{stem}.png"


async def test_html_is_self_contained(pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path) -> None:
    """HTML inlines plotly.js so the file opens offline."""
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time)
    paths = render_surface(report, tmp_path)
    html = paths.html.read_text()
    # Inline plotly.js bundle includes these tokens; a CDN-only HTML would not.
    assert "Plotly" in html
    assert html.count("<script") >= 2, "expected plotly.js script + figure script"
    assert paths.html.stat().st_size > 500_000, (
        "inlined plotly.js bundle should make the file > 500 KB; "
        "if smaller, suspect CDN mode slipped in"
    )


# ----- figure-shape paths (no file I/O) ------------------------------------


async def test_plot_has_surface_and_scatter_traces(pool: asyncpg.Pool, db_cleanup: None) -> None:
    """Figure has exactly one Surface (mesh) and one Scatter3d (market dots),
    and the scatter carries every liquid OTM point that entered the fit."""
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time)
    fig = build_figure(report)

    assert len(fig.data) == 2
    surface_traces = [t for t in fig.data if isinstance(t, go.Surface)]
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
    assert len(surface_traces) == 1
    assert len(scatter_traces) == 1

    surface = surface_traces[0]
    scatter = scatter_traces[0]

    # Mesh: x is k_grid (50 pts), y is t_grid (50 pts), z is (50, 50).
    assert len(surface.x) == 50
    assert len(surface.y) == 50
    assert surface.z.shape == (50, 50)

    # Dots: one per liquid OTM market point in the report.
    assert len(scatter.x) == report.n_points_total
    assert len(scatter.y) == report.n_points_total
    assert len(scatter.z) == report.n_points_total
    # All dots in vol % (Deribit BTC is broadly 20-80% in this range).
    z_min, z_max = float(min(scatter.z)), float(max(scatter.z))
    assert 10.0 < z_min and z_max < 100.0, (
        f"dot z-values should be vol-percent, got [{z_min}, {z_max}]"
    )

    # Dot T values lie strictly inside the mesh T range — no extrapolation.
    t_min, t_max = float(min(surface.y)), float(max(surface.y))
    for t in scatter.y:
        assert t_min - 1e-12 <= float(t) <= t_max + 1e-12, (
            f"dot T={t} outside mesh T range [{t_min}, {t_max}]"
        )


async def test_title_carries_arb_flags_and_params(pool: asyncpg.Pool, db_cleanup: None) -> None:
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time)
    fig = build_figure(report)
    title = fig.layout.title.text

    assert snapshot_time.isoformat() in title
    assert "ρ=" in title and "η=" in title and "γ=" in title
    assert "butterfly_free=True" in title
    assert "calendar_free=True" in title
    assert "n_points=" in title
    assert "RMSE(w)=" in title


# ----- dropped-expiry resilience -------------------------------------------


async def test_plot_handles_fit_with_dropped_expiries(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    """If one expiry was skipped (e.g. no forward), the surface still renders
    cleanly from the remaining expiries — no crash, mesh covers the
    successfully-fitted range only."""
    payload = json.loads(FIXTURE.read_text())
    # Drop the longest-dated expiry's forward — guaranteed to have been in
    # the fitted set on the dense fixture, so dropping it provably reduces
    # the fitted count rather than landing on an already-skipped sub-week one.
    target_expiry_iso = payload["expiries"][-1]

    # Baseline: full fit (compare against this after the drop).
    full_snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        baseline = await fit_surface(conn, full_snapshot_time)
        await conn.execute(
            "TRUNCATE option_quotes, forwards, funding_rates, instruments RESTART IDENTITY CASCADE"
        )

    snapshot_time = await _load_fixture_into_db(pool, drop_forward_for_expiry=target_expiry_iso)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time)

    # The drop landed: the target is skipped for the reason we engineered,
    # and fitted-expiry count drops by exactly one.
    target_expiry = datetime.fromisoformat(target_expiry_iso)
    assert target_expiry in report.skipped_expiries
    assert "no forward" in report.skipped_expiries[target_expiry]
    assert report.n_expiries_fitted == baseline.n_expiries_fitted - 1

    # And the plot renders without raising.
    paths = render_surface(report, tmp_path)
    assert paths.html.exists() and paths.png.exists()

    # Mesh covers only the fitted T range — the dropped expiry must not be
    # in the fitted backbone.
    fig = build_figure(report)
    surface = next(t for t in fig.data if isinstance(t, go.Surface))
    fitted_t_max = float(max(surface.y))
    fitted_expiries = {e.date() for e in report.fit.backbone.expiries}
    assert target_expiry.date() not in fitted_expiries
    # Scatter point count exactly matches the report's count; both have
    # strictly fewer points than the baseline since we dropped an expiry.
    scatter = next(t for t in fig.data if isinstance(t, go.Scatter3d))
    assert len(scatter.x) == report.n_points_total
    assert report.n_points_total < baseline.n_points_total
    assert fitted_t_max > 0.0  # plot is non-degenerate


# ----- failed-fit error path (no DB needed) --------------------------------


def _failed_report() -> SurfaceFitReport:
    """Hand-built degenerate report: fit reports failure, no params/backbone."""
    empty_backbone = SSVIBackbone(expiries=(), t_years=(), theta=())
    failed_fit = SSVIFitResult(
        params=None,
        backbone=empty_backbone,
        rmse=math.nan,
        n_points=0,
        n_expiries=0,
        success=False,
        is_butterfly_free=False,
        is_calendar_free=False,
        butterfly_message="",
        calendar_message="",
        message="non-monotone backbone",
    )
    return SurfaceFitReport(
        snapshot_time=datetime(2026, 5, 25, tzinfo=UTC),
        n_expiries_total=2,
        n_expiries_fitted=0,
        n_points_total=0,
        fit=failed_fit,
        per_expiry_rmse={},
        forwards={},
        skipped_expiries={},
        market_points=[],
    )


def _single_expiry_report() -> SurfaceFitReport:
    """A report whose fit succeeded on only one expiry — can't draw a surface."""
    only_expiry = datetime(2026, 6, 26, tzinfo=UTC)
    backbone = SSVIBackbone(expiries=(only_expiry,), t_years=(0.1,), theta=(0.05,))
    from volsurface.calibration.ssvi import SSVIParams

    fit = SSVIFitResult(
        params=SSVIParams(rho=-0.3, eta=1.0, gamma=0.3),
        backbone=backbone,
        rmse=1e-4,
        n_points=8,
        n_expiries=1,
        success=True,
        is_butterfly_free=True,
        is_calendar_free=True,
        butterfly_message="ok",
        calendar_message="single expiry, nothing to check",
        message="ok",
    )
    return SurfaceFitReport(
        snapshot_time=datetime(2026, 5, 25, tzinfo=UTC),
        n_expiries_total=1,
        n_expiries_fitted=1,
        n_points_total=8,
        fit=fit,
        per_expiry_rmse={only_expiry: 1e-4},
        forwards={only_expiry: 77000.0},
        skipped_expiries={},
        market_points=[
            SurfaceMarketPoint(only_expiry, 0.1, 0.0, 0.35),
        ],
    )


def test_plot_raises_on_failed_fit(tmp_path: Path) -> None:
    """A failed fit has no surface to draw — raise, don't render garbage."""
    with pytest.raises(ValueError, match="cannot plot a failed fit"):
        render_surface(_failed_report(), tmp_path)


def test_plot_raises_on_single_expiry(tmp_path: Path) -> None:
    """One expiry has no T-dimension — explicit error rather than mystery mesh."""
    with pytest.raises(ValueError, match=">= 2 fitted expiries"):
        render_surface(_single_expiry_report(), tmp_path)
