"""Render a fitted SSVI surface as an interactive 3D mesh + market overlay.

Pure rendering: takes a :class:`SurfaceFitReport` (already produced by
``surface_fit.fit_surface``) and writes two files. No DB, no math beyond
SSVI evaluation, no IV solving — all of that already ran upstream.

What the picture is honest about
--------------------------------

- The **mesh** is the SSVI model evaluated on a (k, T) grid. Between fitted
  expiries the T dimension is **linearly interpolated in θ**; only the
  discrete backbone θ_T values are model-fitted to data. Surface area
  between two consecutive backbone tenors is a *model interpolation*, not
  a fitted curve. No-calendar holds along the interpolation because SSVI's
  ``w(k, θ)`` is monotone in θ for ``γ ∈ [0, 0.5]`` and the backbone is
  monotone (the calendar check verified both at fit time) — but the
  interpolated smiles are still model output, not market data.
- The **dots** are the actual liquid OTM market ``σ_BS`` values that
  entered the joint fit. They are the only thing on the plot that is
  "real data."

Note for the live dashboard slice (Weeks 9-10)
----------------------------------------------

This standalone deliverable inlines ``plotly.js`` (``include_plotlyjs="inline"``)
so the HTML opens offline — useful for shipping a self-contained artifact.
For the embedded dashboard, switch to ``include_plotlyjs="cdn"`` to avoid
sending ~3 MB on every page view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from volsurface.analytics.surface_fit import SurfaceFitReport
from volsurface.calibration.ssvi import ssvi_total_variance

_K_CAP = 2.5
_K_PAD_FRAC = 0.10
_MESH_OPACITY = 0.85
_PNG_WIDTH = 1200
_PNG_HEIGHT = 800
_PNG_SCALE = 2


@dataclass(frozen=True, slots=True)
class SurfacePlotPaths:
    """Filenames written by :func:`render_surface`."""

    html: Path
    png: Path


def build_figure(
    report: SurfaceFitReport,
    *,
    n_grid_k: int = 50,
    n_grid_t: int = 50,
) -> go.Figure:
    """Build the Plotly figure without writing anything. Exposed for testing.

    Raises
    ------
    ValueError
        If the fit didn't converge, has no params, or has fewer than two
        fitted expiries (a surface needs at least two backbone tenors).
    """
    if not report.fit.success or report.fit.params is None or report.fit.backbone is None:
        raise ValueError(f"cannot plot a failed fit: {report.fit.message}")
    backbone = report.fit.backbone
    if len(backbone.t_years) < 2:
        raise ValueError(
            f"need >= 2 fitted expiries to draw a surface, got {len(backbone.t_years)}"
        )

    params = report.fit.params

    # K grid: data range padded by 10% on each side, capped at ±K_CAP.
    if report.market_points:
        ks_market = [p.k for p in report.market_points]
        span = max(ks_market) - min(ks_market)
        k_min = max(min(ks_market) - _K_PAD_FRAC * span, -_K_CAP)
        k_max = min(max(ks_market) + _K_PAD_FRAC * span, _K_CAP)
    else:
        k_min, k_max = -1.0, 1.0
    k_grid = np.linspace(k_min, k_max, n_grid_k, dtype=np.float64)

    # T grid: STRICTLY within the fitted tenor range. No extrapolation past
    # the last fitted expiry — we have no fit information there.
    t_min = float(min(backbone.t_years))
    t_max = float(max(backbone.t_years))
    t_grid = np.linspace(t_min, t_max, n_grid_t, dtype=np.float64)

    # Linear interpolation of θ in T from the fitted backbone. Monotone
    # backbone → monotone interp → no new calendar arb introduced by the
    # viz between backbone points.
    theta_interp = np.interp(
        t_grid,
        np.asarray(backbone.t_years, dtype=np.float64),
        np.asarray(backbone.theta, dtype=np.float64),
    )

    # Evaluate SSVI. z shape (rows=T, cols=K) so Plotly's z[i,j] aligns
    # with y=T (rows), x=K (cols).
    z = np.zeros((n_grid_t, n_grid_k), dtype=np.float64)
    for i, (t_i, theta_i) in enumerate(zip(t_grid, theta_interp, strict=True)):
        w_row = ssvi_total_variance(k_grid, float(theta_i), params)
        z[i, :] = np.sqrt(np.maximum(w_row, 0.0) / t_i) * 100.0  # vol %

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=k_grid,
            y=t_grid,
            z=z,
            colorscale="Viridis",
            opacity=_MESH_OPACITY,
            colorbar={"title": {"text": "σ_BS (%)"}},
            name="SSVI fit",
            hovertemplate=(
                "k = %{x:.3f}<br>T = %{y:.3f}y<br>σ_BS = %{z:.2f}%<extra>SSVI mesh</extra>"
            ),
        )
    )

    if report.market_points:
        mp_text = [
            f"expiry {p.expiry.date()}<br>"
            f"k = {p.k:+.4f}<br>"
            f"T = {p.t_years:.4f}y<br>"
            f"σ_BS = {p.sigma_bs * 100:.2f}%"
            for p in report.market_points
        ]
        fig.add_trace(
            go.Scatter3d(
                x=[p.k for p in report.market_points],
                y=[p.t_years for p in report.market_points],
                z=[p.sigma_bs * 100.0 for p in report.market_points],
                mode="markers",
                marker={
                    "size": 4,
                    "color": "black",
                    "line": {"color": "white", "width": 1},
                },
                name="market (liquid OTM)",
                hovertext=mp_text,
                hoverinfo="text",
            )
        )

    p = params
    title_text = (
        f"SSVI surface  snapshot={report.snapshot_time.isoformat()}"
        f"<br>ρ={p.rho:+.4f}  η={p.eta:.4f}  γ={p.gamma:.4f}  "
        f"butterfly_free={report.fit.is_butterfly_free}  "
        f"calendar_free={report.fit.is_calendar_free}  "
        f"n_points={report.fit.n_points}  "
        f"RMSE(w)={report.fit.rmse:.5f}"
    )
    fig.update_layout(
        title={"text": title_text},
        scene={
            "xaxis": {"title": {"text": "log-moneyness  k = ln(K/F)"}},
            "yaxis": {"title": {"text": "time to expiry (years)"}},
            "zaxis": {"title": {"text": "Black-76 implied vol (%)"}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
    )
    return fig


def render_surface(
    report: SurfaceFitReport,
    output_dir: Path,
    *,
    n_grid_k: int = 50,
    n_grid_t: int = 50,
) -> SurfacePlotPaths:
    """Render the surface and write ``surface_<snapshot>.{html,png}``.

    Parameters
    ----------
    report
        Already-computed :class:`SurfaceFitReport`. Mesh comes from
        ``report.fit.params`` and ``report.fit.backbone``; dots come from
        ``report.market_points`` (the same liquid OTM observations that
        entered the joint fit).
    output_dir
        Directory to write into; created if missing.
    n_grid_k, n_grid_t
        Mesh density. 50×50 is the project default and is essentially free
        to evaluate.

    Returns
    -------
    SurfacePlotPaths
        Absolute paths to the written HTML and PNG.

    Raises
    ------
    ValueError
        If the fit didn't converge or has too few expiries to draw — see
        :func:`build_figure`.
    """
    fig = build_figure(report, n_grid_k=n_grid_k, n_grid_t=n_grid_t)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"surface_{report.snapshot_time.strftime('%Y%m%dT%H%M%S')}"
    html_path = output_dir / f"{stem}.html"
    png_path = output_dir / f"{stem}.png"
    fig.write_html(html_path, include_plotlyjs="inline", full_html=True)
    fig.write_image(
        png_path,
        format="png",
        width=_PNG_WIDTH,
        height=_PNG_HEIGHT,
        scale=_PNG_SCALE,
    )
    return SurfacePlotPaths(html=html_path, png=png_path)
