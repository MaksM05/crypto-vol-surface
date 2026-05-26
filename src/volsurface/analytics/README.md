# analytics/

Orchestration layer: pulls stored market data, runs pure modules (`pricer/`,
`calibration/`), produces derived objects and plots. Distinct from
`validation/` which grades the pricer against an external source of truth.

## Files

| File              | Purpose                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `smile_fit.py`    | Fit a single-expiry SVI smile from a stored snapshot. Plot, return diagnostics.                |
| `surface_fit.py`  | Fit a full SSVI surface jointly across expiries. Returns the report with backbone + market pts.|
| `surface_plot.py` | Render the fitted SSVI surface as an interactive Plotly 3D mesh + market dot overlay.          |

## What is and is not "real data" in the 3D plot

The 3D surface plot has two layers, and they are not the same kind of object:

- **Dots** — actual liquid OTM market observations (`σ_BS` from the IV solver
  at each fitted strike / expiry). The only "real data" on the picture.
- **Mesh** — the SSVI model evaluated on a `(k, T)` grid. Between fitted
  expiries the `T` dimension is linearly interpolated in `θ`. Surface area
  between two backbone tenors is a *model interpolation*, not a fitted
  curve. No-calendar is preserved along the interpolation (monotone
  backbone + `w` monotone in `θ` for `γ ∈ [0, 0.5]`), but the interpolated
  smiles are still model output.

The honest read of the picture is "do the dots sit on the mesh?" — the
mesh hugging the dots is what demonstrates the fit. A pretty mesh with no
dots demonstrates nothing.

## Files planned (SCOPE.md Weeks 7-8)

Term structure (ATM IV by tenor), realised-vs-implied vol series, 25Δ
risk-reversal + butterfly series, PCA on daily surface changes.
