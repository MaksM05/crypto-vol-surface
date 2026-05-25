# analytics/

Orchestration layer: pulls stored market data, runs pure modules (`pricer/`,
`calibration/`), produces derived objects and plots. Distinct from
`validation/` which grades the pricer against an external source of truth.

## Files

| File           | Purpose                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| `smile_fit.py` | Fit a single-expiry SVI smile from a stored snapshot. Plot, return diagnostics. |

## Files planned (SCOPE.md Weeks 7-8)

Term structure (ATM IV by tenor), realised-vs-implied vol series, 25Δ
risk-reversal + butterfly series, PCA on daily surface changes. SSVI
full-surface fit (next slice) will share orchestration patterns here.
