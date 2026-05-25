# calibration/

Pure-numpy calibration of volatility-surface parameterisations. No I/O,
no logging, no DB knowledge — same purity rule as `pricer/`.

## Files

| File     | Purpose                                                                 |
| -------- | ----------------------------------------------------------------------- |
| `svi.py` | Raw SVI per-expiry smile: model, fitter, Gatheral–Durrleman no-butterfly check. |

## Where the DB / plot orchestration lives

`analytics/smile_fit.py` does the I/O dance: pulls liquid quotes for one
expiry, solves IV, calls `fit_svi`, runs the butterfly check, renders the
plot. `calibration/` itself never reads from the database.

## Mathematical convention

Total variance `w(k) = σ_BS²·T` as a function of log-moneyness `k = ln(K/F)`.
We fit `w` directly (not `σ`), because:

- the no-arb conditions are stated on `w`,
- the SVI parameterisation is for `w`,
- it composes cleanly with SSVI in the next slice.

Implied vol for display is `σ_BS = √(w/T)`; the orchestrator does that
conversion when it plots.

## What's NOT here yet

- **SSVI** (full surface, multi-expiry parameterisation linking smiles).
- **No-calendar-arb check** (a relationship between *two* smiles — meaningful
  only with SSVI). Both land in the next slice.
