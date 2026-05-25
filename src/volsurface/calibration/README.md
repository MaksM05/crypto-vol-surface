# calibration/

Pure-numpy calibration of volatility-surface parameterisations. No I/O,
no logging, no DB knowledge — same purity rule as `pricer/`.

## Files

| File      | Purpose                                                                          |
| --------- | -------------------------------------------------------------------------------- |
| `svi.py`  | Raw SVI per-expiry smile: model, fitter, Gatheral–Durrleman no-butterfly check.  |
| `ssvi.py` | SSVI surface (power-law φ): joint fitter, both no-arb checks.                    |

## Where the DB / plot orchestration lives

`analytics/smile_fit.py` and `analytics/surface_fit.py` do the I/O dance:
pull liquid quotes for one expiry / all expiries, solve IV, call into
`calibration/`, render diagnostics. `calibration/` itself never reads from
the database.

## No-arb enforcement — honest statement

Butterfly and calendar conditions are enforced with **different strengths**:

- **Butterfly** (both SVI single-smile and SSVI surface): enforced
  *during the fit*. The single-SVI check is post-hoc on a dense `g(k)`
  grid; the SSVI fit goes further — the two Gatheral–Jacquier inequalities
  ((C1) `θ·φ·(1+|ρ|) < 4`, (C2) `θ·φ²·(1+|ρ|) ≤ 4`) are passed to SLSQP
  as hard inequality constraints, so the optimiser stays inside the
  butterfly-free feasible region throughout iteration.
- **Calendar** (SSVI only — a single smile has no calendar dimension):
  enforced by **parameter restriction** (`γ ∈ [0, 0.5]`) and **monotone
  backbone preprocessing** (the fit refuses a non-monotone backbone), then
  **verified numerically post-fit** on a wide `k`-grid for smile
  crossings. This is *not* "arb-free by construction" — it is
  "arb-free by restriction and post-hoc verification." Do not claim
  otherwise in summaries or papers.

## Mathematical convention

Total variance `w(k) = σ_BS²·T` as a function of log-moneyness `k = ln(K/F)`.
We fit `w` directly (not `σ`), because:

- the no-arb conditions are stated on `w`,
- the SVI parameterisation is for `w`,
- it composes cleanly with SSVI in the next slice.

Implied vol for display is `σ_BS = √(w/T)`; the orchestrator does that
conversion when it plots.

## What's NOT here

- Non-power-law `φ` (heston-like or Roger-Tehranchi) — only the power-law
  form is in scope for v1.
- Adaptive isotonic regression when the observed backbone is non-monotone —
  the fit refuses such inputs and the caller decides what to do.
