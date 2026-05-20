# Crypto Vol Surface Builder — v1 Scope

## One-sentence definition of done

A live BTC volatility surface (Deribit) that updates every 5 minutes, with a 6-month historical scrubber, deployed at a public URL, and validated against Deribit's own mark IV to within 0.1 vol points on liquid strikes.

If a feature does not contribute to this, it is v2.

## In scope for v1

- BTC options on Deribit only
- WebSocket + REST ingestion → TimescaleDB
- Black-76 pricer (inverse-contract adjusted), IV solver, analytic Greeks
- Forward curve from perp funding + dated futures basis
- Per-expiry SVI calibration + SSVI full surface (arbitrage-free)
- Historical snapshots at 5-min granularity
- Realized vol vs ATM IV time series
- Term structure (ATM IV by tenor)
- 25-delta risk reversal + butterfly time series
- PCA on daily surface changes (level / slope / curvature)
- FastAPI backend + Next.js + Plotly.js frontend
- Public deployment (Hetzner or Fly.io + Vercel)
- One published research note in `RESEARCH.md`

## Explicitly out of scope (v2 parking lot)

- ETH (add post-v1, ~1 day of work)
- OKX / Bybit cross-venue surfaces
- Dislocation alerts / Telegram / Discord bots
- Heston / SABR alongside SSVI
- Options backtesting framework
- User accounts, auth, paid tier
- Mobile-optimized UI
- Multi-currency / USDT-margined options

## Non-negotiables

1. Pricer validated against Deribit `mark_iv` with an error histogram committed in `README.md`
2. Unit tests for: put-call parity, no-butterfly arb, no-calendar arb, Greeks via bump-and-revalue
3. Inverse contract math correct (delta, P&L, forward) — tested, not assumed
4. Public GitHub from day 1, steady commits (not one giant dump)
5. `README.md` explains the v1 scope and links to at least one concrete finding
6. `docker compose up` works from a clean clone — reviewer can run the project in <5 minutes

## Phase milestones

- [ ] **Week 1** — Ingester live, 24h of BTC option data captured, smile plot from raw `mark_iv`
- [ ] **Week 2** — Tardis backfill (6 months), schema stable, no further migrations expected
- [ ] **Weeks 3–4** — Pricer + IV solver, error histogram vs Deribit < 0.1 vol pts on liquid strikes
- [ ] **Weeks 5–6** — SVI per-expiry, SSVI full surface, no-arb unit tests passing on every snapshot
- [ ] **Weeks 7–8** — Analytics module: RV/IV, term structure, PCA, 25Δ RR & BF time series
- [ ] **Weeks 9–10** — FastAPI + Next.js dashboard deployed, public URL live, WebSocket push working
- [ ] **Week 11** — Within-Deribit dislocation analytics (smile vs realized, calendar spread Z-scores)
- [ ] **Week 12** — `RESEARCH.md` write-up, README polish, 60-second demo video for the CV

## Anti-scope-creep rules

- New idea mid-project → GitHub issue with `v2` label, not a branch
- New venue → v2
- "Wouldn't it be cool if..." → v2
- Refactor that doesn't unblock a milestone → end of phase, not now
- The only valid reason to amend this doc is an explicit decision recorded in the commit message
