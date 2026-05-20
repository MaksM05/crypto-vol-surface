# CLAUDE.md — Crypto Vol Surface Builder

## What this project is

A live BTC volatility surface builder using Deribit options data, with historical replay, SSVI calibration, and a deployed dashboard. See `SCOPE.md` for v1 boundaries — do not exceed them without an explicit decision.

## Domain knowledge — read before generating code

### Deribit inverse contracts
- BTC and ETH options on Deribit are **inverse**: coin-settled, priced and quoted in BTC/ETH, not USD.
- Standard Black-Scholes / Black-76 outputs (delta, P&L) need adjustment to express USD risk.
- Do not assume linear-payoff math. If a function touches Greeks or P&L, the inverse adjustment must be explicit and tested.
- Reference: Deribit knowledge base on inverse contract specs.

### No risk-free rate
- There is no SOFR equivalent in crypto. Do not hardcode one.
- For dated expiries: forward = matching dated future mark price.
- For tenors without a matching future: interpolate using perp funding accrual as the implied carry.
- The ATM strike on the surface follows from the **forward**, not from spot.

### Validation source of truth
- Deribit publishes `mark_iv` and analytic Greeks on every ticker.
- Every IV computed by our pricer is validated against this.
- Tolerance: **<0.1 vol points** on instruments with `bid_ask_spread / mid < 0.05`.
- An error histogram lives in `tests/validation/` and is regenerated when pricer changes.

### Wing noise
- Crypto option BBOs widen dramatically off-ATM and overnight.
- Calibration inputs must be filtered: `bid_ask_spread / mid < 0.15` AND `open_interest > 10`.
- Never calibrate on a raw chain — filter first, every time.

### SSVI arbitrage conditions
- Reference: Gatheral & Jacquier, *Arbitrage-free SVI volatility surfaces*.
- No-butterfly: total variance is convex in log-strike.
- No-calendar: total variance is non-decreasing in tenor.
- Both are enforced as **unit tests on every calibrated surface**. Never skip.

## Architecture rules

- `src/volsurface/` is library code; `notebooks/` is research only and never imported from `src/`.
- Each module has a single responsibility — see `src/volsurface/<module>/README.md` for purpose.
- Configuration via `pydantic-settings`. No globals, no magic constants, no `os.environ` reads outside `config.py`.
- Database access only through the `storage/` module — no raw SQL elsewhere.
- The `pricer/` and `calibration/` modules are pure: numpy in, numpy out, no I/O, no logging side effects.

## Code conventions

- Python 3.12
- `uv` for package management (not pip, not poetry)
- `ruff` for lint + format (config in `pyproject.toml`)
- `mypy` strict on `src/`, lenient on `notebooks/`
- Type hints on every public function
- Numpy-style docstrings on every public function
- No bare `except` — catch specific exceptions
- `async` only where it earns its keep (WebSocket ingester, FastAPI handlers). The pricer is sync numpy.

## Testing requirements

- `pytest` with `pytest-asyncio` for async paths.
- Every public function in `pricer/` and `calibration/` has at least one test.
- Mandatory tests, enforced in CI:
  - `test_put_call_parity` — forward parity within 1e-6
  - `test_greeks_bump_revalue` — analytic Greeks match bump-and-revalue within tolerance
  - `test_no_butterfly_arb` — runs on every calibrated SSVI surface
  - `test_no_calendar_arb` — runs on every calibrated SSVI surface
  - `test_iv_solver_roundtrip` — price → IV → price within 1e-8
- Fixtures in `tests/fixtures/` — small captured chain snapshots for deterministic tests, never live API calls.

## Always

- Filter chains before calibration.
- Validate pricer outputs against Deribit `mark_iv` before merging changes to the pricer.
- Use the project's forward curve, not spot, as the ATM reference.
- Commit small, often, with descriptive messages.
- Update `SCOPE.md` if and only if scope changes are explicitly agreed.

## Never

- Hardcode a risk-free rate.
- Use plain Black-Scholes assuming linear contracts on Deribit BTC/ETH options.
- Calibrate on unfiltered chains.
- Add new venues, currencies, or analytics during v1 — those live in the `v2` parking lot in `SCOPE.md`.
- Import from `notebooks/` into `src/`.
- Skip the no-arb unit tests on SSVI surfaces.
- Add a dependency without checking it builds in Docker.

## Common commands

```bash
# Setup
uv sync
docker compose up -d        # starts TimescaleDB

# Dev loop
ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest -x

# Run components
python -m volsurface.ingestion.deribit_ws
uvicorn volsurface.api.main:app --reload
```

## Ask before doing

- Adding a dependency
- Changing the database schema
- Anything labeled `v2` in `SCOPE.md`
- Anything that touches inverse-contract math without an accompanying test
- Anything that bypasses the chain filter before calibration
