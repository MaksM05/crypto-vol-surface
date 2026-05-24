# pricer/

Pure-numpy primitives. No I/O, no logging, no domain knowledge of Deribit's
inverse contracts. Everything is USD-denominated; conversion from the BTC
premium happens upstream in `validation/`.

## Files

| File           | Purpose                                                                    | Pure? |
| -------------- | -------------------------------------------------------------------------- | ----- |
| `black76.py`   | Forward-based Black-76 price + analytic vega, gamma, theta, standard delta | yes   |
| `iv_solver.py` | Newton-Raphson with Brent fallback. Returns `nan` on no-arb violation.     | yes   |
| `forward.py`   | `ForwardSnapshot` dataclass + Act/365 time-to-expiry helper.               | yes   |
| `inverse.py`   | Deribit-convention inverse delta: `δ_bs − C/S`. The correct hedge delta.   | yes   |

## Known limitation

`inverse.py` is validated against first-principles limits (OTM convergence,
deep-ITM call saturation at 0, deep-ITM put `|δ| > 1`) and the closed-form
formula itself — **not yet against Deribit's published `ticker.delta`**.
Our ingester leaves `option_quotes.deribit_delta` `NULL` (per the REST-
poller Q3 decision), so there is no stored ground truth to compare against.
A later slice will fetch ticker greeks for a small fixture and close that
loop.

## Conventions

- Forward `F` is in USD per BTC; strike `K` is in USD per BTC; price is in USD.
- Time-to-expiry `T` is in years, **Act/365** (see `forward.time_to_expiry_years`).
- Discount factor `df` defaults to `1.0` in callers (to match Deribit's
  `interest_rate = 0` mark-IV convention). The functions accept any `df`.
- `is_call` is boolean (or broadcastable boolean array).
