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

## What is NOT here yet

- **Inverse-contract delta adjustment** — see `black76.standard_delta`'s
  docstring. The textbook delta exported as `standard_delta` is not the
  USD delta of a Deribit coin-settled option; that adjustment is a planned
  separate slice and will live in `pricer/inverse.py`.

## Conventions

- Forward `F` is in USD per BTC; strike `K` is in USD per BTC; price is in USD.
- Time-to-expiry `T` is in years, **Act/365** (see `forward.time_to_expiry_years`).
- Discount factor `df` defaults to `1.0` in callers (to match Deribit's
  `interest_rate = 0` mark-IV convention). The functions accept any `df`.
- `is_call` is boolean (or broadcastable boolean array).
