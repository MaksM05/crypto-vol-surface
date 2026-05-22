# validation/

The IV-error harness that graders the pricer against Deribit's own published
`mark_iv`. This is the Weeks 3-4 milestone gate: max abs error
**< 0.1 vol points on liquid strikes**.

## Unit & convention note (load-bearing)

The harness computes a **spot-referenced, zero-rate** IV — exactly matching
Deribit's `mark_iv` convention:

- `market_price_usd = mark_price_btc × index_price_S` (S = `forwards.index_price`,
  the BTC index Deribit uses for delivery — NOT the forward).
- Discount factor `DF = 1.0` (Deribit's `interest_rate = 0` convention).
- Forward `F = forwards.forward_price` is used in the Black-76 formula itself.

This is **distinct from** the F-based moneyness `ln(K/F)` used later when we
fit SVI / SSVI to a smile. The pricer-side IV uses `S` for the BTC→USD
conversion because that is what Deribit inverts to publish `mark_iv`; the
surface-side moneyness uses `F` because the smile is a forward-strike object.
Mixing the two will produce a basis-sized error in the surface fit.

## Files

| File          | Purpose                                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| `iv_error.py` | `compute_iv_errors(conn, snapshot_time, ...)` → per-row + summary + histogram PNG + summary JSON.        |

## Liquid filter

Per CLAUDE.md: `open_interest > 10 AND (best_ask − best_bid) / mark_price < 0.05`.
Rows with null bid/ask or zero mark_price are not "liquid" (no spread to evaluate).
