# ingestion/

Talks to Deribit's public API. Writes only via `storage/`.

## Two components with deliberately distinct roles

| Component         | Cadence           | Source                                | Writes to DB?       | Role                                                                 |
| ----------------- | ----------------- | ------------------------------------- | ------------------- | -------------------------------------------------------------------- |
| `rest_poller.py`  | every 5 minutes   | Deribit REST (`/public/...`)          | **Yes (sole writer)** | Historical record. The 5-min snapshot stream SCOPE.md guarantees.   |
| `deribit_ws.py`   | real-time stream  | Deribit WS (`book.<instr>.none.1.100ms`) | **No**              | Live top-of-book held **in memory** for the dashboard layer (later). |

REST is the single writer to every ingestion table (`instruments`, `option_quotes`,
`forwards`, `funding_rates`). WS never touches Postgres in v1. This sidesteps the
two-writer collision problem on `option_quotes (instrument_name, time)` by
elimination — not by relying on `ON CONFLICT DO NOTHING` as a guard.

## REST poll cycle (one per `poll_interval_s`)

1. `get_instruments?kind=option` → upsert `instruments` with `last_seen = cycle_time`.
2. `get_instruments?kind=future` → build `{future_name → expiration_timestamp}` map.
   (Authoritative expiry source — never parsed from instrument names.)
3. `get_book_summary_by_currency?kind=option` → bulk-insert `option_quotes`.
   `deribit_delta` is left `NULL` in v1 (book_summary doesn't carry greeks; the
   pricer module computes delta with inverse-contract adjustment).
4. `get_book_summary_by_currency?kind=future` → for each dated future, insert one
   `forwards` row with `forward_price = mark_price`, `index_price =
   estimated_delivery_price`, `expiry` from step 2.
5. `ticker?instrument_name=BTC-PERPETUAL` → one `funding_rates` row.

## Failure semantics

`__main__.py` runs both components inside `asyncio.TaskGroup`. If either task
raises, the group cancels the other and propagates the exception so the process
fails loudly. Restart is a deploy-layer concern.
