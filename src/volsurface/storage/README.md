# storage/

The only module in `src/volsurface/` permitted to issue raw SQL.

## Responsibility

- Own the `asyncpg` connection pool.
- Provide typed, idempotent bulk-insert / upsert functions, one per ingestion table.
- Mirror `storage/schema.sql` exactly — column names and types.

## Conflict semantics

| Table           | Behaviour                | Why                                                                    |
| --------------- | ------------------------ | ---------------------------------------------------------------------- |
| `instruments`   | `ON CONFLICT DO UPDATE`  | Static-ish metadata; rare corrections (e.g. `contract_size`) must land. |
| `option_quotes` | `ON CONFLICT DO NOTHING` | Raw observation, source of truth — never overwritten on replay.        |
| `forwards`      | `ON CONFLICT DO NOTHING` | Same reasoning.                                                        |
| `funding_rates` | `ON CONFLICT DO NOTHING` | Same reasoning.                                                        |

## What does not belong here

- Anything that computes IV, Greeks, forwards, or surfaces. Storage is dumb; analysis lives in `pricer/`, `forwards/`, `calibration/`.
- Domain transformations. Callers pass rows that already match schema column types.
