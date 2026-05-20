# Where I left off

## Done
- Storage layer complete: db.py, config.py, tests (10 passing, all 4 logic
  points personally verified — ON CONFLICT, transactions, return counts,
  cached pool). Strong idempotency test confirmed.

## Next session: Deribit collectors (BOTH, deliberate choice)
- REST poller: use get_book_summary_by_currency (ONE call, not 878 loops),
  every 5 min, bulk-upsert via db.py. Also pull forwards + funding.
- WebSocket subscriber: top-of-book snapshots only, NOT every tick.
  Reconnect + backoff. Scoped as near-real-time layer.
- These two are independent -> first real subagent opportunity.
- Ask Claude for the kickoff prompt before starting.
