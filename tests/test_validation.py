"""Integration test for the IV-error validation harness.

Loads a captured Deribit snapshot fixture, inserts it via the storage layer,
runs the harness, and asserts max abs error < 0.1 vol pts on the liquid
subset — the Weeks 3-4 milestone gate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import asyncpg
import pytest

from volsurface.storage import (
    ForwardRow,
    InstrumentRow,
    OptionQuoteRow,
    insert_forwards,
    insert_option_quotes,
    upsert_instruments,
)
from volsurface.validation.iv_error import compute_iv_errors

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_btc_validation.json"


async def _load_fixture_into_db(
    pool: asyncpg.Pool,
) -> datetime:
    payload = json.loads(FIXTURE.read_text())
    snapshot_time = datetime.fromisoformat(payload["snapshot_time"])
    async with pool.acquire() as conn:
        await upsert_instruments(
            conn,
            [
                InstrumentRow(
                    instrument_name=i["instrument_name"],
                    currency=i["currency"],
                    strike=i["strike"],
                    option_type=i["option_type"],
                    expiry=datetime.fromisoformat(i["expiry"]),
                    contract_size=i["contract_size"],
                    creation_ts=datetime.fromisoformat(i["creation_ts"]),
                    last_seen=snapshot_time,
                )
                for i in payload["instruments"]
            ],
        )
        await insert_forwards(
            conn,
            [
                ForwardRow(
                    time=snapshot_time,
                    expiry=datetime.fromisoformat(f["expiry"]),
                    forward_price=f["forward_price"],
                    index_price=f["index_price"],
                )
                for f in payload["forwards"]
            ],
        )
        await insert_option_quotes(
            conn,
            [
                OptionQuoteRow(
                    time=snapshot_time,
                    instrument_name=q["instrument_name"],
                    mark_price=q["mark_price"],
                    best_bid=q["best_bid"],
                    best_ask=q["best_ask"],
                    open_interest=q["open_interest"],
                    deribit_mark_iv=q["deribit_mark_iv"],
                    deribit_delta=q.get("deribit_delta"),
                )
                for q in payload["option_quotes"]
            ],
        )
    return snapshot_time


async def test_iv_matches_deribit_on_liquid_strikes(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    """Max abs error < 0.1 vol pts on every liquid strike — the headline gate."""
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await compute_iv_errors(
            conn, snapshot_time, pricer_version="test", output_dir=tmp_path
        )

    # Sanity: we captured a non-trivial number of liquid strikes spanning the smile.
    assert report.n_total == 9, "fixture has 9 captured options"
    assert report.n_priced == report.n_total, "every captured option should solve"
    assert report.n_liquid >= 8, (
        f"expected nearly all captured options to be liquid (spread<5%, OI>10); "
        f"got {report.n_liquid}"
    )

    # The actual milestone gate.
    assert report.abs_error_max_liquid < 0.1, (
        f"max abs IV error {report.abs_error_max_liquid:.4f} vol pts exceeds "
        f"0.1 budget. Per-row errors: "
        + ", ".join(
            f"{r.instrument_name}={r.iv_error_pct:+.4f}"
            for r in report.rows
            if r.is_liquid and r.iv_error_pct is not None
        )
    )


async def test_validation_writes_histogram_and_summary(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        await compute_iv_errors(conn, snapshot_time, pricer_version="testver", output_dir=tmp_path)
    assert (tmp_path / "error_histogram_testver.png").exists()
    assert (tmp_path / "error_summary_testver.json").exists()
    summary = json.loads((tmp_path / "error_summary_testver.json").read_text())
    assert summary["pricer_version"] == "testver"
    assert summary["n_total"] == 9
    assert "rows" in summary and len(summary["rows"]) == 9
