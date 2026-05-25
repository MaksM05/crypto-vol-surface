"""Integration test for the SSVI surface-fit orchestrator.

Loads the captured multi-expiry fixture (4 expiries × 10 OTM strikes), inserts
via the storage layer, runs the joint fit, asserts both arb conditions hold.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import asyncpg
import pytest

from volsurface.analytics.surface_fit import fit_surface
from volsurface.storage import (
    ForwardRow,
    InstrumentRow,
    OptionQuoteRow,
    insert_forwards,
    insert_option_quotes,
    upsert_instruments,
)

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_btc_surface.json"


async def _load_fixture_into_db(pool: asyncpg.Pool) -> datetime:
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


async def test_fit_real_snapshot_both_arb_free(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    """The headline gate: 4-expiry BTC surface fits + butterfly + calendar pass."""
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time, output_dir=tmp_path)

    assert report.n_expiries_total == 4, "fixture has 4 captured expiries"
    assert report.n_expiries_fitted >= 3, (
        f"need >= 3 expiries in joint fit; got {report.n_expiries_fitted} "
        f"(skipped: {report.skipped_expiries})"
    )
    assert report.fit.success, f"fit did not converge: {report.fit.message}"
    assert report.fit.params is not None
    assert report.fit.is_butterfly_free, (
        f"butterfly arb at the optimum (should be impossible with SLSQP "
        f"constraints): {report.fit.butterfly_message}"
    )
    assert report.fit.is_calendar_free, f"calendar arb post-fit: {report.fit.calendar_message}"


async def test_fit_per_expiry_rmse_reasonable(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    """Each fitted expiry's RMSE on total variance should be small."""
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_surface(conn, snapshot_time, output_dir=tmp_path)
    assert report.fit.success
    # σ~0.35, T from ~0.09y to ~0.6y, so w ~ 0.01-0.07. RMSE per expiry should
    # be well under 0.01 if SSVI captures the smile shape.
    for exp, rmse in report.per_expiry_rmse.items():
        assert rmse < 0.01, f"expiry {exp.date()} RMSE(w)={rmse:.5f} too high"


async def test_fit_writes_summary_json(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    snapshot_time = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        await fit_surface(conn, snapshot_time, output_dir=tmp_path)
    summaries = list(tmp_path.glob("surface_summary_*.json"))
    assert len(summaries) == 1
    payload = json.loads(summaries[0].read_text())
    # Spot-check the contents.
    assert payload["fit"]["success"] is True
    assert payload["fit"]["is_butterfly_free"] is True
    assert payload["fit"]["is_calendar_free"] is True
    assert payload["params"]["rho"] <= 0.0  # BTC skew is negative
    assert payload["params"]["eta"] > 0.0
    assert 0.0 <= payload["params"]["gamma"] <= 0.5
    assert len(payload["backbone"]) >= 3
    # Backbone must be sorted by t_years ascending.
    t_years = [row["t_years"] for row in payload["backbone"]]
    assert t_years == sorted(t_years)
    # θ_T also monotone (necessary for the fit to have succeeded).
    thetas = [row["theta"] for row in payload["backbone"]]
    assert all(thetas[i] >= thetas[i - 1] for i in range(1, len(thetas)))
