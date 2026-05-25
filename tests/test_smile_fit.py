"""Integration tests for the smile-fit orchestrator.

Reuses ``snapshot_btc_validation.json`` (9 OTM options on 26JUN26) via the
storage layer, then asserts the fit converges and is butterfly-arb-free.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest

from volsurface.analytics.smile_fit import fit_one_smile
from volsurface.storage import (
    ForwardRow,
    InstrumentRow,
    OptionQuoteRow,
    insert_forwards,
    insert_option_quotes,
    upsert_instruments,
)

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_btc_validation.json"


async def _load_fixture_into_db(pool: asyncpg.Pool) -> tuple[datetime, datetime]:
    """Insert the captured snapshot. Returns ``(snapshot_time, expiry)``."""
    payload = json.loads(FIXTURE.read_text())
    snapshot_time = datetime.fromisoformat(payload["snapshot_time"])
    expiry = datetime.fromisoformat(payload["expiry"])
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
    return snapshot_time, expiry


async def test_fit_real_snapshot_is_butterfly_free(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    """The headline gate: the captured smile fits and the result passes Durrleman."""
    snapshot_time, expiry = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_one_smile(conn, snapshot_time, expiry, output_dir=tmp_path)

    assert report.n_quotes_total == 9, "fixture has 9 captured OTM options"
    assert report.n_quotes_liquid >= 8, (
        f"liquidity filter should keep nearly all fixture rows; got {report.n_quotes_liquid}"
    )
    assert report.fit.success, f"fit did not converge: {report.fit.message}"
    assert report.fit.params is not None
    assert report.fit.is_butterfly_free, (
        f"smile failed Durrleman butterfly check: params={report.fit.params}, "
        f"message={report.fit.message}"
    )


async def test_fit_residuals_are_small(
    pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path
) -> None:
    """RMSE on total variance w should be small — these are tight-spread points."""
    snapshot_time, expiry = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        report = await fit_one_smile(conn, snapshot_time, expiry, output_dir=tmp_path)

    assert report.fit.success
    # σ_BS ~ 0.35, T ~ 0.094y → w ~ 0.011. RMSE on w should be << 1e-3.
    assert report.fit.rmse < 1e-3, (
        f"RMSE(w)={report.fit.rmse:.5f} too high for tight-spread fixture"
    )


async def test_fit_writes_plot(pool: asyncpg.Pool, db_cleanup: None, tmp_path: Path) -> None:
    snapshot_time, expiry = await _load_fixture_into_db(pool)
    async with pool.acquire() as conn:
        await fit_one_smile(conn, snapshot_time, expiry, output_dir=tmp_path)
    pngs = list(tmp_path.glob("smile_*.png"))
    assert len(pngs) == 1, f"expected one plot, got {pngs}"


async def test_fit_raises_when_no_forward(pool: asyncpg.Pool, db_cleanup: None) -> None:
    """No forwards row for the requested expiry → ValueError, not silent failure."""
    fake_time = datetime(2026, 1, 1, tzinfo=UTC)
    fake_expiry = datetime(2026, 12, 31, tzinfo=UTC)
    async with pool.acquire() as conn:
        with pytest.raises(ValueError, match="no forward"):
            await fit_one_smile(conn, fake_time, fake_expiry)
