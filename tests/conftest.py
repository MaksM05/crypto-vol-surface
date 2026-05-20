"""Shared pytest fixtures for storage integration tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

from volsurface.config import Settings
from volsurface.storage import close_pool, get_pool


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest_asyncio.fixture
async def pool(settings: Settings) -> AsyncGenerator[asyncpg.Pool[asyncpg.Record], None]:
    """Yield an asyncpg pool. Skip the test cleanly if TimescaleDB is unreachable."""
    try:
        p = await get_pool(settings)
        async with p.acquire() as conn:
            await conn.fetchval("SELECT 1")
            tables_present = await conn.fetchval(
                "SELECT to_regclass('public.instruments') IS NOT NULL"
            )
    except (OSError, asyncpg.PostgresError) as exc:
        await close_pool()
        pytest.skip(f"TimescaleDB not reachable at {settings.database_url}: {exc}")
    if not tables_present:
        await close_pool()
        pytest.skip(
            "schema not applied — drop the volume and re-run "
            "'docker compose up -d' so storage/schema.sql initialises the DB"
        )
    yield p
    await close_pool()


@pytest_asyncio.fixture
async def db_cleanup(pool: asyncpg.Pool[asyncpg.Record]) -> AsyncGenerator[None, None]:
    """Truncate ingestion tables before the test for isolation."""
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE option_quotes, forwards, funding_rates, instruments RESTART IDENTITY CASCADE"
        )
    yield
