import asyncio, httpx, asyncpg
from volsurface.config import Settings
from volsurface.ingestion.rest_poller import run_one_cycle

async def main():
    s = Settings()
    http = httpx.AsyncClient(base_url="https://www.deribit.com/api/v2", timeout=30)
    pool = await asyncpg.create_pool(s.database_url)
    try:
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE option_quotes, forwards, funding_rates, instruments RESTART IDENTITY CASCADE")
        stats = await run_one_cycle(http, pool)
        print("cycle done:", stats)
    finally:
        await http.aclose()
        await pool.close()

asyncio.run(main())
