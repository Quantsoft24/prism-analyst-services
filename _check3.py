import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    e = create_async_engine(
        "postgresql+asyncpg://stock_user:eygbAWxNVvi06sy3ppu25AKxSEi0RZwr"
        "@35.234.221.166:5434/stock_chat"
    )
    async with e.connect() as c:
        # Check filings_index for M&M
        r = await c.execute(text("SELECT DISTINCT company_name, scrip_cd FROM filings_index WHERE company_name ILIKE '%mahindra%' AND company_name ILIKE '%mahindra%' LIMIT 10"))
        print("Mahindra in filings_index:")
        for row in r.fetchall():
            print(f"  {row[1]:15s} {row[0]}")

        # Check filings_index columns
        r2 = await c.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='filings_index' ORDER BY ordinal_position"))
        print("\nfilings_index columns:", [x[0] for x in r2.fetchall()])

        # Check what company_industry has for Mahindra-like companies
        r3 = await c.execute(text("SELECT code, isin, industry FROM company_industry WHERE code ILIKE '%M%M%' AND LENGTH(code) <= 4 LIMIT 10"))
        print("\nShort M*M codes in company_industry:")
        for row in r3.fetchall():
            print(f"  {row[0]:10s} {row[1] or 'no-isin':20s} {row[2] or 'no-industry'}")

    await e.dispose()

asyncio.run(main())
