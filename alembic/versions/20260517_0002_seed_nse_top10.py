"""seed top-10 NSE companies + a default QUANTSOFT firm

Revision ID: 0002_seed_nse_top10
Revises: 0001_initial
Create Date: 2026-05-17

Bootstrap rows so the API is useful from the first deploy:
  - One ``firms`` row (slug=``QUANTSOFT``) — our own firm for dogfooding.
  - Ten NSE-listed companies covering the most-asked-about tickers in
    Indian equity research (matched to the watchlist mocked in the frontend).

This is a *data* migration, not a schema one. It's idempotent: running it
twice won't duplicate rows.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002_seed_nse_top10"
down_revision: str | None = "0001_initial"
branch_labels = None
depends_on = None


COMPANIES: list[dict[str, str | None]] = [
    {
        "ticker": "RELIANCE",
        "name": "Reliance Industries",
        "legal_name": "Reliance Industries Limited",
        "isin": "INE002A01018",
        "cin": "L17110MH1973PLC019786",
        "sector": "Energy",
        "industry": "Oil & Gas — Integrated",
        "website": "https://www.ril.com",
        "description": "Diversified conglomerate — oil-to-chemicals, telecom (Jio), retail, digital services, new energy.",
    },
    {
        "ticker": "TCS",
        "name": "Tata Consultancy Services",
        "legal_name": "Tata Consultancy Services Limited",
        "isin": "INE467B01029",
        "cin": "L22210MH1995PLC084781",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "website": "https://www.tcs.com",
        "description": "India's largest IT services firm — applications, infrastructure, BPS, consulting; ~$30B revenue.",
    },
    {
        "ticker": "HDFCBANK",
        "name": "HDFC Bank",
        "legal_name": "HDFC Bank Limited",
        "isin": "INE040A01034",
        "cin": "L65920MH1994PLC080618",
        "sector": "Financials",
        "industry": "Private Sector Bank",
        "website": "https://www.hdfcbank.com",
        "description": "Largest private-sector bank in India by assets; merged with HDFC Ltd in 2023.",
    },
    {
        "ticker": "INFY",
        "name": "Infosys",
        "legal_name": "Infosys Limited",
        "isin": "INE009A01021",
        "cin": "L85110KA1981PLC013115",
        "sector": "Information Technology",
        "industry": "IT Services & Consulting",
        "website": "https://www.infosys.com",
        "description": "Second-largest Indian IT services exporter; cloud, digital, AI, and engineering services.",
    },
    {
        "ticker": "ICICIBANK",
        "name": "ICICI Bank",
        "legal_name": "ICICI Bank Limited",
        "isin": "INE090A01021",
        "cin": "L65190GJ1994PLC021012",
        "sector": "Financials",
        "industry": "Private Sector Bank",
        "website": "https://www.icicibank.com",
        "description": "Second-largest private bank in India; retail, corporate, and international banking.",
    },
    {
        "ticker": "BHARTIARTL",
        "name": "Bharti Airtel",
        "legal_name": "Bharti Airtel Limited",
        "isin": "INE397D01024",
        "cin": "L74899DL1995PLC070609",
        "sector": "Communication Services",
        "industry": "Telecom",
        "website": "https://www.airtel.in",
        "description": "Pan-India + Africa telecom operator; mobile, broadband, enterprise, payments bank.",
    },
    {
        "ticker": "ITC",
        "name": "ITC",
        "legal_name": "ITC Limited",
        "isin": "INE154A01025",
        "cin": "L16005WB1910PLC001985",
        "sector": "Consumer Staples",
        "industry": "FMCG / Cigarettes / Hotels",
        "website": "https://www.itcportal.com",
        "description": "Cigarettes-led conglomerate with FMCG, hotels, agribusiness, and paperboards segments.",
    },
    {
        "ticker": "SBIN",
        "name": "State Bank of India",
        "legal_name": "State Bank of India",
        "isin": "INE062A01020",
        "cin": "L74899DL2005PLC133156",  # Note: SBI has a statutory CIN-equivalent
        "sector": "Financials",
        "industry": "Public Sector Bank",
        "website": "https://www.sbi.co.in",
        "description": "Largest public-sector bank in India by assets and branch network.",
    },
    {
        "ticker": "LT",
        "name": "Larsen & Toubro",
        "legal_name": "Larsen & Toubro Limited",
        "isin": "INE018A01030",
        "cin": "L99999MH1946PLC004768",
        "sector": "Industrials",
        "industry": "Engineering & Construction",
        "website": "https://www.larsentoubro.com",
        "description": "India's largest EPC company; infrastructure, hydrocarbon, defence, and IT services (LTIMindtree, LTTS).",
    },
    {
        "ticker": "MOIL",
        "name": "MOIL",
        "legal_name": "MOIL Limited",
        "isin": "INE490G01020",
        "cin": "L99999MH1962GOI012398",
        "sector": "Materials",
        "industry": "Mining — Manganese Ore",
        "website": "https://www.moil.nic.in",
        "description": "Government-of-India PSU; largest manganese-ore producer in India.",
    },
]


def upgrade() -> None:
    bind = op.get_bind()

    # ── Firm ──
    bind.execute(
        sa.text(
            """
            INSERT INTO firms (slug, name, subscription_tier, country)
            VALUES (:slug, :name, :tier, :country)
            ON CONFLICT (slug) DO NOTHING
            """
        ),
        {"slug": "QUANTSOFT", "name": "Quantsoft (dev)", "tier": "trial", "country": "IN"},
    )

    # ── Companies ──
    for c in COMPANIES:
        bind.execute(
            sa.text(
                """
                INSERT INTO companies
                    (ticker, name, legal_name, exchange, isin, cin, sector, industry, country, website, description)
                VALUES
                    (:ticker, :name, :legal_name, 'NSE', :isin, :cin, :sector, :industry, 'IN', :website, :description)
                ON CONFLICT (exchange, ticker) DO NOTHING
                """
            ),
            c,
        )

        # Also add a name alias so search by full name works.
        bind.execute(
            sa.text(
                """
                INSERT INTO company_aliases (company_id, kind, value)
                SELECT id, 'name', :alias
                FROM companies
                WHERE exchange = 'NSE' AND ticker = :ticker
                ON CONFLICT (kind, value) DO NOTHING
                """
            ),
            {"ticker": c["ticker"], "alias": c["name"]},
        )


def downgrade() -> None:
    bind = op.get_bind()
    tickers = [c["ticker"] for c in COMPANIES]
    bind.execute(
        sa.text("DELETE FROM companies WHERE exchange = 'NSE' AND ticker = ANY(:tickers)"),
        {"tickers": tickers},
    )
    bind.execute(sa.text("DELETE FROM firms WHERE slug = 'QUANTSOFT'"))
