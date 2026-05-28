"""Populate company_aliases on the catalog DB — schema + data.

Creates the ``company_aliases`` table (if not exists), backfills
``company_industry.company_name`` from ``filings_index``, and algorithmically
generates 8,000-12,000 alias rows covering all companies with names.

Designed to be:
  * **Idempotent** — safe to re-run any number of times (UPSERT semantics).
  * **Transactional** — each phase commits independently; a failure in Phase 3
    does not roll back Phase 1's backfill.
  * **Standalone** — no dependency on PRISM's settings or engine. Connects
    directly to the catalog DB. Runnable as:
        python -m scripts.setup_company_aliases

Environment:
  Reads CATALOG_DATABASE_URL (or POSTGRES_URL) from .env at the project root.
  Falls back to the hardcoded default if neither is set.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Load .env from project root (2 levels up from scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── DB connection ──────────────────────────────────────────────────────────

def _get_async_url() -> str:
    """Resolve the catalog DB URL from environment, converting to asyncpg."""
    raw = (
        os.getenv("CATALOG_DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or "postgresql://stock_user:eygbAWxNVvi06sy3ppu25AKxSEi0RZwr@35.234.221.166:5434/stock_chat"
    )
    # Ensure asyncpg driver
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw


# ── Normalization (mirrors company_repo._normalize_query) ──────────────────

_NOISE_SUFFIXES = ("ltd", "limited", "pvt", "private", "inc", "corp", "corporation")


def _normalize(q: str) -> str:
    """Normalize a string for alias matching.

    MUST stay in sync with company_repo._normalize_query — both the alias
    table and the query-time lookup must normalize identically.
    """
    s = q.strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suffix in _NOISE_SUFFIXES:
        if s.endswith(" " + suffix):
            s = s[: -(len(suffix) + 1)].rstrip()
            break
    return s


# ── Alias generation logic ─────────────────────────────────────────────────

# Curated aliases from stock-chat's resolver + common market parlance.
# These override any algorithmically generated aliases for the same code.
_CURATED_ALIASES: list[tuple[str, str, float]] = [
    # (alias, code, confidence)
    ("RIL", "RELIANCE", 0.98),
    ("L&T", "LT", 0.98),
    ("LNT", "LT", 0.98),
    ("L AND T", "LT", 0.98),
    ("LARSEN", "LT", 0.90),
    ("HUL", "HINDUNILVR", 0.98),
    ("SBI", "SBIN", 0.98),
    ("LIC", "LICI", 0.98),
    ("HDFC", "HDFCBANK", 0.95),
    ("HDFC BANK", "HDFCBANK", 0.98),
    ("SUN PHARMA", "SUNPHARMA", 0.98),
    ("DR REDDYS", "DRREDDY", 0.98),
    ("DR REDDY", "DRREDDY", 0.98),
    ("POWER GRID", "POWERGRID", 0.98),
    ("INFY", "INFY", 0.98),
    ("INFOSYS", "INFY", 0.98),
    ("TAMO", "TATAMOTORS", 0.95),
    ("TATA MOTORS", "TATAMOTORS", 0.98),
    ("TATA STEEL", "TATASTEEL", 0.98),
    ("TATA POWER", "TATAPOWER", 0.98),
    ("TCS", "TCS", 0.98),
    ("WIPRO", "WIPRO", 0.98),
    ("BAJAJ FIN", "BAJFINANCE", 0.95),
    ("BAJAJ FINANCE", "BAJFINANCE", 0.98),
    ("BAJAJ AUTO", "BAJAJ-AUTO", 0.98),
    ("AXIS", "AXISBANK", 0.90),
    ("AXIS BANK", "AXISBANK", 0.98),
    ("ICICI", "ICICIBANK", 0.90),
    ("ICICI BANK", "ICICIBANK", 0.98),
    ("ICICI PRU", "ICICIPRULI", 0.95),
    ("KOTAK", "KOTAKBANK", 0.90),
    ("KOTAK BANK", "KOTAKBANK", 0.98),
    ("M&M", "M&MFIN", 0.90),
    ("MAHINDRA", "M&MFIN", 0.85),
    ("ADANI GREEN", "ADANIGREEN", 0.98),
    ("ADANI PORTS", "ADANIPORTS", 0.98),
    ("ADANI ENT", "ADANIENT", 0.95),
    ("ADANI ENTERPRISES", "ADANIENT", 0.98),
    ("MARUTI", "MARUTI", 0.98),
    ("NESTLE", "NESTLEIND", 0.95),
    ("NESTLE INDIA", "NESTLEIND", 0.98),
    ("ITC", "ITC", 0.98),
    ("NTPC", "NTPC", 0.98),
    ("ONGC", "ONGC", 0.98),
    ("BPCL", "BPCL", 0.98),
    ("IOC", "IOC", 0.98),
    ("HPCL", "HINDPETRO", 0.95),
    ("BEL", "BEL", 0.98),
    ("BHEL", "BHEL", 0.98),
    ("GAIL", "GAIL", 0.98),
    ("SAIL", "SAIL", 0.98),
    ("DLF", "DLF", 0.98),
    ("JSW STEEL", "JSWSTEEL", 0.98),
    ("JSW", "JSWSTEEL", 0.85),
    ("COAL INDIA", "COALINDIA", 0.98),
    ("TECH MAHINDRA", "TECHM", 0.98),
    ("HCLTECH", "HCLTECH", 0.98),
    ("BRITANNIA", "BRITANNIA", 0.98),
    ("ASIAN PAINTS", "ASIANPAINT", 0.98),
    ("ULTRA CEMENT", "ULTRACEMCO", 0.90),
    ("ULTRATECH", "ULTRACEMCO", 0.95),
    ("TITAN", "TITAN", 0.98),
    ("EICHER", "EICHERMOT", 0.90),
    ("EICHER MOTORS", "EICHERMOT", 0.98),
    ("INDUSIND", "INDUSINDBK", 0.95),
    ("INDUSIND BANK", "INDUSINDBK", 0.98),
    ("CIPLA", "CIPLA", 0.98),
    ("DIVIS", "DIVISLAB", 0.90),
    ("DIVIS LAB", "DIVISLAB", 0.98),
    ("ETERNAL", "ETERNAL", 0.98),
    ("ZOMATO", "ETERNAL", 0.95),
]


def _generate_aliases_from_name(
    code: str, company_name: str
) -> list[tuple[str, str, float]]:
    """Generate aliases from a company name. Returns [(alias, code, confidence)]."""
    aliases: list[tuple[str, str, float]] = []
    if not company_name:
        return aliases

    name = company_name.strip()

    # 1. Suffix-stripped short name
    short = re.sub(
        r"\s+(Ltd|Limited|Pvt|Private|Private Limited|Corp|Corporation|Inc)\.?$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    if short and short.lower() != code.lower():
        aliases.append((short, code, 0.90))

    # Also add just the first word if multi-word (e.g. "Reliance" from "Reliance Industries")
    words = short.split()
    if len(words) > 1 and len(words[0]) > 2:
        aliases.append((words[0], code, 0.80))

    # 2. First-letter acronym (3+ significant words)
    # Strip legal suffixes and stop words before computing
    significant = [
        w
        for w in name.replace("&", "and").split()
        if w.lower()
        not in (
            "ltd", "limited", "pvt", "private", "of", "the", "and",
            "&", "inc", "corp", "corporation", "-", "ltd-$",
        )
        and len(w) > 0
    ]
    if len(significant) >= 2:
        acronym = "".join(w[0].upper() for w in significant)
        if len(acronym) >= 2 and acronym.lower() != code.lower():
            aliases.append((acronym, code, 0.90 if len(significant) >= 3 else 0.80))

    # 3. &/and variants
    if "&" in name:
        and_version = name.replace("&", "and")
        aliases.append((and_version, code, 0.90))
        # Also stripped version without & or "and"
        no_amp = name.replace("&", " ").replace("  ", " ").strip()
        no_amp = re.sub(
            r"\s+(Ltd|Limited|Pvt|Private|Private Limited)\.?$",
            "", no_amp, flags=re.IGNORECASE,
        ).strip()
        if no_amp and no_amp != short:
            aliases.append((no_amp, code, 0.85))
    elif " and " in name.lower():
        amp_version = re.sub(r"\band\b", "&", name, flags=re.IGNORECASE)
        aliases.append((amp_version, code, 0.90))

    return aliases


# ── Main execution ─────────────────────────────────────────────────────────

async def main() -> None:
    url = _get_async_url()
    logger.info("Connecting to catalog DB...")
    engine = create_async_engine(url, echo=False)

    async with engine.begin() as conn:
        # ── Phase 1: Backfill company_name ──────────────────────────────
        logger.info("Phase 1: Backfilling company_industry.company_name from filings_index...")
        result = await conn.execute(text("""
            UPDATE company_industry ci
            SET company_name = sub.company_name
            FROM (
                SELECT DISTINCT ON (fi.isin) fi.isin, fi.company_name
                FROM filings_index fi
                WHERE fi.isin IS NOT NULL
                  AND fi.company_name IS NOT NULL
                  AND fi.company_name != ''
                ORDER BY fi.isin, fi.announcement_dt DESC
            ) sub
            WHERE ci.isin = sub.isin
              AND (ci.company_name IS NULL OR ci.company_name = '')
        """))
        backfilled = result.rowcount
        logger.info("Phase 1 complete: %d company names backfilled", backfilled)

    async with engine.begin() as conn:
        # ── Phase 2: Create company_aliases table ───────────────────────
        logger.info("Phase 2: Creating company_aliases table (if not exists)...")

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS company_aliases (
                id          SERIAL PRIMARY KEY,
                alias       TEXT NOT NULL,
                alias_norm  TEXT NOT NULL,
                code        TEXT NOT NULL,
                source      TEXT NOT NULL DEFAULT 'algo',
                confidence  FLOAT NOT NULL DEFAULT 0.8,
                created_at  TIMESTAMPTZ DEFAULT now(),
                UNIQUE(alias_norm, code)
            )
        """))

        # Create indexes (IF NOT EXISTS)
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_aliases_exact
            ON company_aliases (alias_norm)
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_aliases_code
            ON company_aliases (code)
        """))

        # pg_trgm GIN index — requires the extension (already installed)
        try:
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_aliases_trgm
                ON company_aliases USING gin (alias_norm gin_trgm_ops)
            """))
            logger.info("pg_trgm GIN index created/verified")
        except Exception as e:
            logger.warning("Could not create pg_trgm index (extension missing?): %s", e)

        logger.info("Phase 2 complete: company_aliases table ready")

    async with engine.begin() as conn:
        # ── Phase 3: Generate and upsert aliases ────────────────────────
        logger.info("Phase 3: Generating aliases...")

        # 3a. Fetch all companies with names
        rows = (await conn.execute(text(
            "SELECT code, company_name, industry_rank FROM company_industry "
            "WHERE company_name IS NOT NULL AND company_name != '' "
            "ORDER BY code"
        ))).fetchall()
        logger.info("Found %d companies with names", len(rows))

        # 3b. Build the alias candidate list
        # Structure: {alias_norm: (alias, code, confidence, industry_rank)}
        # If multiple codes claim the same alias_norm, keep the one with
        # the best industry_rank (lower = more important company).
        candidates: dict[str, tuple[str, str, float, int]] = {}
        duplicates_skipped = 0

        def _add_candidate(alias: str, code: str, confidence: float, rank: int, source: str = "algo") -> None:
            nonlocal duplicates_skipped
            norm = _normalize(alias)
            if not norm or len(norm) < 2:
                return  # too short to be useful
            key = norm
            existing = candidates.get(key)
            if existing is not None:
                if existing[1] == code:
                    # Same code, update confidence if higher
                    if confidence > existing[2]:
                        candidates[key] = (alias, code, confidence, rank)
                    return
                # Different code claims the same alias — keep higher-ranked
                if rank < existing[3]:
                    candidates[key] = (alias, code, confidence, rank)
                    duplicates_skipped += 1
                else:
                    duplicates_skipped += 1
                return
            candidates[key] = (alias, code, confidence, rank)

        # 3c. Curated aliases first (highest priority)
        valid_codes = {r[0] for r in rows}
        curated_count = 0
        for alias, code, confidence in _CURATED_ALIASES:
            if code in valid_codes:
                _add_candidate(alias, code, confidence, 0, "curated")
                curated_count += 1
            else:
                logger.warning("Curated alias '%s' -> '%s' skipped: code not in catalog", alias, code)

        # 3d. Algorithmic aliases from company names
        algo_count = 0
        for code, company_name, industry_rank in rows:
            rank = industry_rank if industry_rank is not None else 9999
            generated = _generate_aliases_from_name(code, company_name)
            for alias, target_code, confidence in generated:
                _add_candidate(alias, target_code, confidence, rank)
                algo_count += 1

        # 3e. BSE code cross-references
        bse_rows = (await conn.execute(text("""
            SELECT DISTINCT ci.code, fi.scrip_cd, ci.industry_rank
            FROM company_industry ci
            JOIN filings_index fi ON ci.isin = fi.isin
            WHERE fi.scrip_cd IS NOT NULL
              AND fi.scrip_cd != ''
              AND fi.scrip_cd != ci.code
        """))).fetchall()
        bse_count = 0
        for code, scrip_cd, rank in bse_rows:
            _add_candidate(scrip_cd, code, 0.99, rank or 9999, "bse_xref")
            bse_count += 1

        logger.info(
            "Generated %d candidates: %d curated, %d algorithmic, %d BSE cross-refs, %d duplicates skipped",
            len(candidates), curated_count, algo_count, bse_count, duplicates_skipped,
        )

        # 3f. Upsert into company_aliases
        param_list = []
        for alias_norm, (alias, code, confidence, _rank) in candidates.items():
            source = "curated" if any(
                a == alias and c == code for a, c, _ in _CURATED_ALIASES
            ) else "algo"
            param_list.append({
                "alias": alias,
                "alias_norm": alias_norm,
                "code": code,
                "source": source,
                "confidence": confidence,
            })

        batch_size = 1000
        inserted = 0
        errors = 0
        logger.info("Starting batch upsert of %d aliases...", len(param_list))
        for i in range(0, len(param_list), batch_size):
            batch = param_list[i : i + batch_size]
            try:
                await conn.execute(
                    text("""
                        INSERT INTO company_aliases (alias, alias_norm, code, source, confidence)
                        VALUES (:alias, :alias_norm, :code, :source, :confidence)
                        ON CONFLICT (alias_norm, code) DO UPDATE
                        SET alias = EXCLUDED.alias,
                            source = EXCLUDED.source,
                            confidence = EXCLUDED.confidence,
                            created_at = now()
                    """),
                    batch
                )
                inserted += len(batch)
                logger.info(
                    "Upserted batch %d/%d (%d/%d aliases total)...",
                    (i // batch_size) + 1,
                    (len(param_list) + batch_size - 1) // batch_size,
                    inserted,
                    len(param_list)
                )
            except Exception as e:
                logger.warning("Batch upsert failed, falling back to individual inserts for this batch: %s", e)
                for item in batch:
                    try:
                        await conn.execute(
                            text("""
                                INSERT INTO company_aliases (alias, alias_norm, code, source, confidence)
                                VALUES (:alias, :alias_norm, :code, :source, :confidence)
                                ON CONFLICT (alias_norm, code) DO UPDATE
                                SET alias = EXCLUDED.alias,
                                    source = EXCLUDED.source,
                                    confidence = EXCLUDED.confidence,
                                    created_at = now()
                            """),
                            item
                        )
                        inserted += 1
                    except Exception as ex:
                        errors += 1
                        if errors <= 10:
                            logger.warning(
                                "Failed to upsert alias '%s' -> '%s': %s",
                                item["alias"], item["code"], ex,
                            )
                        if errors == 11:
                            logger.warning("Suppressing further upsert error logs...")

        logger.info(
            "Phase 3 complete: %d aliases upserted, %d errors",
            inserted, errors,
        )

    # ── Summary ─────────────────────────────────────────────────────────
    async with engine.connect() as conn:
        total = (await conn.execute(text(
            "SELECT COUNT(*) FROM company_aliases"
        ))).scalar()
        distinct_codes = (await conn.execute(text(
            "SELECT COUNT(DISTINCT code) FROM company_aliases"
        ))).scalar()
        logger.info(
            "=== SUMMARY === %d total aliases covering %d companies "
            "(%d company_names backfilled)",
            total, distinct_codes, backfilled,
        )

    await engine.dispose()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
