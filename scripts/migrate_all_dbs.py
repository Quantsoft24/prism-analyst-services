#!/usr/bin/env python
"""Apply Alembic migrations to the primary DB and every fallback DB.

Each entry in ``DATABASE_URL`` + ``DATABASE_URL_FALLBACKS`` is an independent
database (a separate Neon project, etc.) that needs the schema before the app
can fail over to it. Run this once after adding a new fallback URL:

    python scripts/migrate_all_dbs.py

URLs that can't be reached (e.g. a Neon project whose compute allowance is
spent) are skipped with a warning — re-run later when they're back. pgvector is
enabled automatically by migration 0004 (``CREATE EXTENSION IF NOT EXISTS
vector``), so each DB gets it without extra steps.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Repo root on path so ``src.config`` imports when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings  # noqa: E402


def _safe(url: str) -> str:
    """Mask the password when printing a connection string."""
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return url or "(primary from DB_* parts)"


def main() -> int:
    primary = settings.DATABASE_URL.strip()
    fallbacks = [u.strip() for u in settings.DATABASE_URL_FALLBACKS.split(",") if u.strip()]
    # "" sentinel → let the subprocess build the primary from DB_* parts.
    targets = ([primary] if primary else [""]) + fallbacks

    failures = 0
    for i, url in enumerate(targets):
        print(f"\n=== [{i + 1}/{len(targets)}] alembic upgrade head -> {_safe(url)} ===")
        env = dict(os.environ)
        env["DATABASE_URL_FALLBACKS"] = ""  # migrate exactly this DB
        if url:
            env["DATABASE_URL"] = url
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"], env=env
        )
        if result.returncode != 0:
            failures += 1
            print(f"  !! FAILED (likely unreachable / capped) — skipping: {_safe(url)}")

    migrated = len(targets) - failures
    print(f"\nDone. {migrated}/{len(targets)} database(s) migrated.")
    # Non-zero only if EVERY target failed (so CI/automation notices a total outage).
    return 1 if migrated == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
