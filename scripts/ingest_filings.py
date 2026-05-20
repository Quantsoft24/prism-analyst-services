"""Ingestion CLI — run the pipeline over the source registry.

Usage (from prism-analyst-services/, with .venv active):

    python -m scripts.ingest_filings                 # all sources in the registry
    python -m scripts.ingest_filings --ticker MOIL   # just one company
    python -m scripts.ingest_filings --force         # re-ingest even if fingerprint matches
    python -m scripts.ingest_filings --registry config/ingestion_sources.yml

Reads the declarative registry (config/ingestion_sources.yml), runs the
``IngestionService`` for each source, and prints a per-source result table.

This is deliberately a plain script, not an agent — ingestion is ETL. In
Year 2 these calls get wrapped in Prefect flows for scheduling without
changing the pipeline internals.

Requirements before running:
  * DATABASE_URL set + migrations applied (alembic upgrade head)
  * GEMINI_API_KEY set (embeddings)
  * If PARSER_BACKEND=docling: the docling sidecar running on DOCLING_SERVICE_URL
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.config import settings
from src.core.database import dispose_engine, init_engine, session_scope
from src.services.ingestion import IngestionService, load_registry
from src.services.model_router import dispose_router, init_router


async def _run(registry_path: str, ticker: str | None, force: bool) -> int:
    # Bring up the router (needed for the embedder) + DB engine, mirroring the
    # app lifespan so the script runs in the same conditions as the server.
    if settings.MODEL_ROUTER_ENABLED:
        keys = settings.gemini_api_keys
        if not keys:
            print("ERROR: no GEMINI_API_KEY configured — embeddings will fail.", file=sys.stderr)
            return 2
        init_router(api_keys=keys)
    init_engine()

    try:
        registry = load_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading registry: {exc}", file=sys.stderr)
        return 2

    sources = registry.for_ticker(ticker) if ticker else registry.all()
    if not sources:
        print(f"No sources found{' for ' + ticker if ticker else ''} in {registry_path}.")
        return 1

    print(f"Ingesting {len(sources)} source(s) · parser={settings.PARSER_BACKEND}\n")
    results = []
    # Each source ingests in its own transaction so one failure doesn't roll
    # back the others.
    for src in sources:
        async with session_scope() as session:
            service = IngestionService(session)
            result = await service.ingest(src, force=force)
            results.append(result)
            icon = {"ingested": "✅", "skipped": "⏭️", "failed": "❌"}.get(result.status, "?")
            print(
                f"  {icon} {result.ticker:<12} {result.status:<9} "
                f"chunks={result.chunk_count:<4} {result.detail}"
            )

    ok = sum(1 for r in results if r.status in ("ingested", "skipped"))
    print(f"\nDone: {ok}/{len(results)} succeeded.")
    return 0 if ok == len(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest filings from the source registry.")
    parser.add_argument("--registry", default=settings.INGESTION_REGISTRY_PATH)
    parser.add_argument("--ticker", default=None, help="Ingest only this ticker.")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already parsed.")
    args = parser.parse_args()

    async def _main() -> int:
        try:
            return await _run(args.registry, args.ticker, args.force)
        finally:
            dispose_router()
            await dispose_engine()

    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
