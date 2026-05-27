# Catalog DB indexes — pg_trgm for typo-tolerant company search

> **Status (2026-05-26):** Python-side `rapidfuzz` ranking ships today. This
> document is the **PRIORITY follow-up** to install `pg_trgm` on the shared
> catalog DB once admin access is available (target: within 1 day per
> product call). It moves us from ~25 ms in-Python re-ranking to ~5–10 ms
> in-DB ranked scans on the full 4,773-row catalog — and unlocks broader
> recall on hard typos that don't share any substring with the real name.

---

## Why this lives outside `alembic/`

PRISM's alembic config targets the **primary** Postgres (PRISM-owned —
`agent_runs`, `firms`, `users`, `firm_integrations`). The catalog DB
(`company_industry`, `filings_index`, `document_texts`) is **owned by the
stock-chat service**; we read it via a separate read-only secondary engine
(`CATALOG_DATABASE_URL`). PRISM's alembic chain MUST NOT touch the catalog
schema — we'd be writing to someone else's DB without their migration
chain seeing it.

Two correct options, in priority order:

1. **Hand-run the SQL below on the catalog DB** (this sprint).
2. **Add a second alembic env** for catalog-only migrations, and coordinate
   the chain with the stock-chat team (future follow-up, not blocking).

---

## The SQL

Run as a Postgres role with `CREATE EXTENSION` privilege on the catalog DB
(typically the DB owner or a superuser). `CREATE INDEX CONCURRENTLY` is
**required** for the index step on a live table — without it, the index
build takes a long `ACCESS EXCLUSIVE` lock that blocks the stock-chat
service's writes.

```sql
-- 1. Enable trigram support. Idempotent. Owner / superuser only.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. GIN trigram indexes for typo-tolerant search on the columns PRISM hits.
--    CONCURRENTLY = no write lock; safe on a live table. Each index takes
--    ~1-3 s on the 4,773-row table.
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_company_industry_name_trgm
  ON company_industry USING gin (company_name gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_company_industry_code_trgm
  ON company_industry USING gin (code gin_trgm_ops);

-- 3. Verify. Both should appear in pg_indexes.
SELECT indexname FROM pg_indexes
 WHERE tablename = 'company_industry'
   AND indexname LIKE '%trgm%';
```

If `CREATE EXTENSION` fails with `permission denied`, the role lacks the
extension privilege — escalate to the catalog DB owner.

---

## How PRISM consumes it (after install)

Once the indexes are live:

1. Set `CATALOG_TRIGRAM_ENABLED=true` in PRISM's env (Pydantic Settings,
   default `false`). Bool flag — keeps the fallback path available if we
   ever need to disable trigram queries fast.
2. `src/repositories/company_repo.py::_fetch_stage1_candidates` gains a
   prepended trigram-similarity scan:
   ```sql
   SELECT *, similarity(company_name, :q) AS score
   FROM company_industry
   WHERE company_name % :q              -- trigram similarity > 0.3 by default
      OR code % :q
   ORDER BY score DESC NULLS LAST
   LIMIT 50;
   ```
   When this returns rows, we skip the wider ILIKE substring scan and feed
   the trigram-ranked rows straight into rapidfuzz for final ranking. When
   it returns zero, we fall through to today's substring path.
3. The Python re-rank stays as a final pass — rapidfuzz gives us a more
   nuanced score for the "did you mean" surface and handles the prefix-bias
   for ticker-shaped queries.

---

## Verification queries

After install, these queries should return Reliance Industries as the top
hit:

```sql
SELECT code, company_name, similarity(company_name, 'Reliac') AS score
  FROM company_industry
 WHERE company_name % 'Reliac'
 ORDER BY score DESC
 LIMIT 5;

-- Expected: Reliance Industries Limited at score ≈ 0.4
```

And benchmark — should be < 25 ms with the GIN index, > 200 ms without:

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM company_industry
 WHERE company_name % 'Reliac'
 ORDER BY similarity(company_name, 'Reliac') DESC
 LIMIT 5;
```

---

## Rollback

Both indexes can be dropped safely without affecting the stock-chat
service's read paths (they only use B-tree indexes on `code` / `isin` /
`industry`):

```sql
DROP INDEX CONCURRENTLY IF EXISTS ix_company_industry_name_trgm;
DROP INDEX CONCURRENTLY IF EXISTS ix_company_industry_code_trgm;
-- Leave the extension installed — it's harmless and other tools may use it.
```
