# PRISM Analyst Services

> **AI-powered equity research platform — backend API.**
> FastAPI + PostgreSQL + Google ADK (Phase 3). India-first agentic workspace
> for financial analysts.

## Architecture

```
api.thequantsoft.co.in → Nginx → FastAPI (:8000) → PostgreSQL (Neon / RDS)
```

Companion repos in the PRISM platform:
- **Frontend**: [prism-analyst-platform](https://github.com/Quantsoft24/prism-analyst-platform) — Next.js 15
- **Landing**: Bundled in frontend repo — Express.js

## Tech stack

| Layer | Choice |
|-------|--------|
| Web framework | FastAPI + Pydantic v2 |
| ORM / migrations | SQLAlchemy 2.x (async) + Alembic |
| Database | PostgreSQL 16 (Neon for dev/staging, AWS RDS for prod) |
| Agent runtime | Google ADK (Phase 3) |
| LLM routing | Gemini (primary) → OpenRouter (fallback) via LiteLLM (Phase 2) |
| Tests | pytest + httpx async client + real Postgres |
| CI | GitHub Actions (lint + migrate + test against Postgres service) |
| Language | Python 3.12+ |

## Project structure

```
prism-analyst-services/
├── src/
│   ├── main.py              # FastAPI app entrypoint + lifespan
│   ├── config.py            # Pydantic Settings (env-driven)
│   ├── core/
│   │   ├── database.py      # Async engine, session factory, FastAPI dep
│   │   └── auth.py          # Auth dependency (dev-mode stub for now)
│   ├── models/              # SQLAlchemy ORM models
│   │   ├── base.py          # Declarative base + mixins
│   │   ├── firm.py          # Firm (tenant)
│   │   ├── user.py          # User, FirmMembership
│   │   └── company.py       # Company, CompanyAlias
│   ├── repositories/        # Data access layer
│   │   └── company_repo.py
│   ├── schemas/             # Pydantic request/response shapes
│   │   ├── common.py        # Pagination envelope
│   │   └── company.py
│   ├── routers/             # FastAPI route modules
│   │   └── companies.py     # GET /api/v1/companies (+ /{id_or_ticker})
│   ├── agents/              # Google ADK agents (Phase 3)
│   ├── tools/               # Agent-callable tools (Phase 2+)
│   └── services/            # Business logic layer (Phase 2+)
├── alembic/
│   ├── env.py
│   └── versions/            # Numbered migration files
├── tests/
│   ├── conftest.py          # Async DB fixtures (savepoint rollback)
│   ├── test_health.py
│   └── test_companies.py    # End-to-end against real Postgres
├── .github/workflows/
│   ├── ci.yml               # Lint + Alembic + pytest with PG service
│   └── deploy.yml           # SSH deploy on push to production
├── alembic.ini
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## Local development

### Prerequisites

- Python 3.12+
- PostgreSQL 16 (one of):
  - **[Neon](https://neon.tech)** — free tier, recommended for dev/staging.
    Sign up, create a project, copy the connection string.
  - **Local Docker:** `docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:16-alpine`

### Setup

```bash
git clone https://github.com/Quantsoft24/prism-analyst-services.git
cd prism-analyst-services

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"

cp .env.example .env
# Edit .env — at minimum set DATABASE_URL.
```

### Apply migrations + run

```bash
alembic upgrade head           # Creates schema + seeds 10 NSE companies
uvicorn src.main:app --reload --port 8000
```

Open:
- API: <http://localhost:8000/api/v1/companies>
- Health: <http://localhost:8000/health>
- Swagger UI: <http://localhost:8000/docs> (DEBUG=true only)

### Tests

Tests hit a **real** Postgres — no mocks. Locally, create a separate test DB:

```bash
# One-time: create the test database
createdb prism_test
# (or via docker: docker exec -it <pg-container> psql -U postgres -c "CREATE DATABASE prism_test")

# Apply migrations to the test DB
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/prism_test \
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/prism_test \
alembic upgrade head

# Run tests
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/prism_test \
pytest tests/ -v
```

CI does the same automatically against a Postgres service container.

### Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Database

### Migrations

```bash
alembic upgrade head                            # apply all
alembic downgrade -1                            # roll back one
alembic revision --autogenerate -m "add foo"    # create a new migration
```

Migrations are numbered with the date prefix (e.g. `20260517_0001_*`) for
chronological clarity. Both schema and seed migrations live in
`alembic/versions/`.

### Provider notes

- **Neon (recommended for dev/staging).** `DATABASE_URL` is a single string;
  enable `?sslmode=require`. Set `DB_SSL_MODE=require`. pgvector is
  pre-installed when we need it in Phase 2.
- **AWS RDS (prod).** Set `DB_SSL_MODE=verify-ca` and download the AWS RDS
  CA bundle. Use `ap-south-1` (Mumbai) for data residency.

## API endpoints (Phase 1)

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/health` | Load-balancer health probe | none |
| GET | `/` | Service metadata | none |
| GET | `/docs` | Swagger UI | DEBUG only |
| GET | `/api/v1/companies` | Paginated list with search / sector / exchange filters | firm context |
| GET | `/api/v1/companies/{id_or_ticker}` | Detail (UUID or NSE ticker) | firm context |

Auth is a dev-mode stub in Slice 1 — set `X-Dev-Firm: QUANTSOFT` header, or
omit and we default to `DEV_FIRM_ID`. Phase 1 W3 wires real Clerk JWT.

## Production deployment

### Docker

```bash
docker build -t prism-backend .
docker run -p 8000:8000 --env-file .env prism-backend
```

### Docker Compose (full platform)

Managed by `docker-compose.prod.yml` in the
[frontend repo](https://github.com/Quantsoft24/prism-analyst-platform).

```bash
cd ~/PRISM/prism-analyst-platform
docker compose -f docker-compose.prod.yml up -d backend
```

The deploy workflow does NOT auto-run migrations yet (Phase 1 follow-up).
Run `alembic upgrade head` manually after deploy for now.

### CI / CD

- **CI** (`.github/workflows/ci.yml`): ruff lint → Alembic upgrade head → pytest with coverage, all against a Postgres 16 service container.
- **Deploy** (`.github/workflows/deploy.yml`): SSH into EC2 on push to `production` → git pull → docker rebuild → health check.

## License

Proprietary — © 2026 TheQuantSoft. All rights reserved.
