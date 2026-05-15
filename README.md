# PRISM Analyst Services

> **AI-powered equity research platform — backend API.**
> Provides health checks, agent orchestration, and data services for the PRISM frontend.

## Architecture

```
api.thequantsoft.co.in → Nginx → FastAPI (:8000)
```

Part of the PRISM platform:
- **Frontend**: [prism-analyst-platform](https://github.com/Quantsoft24/prism-analyst-platform) — Next.js 15
- **Backend**: This repo — FastAPI
- **Landing**: Bundled in frontend repo — Express.js

## Tech Stack

| Layer     | Technology                                    |
|-----------|-----------------------------------------------|
| Framework | FastAPI, Pydantic Settings                    |
| Agent     | Google ADK (Phase 2+)                         |
| LLM       | Gemini (primary) → OpenRouter (fallback)      |
| Database  | PostgreSQL — provider-agnostic (optional)      |
| Language  | Python 3.12+                                  |
| Infra     | Docker, GitHub Actions, Nginx (via frontend)  |

## Project Structure

```
prism-analyst-services/
├── src/
│   ├── main.py              # FastAPI app entrypoint
│   ├── config.py            # Pydantic Settings (all env vars)
│   ├── agents/              # Google ADK agent definitions (Phase 2)
│   ├── tools/               # ADK tool functions (Phase 2)
│   ├── routers/             # FastAPI route handlers (Phase 2)
│   ├── services/            # Business logic layer (Phase 2)
│   ├── repositories/        # Data access layer (Phase 2)
│   ├── schemas/             # Pydantic request/response models (Phase 2)
│   └── core/                # Infrastructure (DB, middleware, logging)
├── tests/                   # pytest test suite
├── .github/workflows/
│   ├── ci.yml               # Lint + test on PR
│   └── deploy.yml           # Auto-deploy on push to production
├── Dockerfile               # Production image (python:3.12-slim + uvicorn)
├── pyproject.toml            # Dependencies + dev tools config
└── .env.example              # Environment template
```

## Local Development

### Prerequisites
- Python 3.12+
- pip

### Setup
```bash
git clone https://github.com/Quantsoft24/prism-analyst-services.git
cd prism-analyst-services

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your credentials (database is optional)
```

### Run
```bash
uvicorn src.main:app --reload --port 8000
```

Visit [http://localhost:8000/health](http://localhost:8000/health)

### Test
```bash
pytest tests/ -v
ruff check src/ tests/
```

## Environment Variables

```env
# Required
HOST=0.0.0.0
PORT=8000
DEBUG=false
CORS_ORIGINS=["https://prism.thequantsoft.co.in"]
AUTH_ENABLED=false

# Database (OPTIONAL — any PostgreSQL provider)
# DB_HOST=your-host.rds.amazonaws.com
# DB_PORT=5432
# DB_NAME=postgres
# DB_USER=postgres
# DB_PASSWORD=your-password
# DB_SSL_MODE=verify-ca
# DB_SSL_ROOT_CERT=/app/global-bundle.pem

# LLM Keys (Phase 2)
# GEMINI_API_KEY=your-key
# OPENROUTER_API_KEY=your-key

# Web Search (Phase 2)
# TAVILY_API_KEY=your-key
```

## Production Deployment

### Docker (standalone)
```bash
docker build -t prism-backend .
docker run -p 8000:8000 --env-file .env prism-backend
```

### Docker Compose (with full platform)
Managed by `docker-compose.prod.yml` in the [frontend repo](https://github.com/Quantsoft24/prism-analyst-platform).

```bash
cd ~/PRISM/prism-analyst-platform
docker compose -f docker-compose.prod.yml up -d backend
```

### CI/CD
- **CI**: `ruff check` + `pytest` on every PR to `main`/`production`
- **Deploy**: Auto-deploy on push to `production` branch via SSH → Docker rebuild → health check

## API Endpoints

| Method | Path      | Description          | Status   |
|--------|-----------|----------------------|----------|
| GET    | `/health` | Health check         | ✅ Live  |
| GET    | `/`       | Service metadata     | ✅ Live  |
| GET    | `/docs`   | Swagger UI (debug)   | Debug only |
| POST   | `/chat`   | Agent chat (Phase 2) | Planned  |
| GET    | `/stream` | SSE stream (Phase 2) | Planned  |

## License

Proprietary — © 2026 TheQuantSoft. All rights reserved.
