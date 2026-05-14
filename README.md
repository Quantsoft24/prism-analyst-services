# PRISM Analyst Services

AI-powered equity research platform — backend API.

## Tech Stack

- **Framework**: FastAPI
- **Agent Framework**: Google ADK (Phase 2+)
- **Database**: PostgreSQL (AWS RDS / Google Cloud SQL)
- **LLM**: Gemini (primary) → OpenRouter (fallback) → Ollama (edge)
- **Language**: Python 3.12+

## Project Structure

```
src/
├── main.py              # FastAPI app entrypoint
├── config.py            # Pydantic Settings (all env vars)
├── agents/              # Google ADK agent definitions (Phase 2)
├── tools/               # ADK tool functions (Phase 2)
├── routers/             # FastAPI route handlers (Phase 2)
├── services/            # Business logic layer (Phase 2)
├── repositories/        # Data access layer (Phase 2)
├── schemas/             # Pydantic request/response models (Phase 2)
└── core/                # Infrastructure (DB, middleware, logging)
```

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/Quantsoft24/prism-analyst-services.git
cd prism-analyst-services

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 5. Run development server
uvicorn src.main:app --reload --port 8000

# 6. Verify
# Visit http://localhost:8000/health
```

## Testing

```bash
pytest tests/ -v
```

## Deployment

```bash
docker build -t prism-backend .
docker run -p 8000:8000 --env-file .env prism-backend
```
