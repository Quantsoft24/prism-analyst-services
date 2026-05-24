FROM python:3.12-slim

WORKDIR /app

# System deps — gcc for any C extensions (asyncpg has wheels for slim,
# but keep gcc in case future deps don't).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — `pip install .` reads pyproject.toml.
# A separate COPY step for pyproject keeps this layer cached when only
# source files change.
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Source + migrations + alembic config + integration registry.
# Alembic is shipped in the image so production can run `alembic upgrade head`
# without rebuilding (see deploy.yml).
# `config/` carries the integration registry (`integrations.yml`) — without
# this COPY, the registry loads zero tools and `/api/v1/integrations` returns
# an empty list. The path is set by ``settings.INTEGRATIONS_REGISTRY_PATH``.
COPY src/ src/
COPY alembic/ alembic/
COPY config/ config/
COPY alembic.ini .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
