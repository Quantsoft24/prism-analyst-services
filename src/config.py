"""PRISM Analyst Services — Configuration Module.

Centralized configuration using Pydantic Settings. All environment variables
are defined here with defaults. Supports any PostgreSQL provider
(Neon for dev/staging, AWS RDS for prod, local Docker, etc.).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    API_PREFIX: str = "/api/v1"

    # ── Database (PostgreSQL — provider-agnostic) ──
    # Either set DATABASE_URL directly (Neon / AWS RDS style),
    # or set DB_* parts individually and we'll assemble it.
    DATABASE_URL: str = ""
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "prism"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_SSL_MODE: str = "prefer"

    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_PRE_PING: bool = True
    DB_ECHO: bool = False

    # ── LLM: Gemini API Keys (round-robin) ──
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEY_1: str = ""
    GEMINI_API_KEY_2: str = ""
    GEMINI_API_KEY_3: str = ""
    GEMINI_API_KEY_4: str = ""

    # ── LLM: OpenRouter (fallback) ──
    OPENROUTER_API_KEY: str = ""

    # ── Web Search ──
    TAVILY_API_KEY: str = ""

    # ── Auth (Phase 1 W3 will wire Clerk; until then, dev-mode firm ID) ──
    AUTH_ENABLED: bool = False
    DEV_FIRM_ID: str = "QUANTSOFT"

    # ── Agent runtime (Google ADK) ──
    # Provider: "ai_studio" (free key from aistudio.google.com) or "vertex"
    # (GCP project + region). Free tier is plenty for dev; move to Vertex
    # when paying customers + data residency matter (Phase 4+).
    ADK_PROVIDER: str = "ai_studio"
    ADK_VERTEX_PROJECT: str = ""
    ADK_VERTEX_LOCATION: str = "asia-south1"

    # Default models per task tier — per architecture doc §1, we route
    # cheap/fast steps to Flash and quality steps to Pro.
    #
    # NOTE: with Slice 4's ``ModelRouter``, agents declare a ``model_tier``
    # ("fast" / "quality" / "classify" / "embedding") instead of a specific
    # model name. These two settings are kept for explicit per-agent overrides
    # and for the legacy non-routed path; production agents should use tiers.
    AGENT_MODEL_FAST: str = "gemini-2.5-flash"
    AGENT_MODEL_QUALITY: str = "gemini-2.5-pro"

    # Hard caps per agent invocation — fail-safes, not optimization knobs.
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TIMEOUT_SECONDS: int = 60
    AGENT_MAX_COST_INR: float = 10.0  # abort if estimated cost exceeds this

    # ── Model Router (Slice 4 — multi-key + multi-model fallback) ──
    # When enabled, agents resolve ``model_tier`` through ``ModelRouter``
    # backed by ``litellm.Router``. Turn off to bypass the router entirely
    # (single-model legacy path) — useful for narrow LLM regression tests.
    MODEL_ROUTER_ENABLED: bool = True
    # Seconds a deployment stays cooled-down after a 429 / capacity error.
    # 60s matches Gemini's per-minute rate-limit window; raise for daily caps.
    MODEL_ROUTER_COOLDOWN_SECONDS: int = 60
    # See LiteLLM docs — "usage-based-routing-v2" honors per-deployment RPM/TPM
    # caps which is what we need for free tier. Alternatives:
    # "simple-shuffle" (random), "least-busy" (no rate awareness),
    # "latency-based-routing" (paid tier when SLO matters).
    MODEL_ROUTER_STRATEGY: str = "usage-based-routing-v2"

    # ── RAG / Filings ingestion (Slice 5) ──
    # Storage URL for raw filing PDFs — uses fsspec scheme dispatch.
    # Dev:  "file://./.data/filings"   (local disk, gitignored)
    # Prod: "s3://prism-filings-ap-south-1/filings"  (when paid tier is on)
    # Anything fsspec supports works (gcs://, az://, http(s)://, etc.).
    FILINGS_STORAGE_URL: str = "file://./.data/filings"

    # Embedding dimension produced by the embedding-tier model.
    # Gemini text-embedding-004 / gemini-embedding-002 default to 768; both
    # support 3072 via Matryoshka representations. Higher = better recall +
    # 4x storage. 768 is the well-balanced choice for finance-domain text.
    EMBEDDING_DIMENSION: int = 768

    # Where the declarative ingestion source registry lives. YAML for now;
    # migrates to a DB table (``ingestion_sources``) in Slice 5C without
    # changing any consumer code — ``FilingsRegistry`` is the seam.
    INGESTION_REGISTRY_PATH: str = "config/ingestion_sources.yml"

    # Hybrid retrieval — Reciprocal Rank Fusion (RRF) parameters.
    # ``k`` is the standard RRF constant (60 is the value used in the
    # original RRF paper; rarely needs tuning).
    RETRIEVAL_RRF_K: int = 60
    # Per-source candidate pool size before fusion. Larger = better recall,
    # more compute. 50 is a strong default; raise to 100+ when reranker lands.
    RETRIEVAL_TOP_K_DENSE: int = 50
    RETRIEVAL_TOP_K_SPARSE: int = 50
    # Final results returned after fusion.
    RETRIEVAL_TOP_K_FINAL: int = 10

    # ── PDF parsing (Slice 5B) ──
    # Which parser backend to use:
    #   "pdfplumber" — lightweight, in-process, pure Python. Works immediately.
    #   "docling"    — best-in-class table extraction, runs as a Docker sidecar.
    # Same ``PdfParser`` interface either way; flip when you start the sidecar.
    PARSER_BACKEND: str = "pdfplumber"
    # URL of the docling sidecar service (only used when PARSER_BACKEND=docling).
    DOCLING_SERVICE_URL: str = "http://localhost:8100"
    # Timeout for a single parse request to the sidecar. docling on CPU is
    # SLOW (observed ~400s for a fact sheet incl. first-call model download),
    # so this is generous. On GPU it drops to seconds. pdfplumber (the default)
    # ignores this entirely.
    DOCLING_TIMEOUT_SECONDS: int = 600

    # ── Chunking (Slice 5B) ──
    # Target chunk size in tokens. ~512 balances retrieval granularity against
    # embedding cost and context budget. Overlap preserves cross-boundary context.
    CHUNK_TARGET_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    # Tokenizer for chunk sizing — tiktoken encoding name. cl100k_base is a
    # good model-agnostic proxy; Gemini's true tokenizer differs slightly but
    # the estimate is close enough for budgeting.
    CHUNK_TOKENIZER: str = "cl100k_base"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def _build_url(self, driver: str, strip_sslmode: bool) -> str:
        """Build a SQLAlchemy URL for a given DBAPI driver.

        Drivers:
          * ``asyncpg`` — used by the FastAPI app at runtime. Doesn't accept
            ``sslmode=`` in the URL; we pass SSL through ``connect_args``.
          * ``psycopg`` — psycopg3, used by Alembic for migrations. Accepts
            standard libpq URL params including ``sslmode``, so we preserve
            them.
        """
        if self.DATABASE_URL:
            url = self.DATABASE_URL
            # Normalize both postgres:// and postgresql:// to the explicit driver.
            for prefix in ("postgresql://", "postgres://"):
                if url.startswith(prefix):
                    url = f"postgresql+{driver}://" + url[len(prefix):]
                    break
            return _strip_sslmode(url) if strip_sslmode else url

        return (
            f"postgresql+{driver}://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def async_database_url(self) -> str:
        """Async URL for the FastAPI runtime — ``postgresql+asyncpg://``.

        ``sslmode`` query params are stripped because asyncpg rejects them;
        SSL is controlled via ``db_connect_args`` instead.
        """
        return self._build_url("asyncpg", strip_sslmode=True)

    @property
    def sync_database_url(self) -> str:
        """Sync URL for Alembic — ``postgresql+psycopg://`` (psycopg3).

        Explicit driver prefix (not bare ``postgresql://``) so SQLAlchemy
        doesn't fall back to psycopg2, which we deliberately don't install.
        Preserves ``sslmode=`` query params — psycopg3 honors them as libpq
        does, which is exactly what managed providers like Neon expect.
        """
        return self._build_url("psycopg", strip_sslmode=False)

    @property
    def db_connect_args(self) -> dict:
        """Asyncpg connect args — controls SSL mode."""
        # asyncpg understands ssl=True/False/<ssl.SSLContext>. For managed
        # providers (Neon, RDS) we want SSL on; for local Docker, off.
        if self.DB_SSL_MODE in ("require", "verify-ca", "verify-full"):
            return {"ssl": True}
        if self.DB_SSL_MODE == "disable":
            return {"ssl": False}
        # "prefer" / "allow" — let asyncpg decide
        return {}

    @property
    def gemini_api_keys(self) -> list[str]:
        """All non-empty Gemini API keys, for round-robin."""
        keys = [
            self.GEMINI_API_KEY,
            self.GEMINI_API_KEY_1,
            self.GEMINI_API_KEY_2,
            self.GEMINI_API_KEY_3,
            self.GEMINI_API_KEY_4,
        ]
        return [k for k in keys if k]


def _strip_sslmode(url: str) -> str:
    """Asyncpg rejects libpq's ``sslmode`` query param. Strip it cleanly."""
    if "?" not in url:
        return url
    base, query = url.split("?", 1)
    params = [p for p in query.split("&") if p and not p.startswith("sslmode=")]
    return base + ("?" + "&".join(params) if params else "")


settings = Settings()
