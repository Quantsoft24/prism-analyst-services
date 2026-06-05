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

    # ── Auth ──
    # AUTH_ENABLED=false → dev-mode firm header stub (zero behaviour change).
    # AUTH_ENABLED=true  → verify Supabase JWTs + JIT-provision users/firms.
    #
    # Provider = Supabase (decided 2026-06-05; see final_docs/12). The backend
    # only verifies a standard OIDC JWT, so switching providers later is a small
    # change (a new TokenVerifier). For the ≤100-user pilot we verify Supabase's
    # HS256 access token with the project JWT secret (server-side only); set
    # SUPABASE_JWT_SECRET from the Supabase dashboard (Settings → API → JWT
    # Secret). SUPABASE_URL is informational / for a future JWKS swap.
    AUTH_ENABLED: bool = False
    DEV_FIRM_ID: str = "QUANTSOFT"
    SUPABASE_URL: str = ""
    # Supabase's CURRENT signing key is asymmetric (ECC P-256) → verify via the
    # project JWKS (derived from SUPABASE_URL if SUPABASE_JWKS_URL is blank). No
    # secret needed on our server. SUPABASE_JWT_SECRET is an optional HS256
    # fallback for legacy/un-migrated projects.
    SUPABASE_JWKS_URL: str = ""
    SUPABASE_JWT_SECRET: str = ""
    SUPABASE_JWT_AUD: str = "authenticated"  # Supabase default audience claim
    # Where the configurable gating matrix lives (see src/auth/policy.py).
    ACCESS_POLICY_PATH: str = "config/access_policy.yml"

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

    # ── Integrations (agent tools / MCP / sub-agents) ──
    # Declarative registry of agent-callable resources. Same YAML→DB migration
    # seam as the ingestion registry. See docs/INTEGRATION_INTAKE.md.
    INTEGRATIONS_REGISTRY_PATH: str = "config/integrations.yml"
    # Base URL for the teammate-built stock-chat filings service (no caller auth;
    # network-restricted). Referenced by the stock-chat integration tool.
    STOCK_CHAT_URL: str = "http://localhost:8011"

    # ── Prism Financials (external numeric-Q&A service) ──
    # Teammate-built text-to-SQL service over CMIE Prowess (FastAPI, prod
    # `http://35.234.221.166:8000`). POST /ask turns a finance question into
    # safe read-only Postgres SQL and returns structured rows + the SQL.
    # Referenced by the prism-financials integration tool. The endpoint is
    # currently open (no caller auth); PRISM_FINANCIALS_API_KEY stays empty
    # until the service adds X-API-Key auth — the wrapper sends the header only
    # when it's set, so no secret ever lands in git.
    #
    # NOTE: the upstream runs on :8000 — the same port PRISM itself binds to.
    # The default below is :8013 (a placeholder), NOT :8000, so a missing env
    # var fails loudly with "connection refused" instead of silently routing
    # /ask back into PRISM's own FastAPI server (which would 404).
    PRISM_FINANCIALS_URL: str = "http://localhost:8013"
    PRISM_FINANCIALS_API_KEY: str = ""

    # ── Prism News (external financial-news + sentiment service) ──
    # Teammate-built FastAPI service (showtimeapp/NewsRSS) over 82 Indian RSS
    # feeds with OpenAI sentiment + a 4,149-company alias master. Prod runs on
    # :8001. Exposes REST (/news, /news/summary, /news/trending, /news/compare,
    # /news/sources, /news/companies, /news/sectors, /health, /stats) AND an MCP
    # endpoint (/mcp). We wire it as a ``python`` typed agent wrapper (NOT
    # ``openapi``) — same convention as stock-chat / bmc / prism-financials —
    # so the LLM sees only the 3-4 agent-useful tools, gets trimmed responses,
    # and inherits our structured error contract. The frontend /news page talks
    # through PRISM's own /api/v1/news/* proxy router (one CORS/auth surface),
    # mirroring the BMC pattern. Endpoint is open today (no caller auth);
    # PRISM_NEWS_API_KEY stays empty until a gateway is added — the wrapper
    # sends the header only when set, so no secret lands in git.
    #
    # Default :8014 is a deliberate placeholder (NOT :8001) so a missing env
    # var fails loudly with "connection refused" rather than silently hitting
    # the wrong local service. Set PRISM_NEWS_URL to the real GCP host in .env.
    PRISM_NEWS_URL: str = "http://localhost:8014"
    PRISM_NEWS_API_KEY: str = ""

    # RAG / pdf-parsing / chunking settings retired with the read-on-demand
    # cutover (2026-05-24) — PRISM no longer maintains its own embedding/chunk
    # index. Filings narrative Q&A comes via stock-chat's read-on-demand tools.
    CHUNK_TOKENIZER: str = "cl100k_base"

    # ── BMC (external service) ──
    # PRISM's own RAG-based BMC is retired; the teammate-built `bmc` service
    # (FastAPI on port 8012, owns its own 5 tables in the shared Postgres) is
    # the source of truth. PRISM's /api/v1/bmc/* router is a thin proxy here;
    # the chat agent also reaches it via the integration registry (typed
    # wrappers in src/integrations/tools/bmc.py). No caller auth — must be
    # network-restricted to the PRISM backend's IP. Dev: localhost; prod: VM IP.
    BMC_URL: str = "http://localhost:8012"

    # ── Catalog DB (READ-ONLY secondary engine pointing at the stock_chat
    # Postgres). PRISM's company lookup tools + /api/v1/companies router read
    # from `company_industry` here (4,773 rows) instead of maintaining a
    # duplicate `companies` table in PRISM's primary DB. Same for any future
    # read against `filings_index` / `document_texts`. NEVER write through
    # this engine — those tables are owned by the stock-chat / bmc services.
    # If left blank, falls back to ``POSTGRES_URL`` (the teammate's env-var name).
    CATALOG_DATABASE_URL: str = ""
    POSTGRES_URL: str = ""  # back-compat: read teammate's .env if set

    # ── Investment DB (READ-ONLY secondary engine — new AWS RDS ``investment``
    # Postgres backing the Stock Dashboard). Two tables only: ``master_securities``
    # (8,230 NSE/BSE securities) + ``prices_and_securities`` (21.5M daily OHLC /
    # volume / value / market-cap rows). Owned externally — NEVER write through
    # this engine. Provide either a full ``INVESTMENT_DATABASE_URL`` OR the
    # ``INVESTMENT_DB_*`` parts (preferred when the password has URL-unsafe chars
    # like ``$`` — we URL-encode via SQLAlchemy's URL.create). If unset, the
    # Stock Dashboard endpoints 503 and the rest of the app is unaffected.
    INVESTMENT_DATABASE_URL: str = ""
    INVESTMENT_DB_HOST: str = ""
    INVESTMENT_DB_PORT: int = 5432
    INVESTMENT_DB_NAME: str = "investment"
    INVESTMENT_DB_USER: str = "postgres"
    INVESTMENT_DB_PASSWORD: str = ""
    INVESTMENT_DB_SSL_MODE: str = "require"  # RDS requires TLS

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def _build_url(self, driver: str, strip_sslmode: bool) -> str:
        """Build a SQLAlchemy URL for the requested DBAPI driver.

        Normalises ANY scheme prefix to ``postgresql+{driver}://``, including:
          * bare ``postgresql://`` / ``postgres://``       (env-var convention)
          * already-driver-tagged ``postgresql+asyncpg://`` / ``+psycopg://``
            (so Alembic gets the SYNC driver even when CI passes an async URL)

        Drivers we care about:
          * ``asyncpg`` — FastAPI runtime. Doesn't accept ``sslmode=`` in the
            URL; SSL goes through ``connect_args``.
          * ``psycopg`` — psycopg3, used by Alembic for migrations. Accepts
            libpq URL params (``sslmode=...``) directly.
        """
        if self.DATABASE_URL:
            url = self.DATABASE_URL
            # Any existing scheme → the requested driver.
            if url.startswith("postgresql+"):
                # postgresql+asyncpg://... or postgresql+psycopg://... — strip
                # the driver, re-attach the requested one.
                _, _, rest = url.partition("://")
                url = f"postgresql+{driver}://{rest}"
            elif url.startswith("postgresql://"):
                url = f"postgresql+{driver}://" + url[len("postgresql://"):]
            elif url.startswith("postgres://"):
                url = f"postgresql+{driver}://" + url[len("postgres://"):]
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
    def async_catalog_database_url(self) -> str:
        """Async URL for the read-only catalog DB (stock_chat Postgres).
        Prefers ``CATALOG_DATABASE_URL``; falls back to ``POSTGRES_URL`` (the
        teammate's existing env-var name). Returns "" if neither is set —
        callers should handle gracefully (catalog features just don't load).
        Strips ``sslmode`` and normalizes the driver to asyncpg (same logic
        as the primary URL builder)."""
        raw = self.CATALOG_DATABASE_URL or self.POSTGRES_URL
        if not raw:
            return ""
        url = raw
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                url = "postgresql+asyncpg://" + url[len(prefix):]
                break
        return _strip_sslmode(url)

    @property
    def async_investment_database_url(self) -> str:
        """Async URL for the read-only investment DB (RDS ``investment``).

        Prefers ``INVESTMENT_DATABASE_URL``; otherwise builds from the
        ``INVESTMENT_DB_*`` parts via SQLAlchemy's ``URL.create`` (which
        percent-encodes the password — important because the RDS password
        contains ``$``). Returns "" if neither is configured (Stock Dashboard
        endpoints then degrade gracefully). Normalizes the driver to asyncpg
        and strips ``sslmode`` (SSL goes through ``investment_connect_args``)."""
        if self.INVESTMENT_DATABASE_URL:
            url = self.INVESTMENT_DATABASE_URL
            for prefix in ("postgresql://", "postgres://"):
                if url.startswith(prefix):
                    url = "postgresql+asyncpg://" + url[len(prefix):]
                    break
            return _strip_sslmode(url)
        if not self.INVESTMENT_DB_HOST:
            return ""
        from sqlalchemy.engine import URL

        return URL.create(
            "postgresql+asyncpg",
            username=self.INVESTMENT_DB_USER,
            password=self.INVESTMENT_DB_PASSWORD,
            host=self.INVESTMENT_DB_HOST,
            port=self.INVESTMENT_DB_PORT,
            database=self.INVESTMENT_DB_NAME,
        ).render_as_string(hide_password=False)

    @property
    def investment_connect_args(self) -> dict:
        """Asyncpg connect args for the investment DB — TLS handling.

        ``require``/``prefer``/``allow`` → encrypt but DON'T verify the cert
        chain (libpq ``sslmode=require`` semantics) via an unverified SSL
        context, so we don't need to ship the Amazon RDS CA bundle to connect.
        ``verify-ca``/``verify-full`` → strict verification (``ssl=True``);
        point ``INVESTMENT_DB_SSL_MODE`` there once the RDS global-bundle.pem is
        installed in the trust store. ``disable`` → plaintext."""
        mode = self.INVESTMENT_DB_SSL_MODE
        if mode == "disable":
            return {"ssl": False}
        if mode in ("verify-ca", "verify-full"):
            return {"ssl": True}
        # require / prefer / allow — encrypt without chain verification
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return {"ssl": ctx}

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
