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
    # Optional comma-separated FALLBACK DB URLs (e.g. spare Neon projects). When
    # the active DB rejects connections — e.g. a Neon free-tier project whose
    # monthly compute allowance is spent and whose endpoint is disabled — the app
    # automatically fails over to the next URL here. NOTE: these are independent
    # databases (data is NOT replicated); a failover lands on a separate dataset.
    # Acceptable for an internal/dev tool where losing data is fine.
    DATABASE_URL_FALLBACKS: str = ""
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
    GEMINI_API_KEY_5: str = ""
    GEMINI_API_KEY_6: str = ""
    GEMINI_API_KEY_7: str = ""
    GEMINI_API_KEY_8: str = ""

    # ── LLM: OpenAI (optional, provider-swap ready) ──
    # Blank by default — Gemini-only runs are unaffected. Set this (gitignored
    # .env only) to route any ``openai/<model>`` entries added to a tier in
    # ``model_router_config.py`` through the ModelRouter. The router pairs
    # ``openai/*`` deployments with this key (Gemini models keep using the
    # Gemini keys). Lets us use a stronger composer (e.g. GPT-4.x) for the
    # ``quality`` tier without touching router code.
    OPENAI_API_KEY: str = ""

    # ── LLM: DeepSeek (fallback after OpenAI, before Gemini) ──
    # Blank by default — inert until set. When present, the router pairs any
    # ``deepseek/<model>`` entries in a tier (``model_router_config.py``) with
    # this key (via the generic ``<PROVIDER>_API_KEY`` convention in
    # ``ModelRouter._keys_for_provider``). DeepSeek's API is OpenAI-compatible
    # and cheap; used as the 1st fallback when the OpenAI group errors/cools.
    DEEPSEEK_API_KEY: str = ""

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
    # Daily message caps per tier (see src/services/rate_limit.py).
    RATE_LIMITS_PATH: str = "config/rate_limits.yml"

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
    # A multi-step turn (resolve → gather → quality-tier compose) plus the
    # router's multi-key 429 back-off can legitimately need >60s on the free
    # tier; 60s was clipping otherwise-good turns into a timeout. 90s gives the
    # fallback chain room without making a stuck turn hang absurdly long.
    AGENT_TIMEOUT_SECONDS: int = 90
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

    # ── Chat UX: deep-dive "explore further" suggestions ──
    # Max number of deep-dive suggestion chips surfaced under an answer. Capped
    # so we never overwhelm the user; tune here (or via env) without touching
    # the rule logic. See src/services/deep_dive.py.
    DEEP_DIVE_MAX_SUGGESTIONS: int = 3

    # ── Integrations (agent tools / MCP / sub-agents) ──
    # Declarative registry of agent-callable resources. Same YAML→DB migration
    # seam as the ingestion registry. See docs/INTEGRATION_INTAKE.md.
    INTEGRATIONS_REGISTRY_PATH: str = "config/integrations.yml"
    # Base URL for the teammate-built stock-chat filings service (no caller auth;
    # network-restricted). Referenced by the stock-chat integration tool.
    STOCK_CHAT_URL: str = "http://localhost:8011"

    # ── Prism Financials (external numeric-Q&A service) ──
    # Teammate-built finance Q&A service over the investment DB (FastAPI). As of
    # the security_id migration (2026-06) it runs on prod `http://35.234.221.166:8090`
    # and takes {question, security_id?/security_ids?}; POST /ask returns a typed,
    # operation-specific result (lookup/trend/compare/rank/screen/statement) plus
    # an NL answer + SQL provenance. Referenced by the prism-financials tool. The
    # endpoint is open (no caller auth); PRISM_FINANCIALS_API_KEY stays empty until
    # the service adds X-API-Key — the wrapper sends the header only when set.
    #
    # Default is the prod URL so it works out of the box; override in .env for a
    # local `uvicorn` run. ⚠️ If .env still has the OLD :8000 port, update it to
    # :8090.
    PRISM_FINANCIALS_URL: str = "http://35.234.221.166:8090"
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

    # ── Prism Filings (external corporate-filings / announcements service) ──
    # Teammate-built FastAPI service (aaaddditya/Prism_filing_news, prod
    # `http://35.234.221.166:8002`) aggregating 32 official Indian regulator/
    # exchange RSS feeds (RBI/SEBI/BSE/NSE/PIB) into a single /filings query.
    # Each filing is auto-tagged with company/sector/industry/scrip_code, so we
    # can scope the Stock Dashboard's Announcements pane to the selected company.
    # The frontend talks through PRISM's own /api/v1/stocks/announcements proxy
    # (one CORS/mixed-content surface), mirroring the /reports → stock-chat path.
    # Endpoint is open today (no caller auth) — must stay network-restricted.
    #
    # Default :8002 mirrors the upstream's external port for a same-host dev run;
    # set PRISM_FILINGS_URL to the real GCP host in .env for prod.
    PRISM_FILINGS_URL: str = "http://localhost:8002"

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

    # BMC is a FIRM-WIDE shared library: every Business Model Canvas is visible to
    # everyone in the app (guest OR signed-in), NOT scoped to the requesting user.
    # So all BMC calls (the /api/v1/bmc/* proxy AND the agent's bmc_* tools) pin a
    # single constant firm_id instead of the request principal's per-user/anonymous
    # firm. This decouples BMC from auth identity (which now churns across the Neon
    # DB fallbacks). Point it at wherever the canvases live if not "default".
    BMC_SHARED_FIRM_ID: str = "default"

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

    # ── SEBI DB (READ-ONLY secondary engine — the ``sebi`` Postgres backing the
    # Regulatory Lens feature). Single content table ``content`` (~40k rows of
    # SEBI circulars/regulations/orders with an AI-enriched ``ai_tags`` JSON
    # column) + ``weekly_summaries`` (digest) + ``insight_feed`` (AI signals).
    # Owned externally and exposed via a read-only ``frontend`` role — NEVER
    # write through this engine. Provide a full ``SEBI_DATABASE_URL`` OR the
    # ``SEBI_DB_*`` parts. If unset, the Regulatory Lens endpoints 503 and the
    # rest of the app is unaffected. Plain VM Postgres (not RDS) → no TLS.
    SEBI_DATABASE_URL: str = ""
    SEBI_DB_HOST: str = ""
    SEBI_DB_PORT: int = 15432
    SEBI_DB_NAME: str = "sebi"
    SEBI_DB_USER: str = "frontend"
    SEBI_DB_PASSWORD: str = ""
    SEBI_DB_SSL_MODE: str = "disable"  # plain VM Postgres, no TLS

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

    @staticmethod
    def _to_asyncpg(raw: str) -> str:
        """Normalize an arbitrary raw Postgres URL to ``postgresql+asyncpg://``
        with the ``sslmode`` query param stripped (asyncpg rejects it; SSL goes
        through ``db_connect_args``). Used for the fallback URLs, which are
        always full connection strings."""
        url = raw.strip()
        if url.startswith("postgresql+"):
            _, _, rest = url.partition("://")
            url = f"postgresql+asyncpg://{rest}"
        elif url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        elif url.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://"):]
        return _strip_sslmode(url)

    @property
    def async_database_urls(self) -> list[str]:
        """The primary async URL followed by any configured fallbacks, in order,
        de-duplicated. The failover engine (``src/core/database.py``) walks this
        list when the active DB stops accepting connections."""
        urls = [self.async_database_url]
        for raw in self.DATABASE_URL_FALLBACKS.split(","):
            if raw.strip():
                urls.append(self._to_asyncpg(raw))
        seen: set[str] = set()
        out: list[str] = []
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        return out

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
    def async_sebi_database_url(self) -> str:
        """Async URL for the read-only SEBI DB (``sebi`` Postgres).

        Prefers ``SEBI_DATABASE_URL``; otherwise builds from the ``SEBI_DB_*``
        parts via SQLAlchemy's ``URL.create`` (percent-encodes the password).
        Returns "" if neither is configured (Regulatory Lens endpoints then
        degrade gracefully). Normalizes the driver to asyncpg and strips
        ``sslmode`` (SSL goes through ``sebi_connect_args``)."""
        if self.SEBI_DATABASE_URL:
            url = self.SEBI_DATABASE_URL
            for prefix in ("postgresql://", "postgres://"):
                if url.startswith(prefix):
                    url = "postgresql+asyncpg://" + url[len(prefix):]
                    break
            return _strip_sslmode(url)
        if not self.SEBI_DB_HOST:
            return ""
        from sqlalchemy.engine import URL

        return URL.create(
            "postgresql+asyncpg",
            username=self.SEBI_DB_USER,
            password=self.SEBI_DB_PASSWORD,
            host=self.SEBI_DB_HOST,
            port=self.SEBI_DB_PORT,
            database=self.SEBI_DB_NAME,
        ).render_as_string(hide_password=False)

    @property
    def sebi_connect_args(self) -> dict:
        """Asyncpg connect args for the SEBI DB — TLS handling. Mirrors
        ``investment_connect_args``. The SEBI VM Postgres serves plaintext, so
        the default ``disable`` yields ``{"ssl": False}``."""
        mode = self.SEBI_DB_SSL_MODE
        if mode == "disable":
            return {"ssl": False}
        if mode in ("verify-ca", "verify-full"):
            return {"ssl": True}
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
        """All non-empty Gemini API keys, for round-robin + 429 fallback.
        The ModelRouter replicates every model across ALL of these, so adding
        keys directly widens free-tier headroom (the composer leans on this)."""
        keys = [
            self.GEMINI_API_KEY,
            self.GEMINI_API_KEY_1,
            self.GEMINI_API_KEY_2,
            self.GEMINI_API_KEY_3,
            self.GEMINI_API_KEY_4,
            self.GEMINI_API_KEY_5,
            self.GEMINI_API_KEY_6,
            self.GEMINI_API_KEY_7,
            self.GEMINI_API_KEY_8,
        ]
        # De-dupe (a key pasted twice would double-count as one deployment) while
        # preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for k in keys:
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out


def _strip_sslmode(url: str) -> str:
    """Asyncpg rejects libpq's ``sslmode`` query param. Strip it cleanly."""
    if "?" not in url:
        return url
    base, query = url.split("?", 1)
    params = [p for p in query.split("&") if p and not p.startswith("sslmode=")]
    return base + ("?" + "&".join(params) if params else "")


settings = Settings()
