"""PRISM Analyst Services — FastAPI Application.

Boots the FastAPI app with:
  - CORS configured per environment
  - Async PostgreSQL engine initialized on startup, disposed on shutdown
  - All ``/api/v1/...`` routers mounted under one versioned prefix
  - OpenAPI docs at ``/docs`` (dev only) — third-party API consumers
    depend on this spec, so router metadata must stay accurate.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.core.database import dispose_engine, init_engine
from src.core.investment_database import (
    dispose_investment_engine,
    init_investment_engine,
    is_investment_configured,
)
from src.core.sebi_database import (
    dispose_sebi_engine,
    init_sebi_engine,
    is_sebi_configured,
)
from src.integrations import dispose_registry, init_registry
from src.routers import (
    bmc_router,
    chat_router,
    integrations_router,
    me_router,
    news_router,
    portfolio_router,
    regulatory_router,
    router_health_router,
    stocks_router,
)
from src.services.model_router import dispose_router, init_router

logger = logging.getLogger(__name__)


def _configure_adk_env() -> None:
    """Bridge our config into the env vars Google ADK / google-genai expect.

    ADK reads ``GOOGLE_API_KEY`` for AI Studio mode and toggles via
    ``GOOGLE_GENAI_USE_VERTEXAI`` for Vertex mode. Users configure only
    ``GEMINI_API_KEY`` / ``ADK_PROVIDER`` in ``.env`` — we translate here.

    Also disables ADK's internal OpenTelemetry instrumentation. ADK uses
    OTel for span tracing of its agent loop; our SSE generator pattern
    closes the inner runner generator early (on ``is_final_response()``),
    which the OTel cleanup can't handle cleanly across asyncio contexts
    and produces noisy "Failed to detach context" log spam without
    affecting functionality. We'll plug in our own instrumentation
    (Sentry / OTel-direct) in Phase 1 W1.
    """
    if settings.ADK_PROVIDER == "vertex":
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        if settings.ADK_VERTEX_PROJECT:
            os.environ["GOOGLE_CLOUD_PROJECT"] = settings.ADK_VERTEX_PROJECT
        if settings.ADK_VERTEX_LOCATION:
            os.environ["GOOGLE_CLOUD_LOCATION"] = settings.ADK_VERTEX_LOCATION
    else:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        if settings.GEMINI_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY

    # Silence ADK's internal OTel — see docstring above.
    # Honors any pre-existing value so operators can opt back in.
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def _configure_auth() -> None:
    """Install the token verifier when real auth is enabled.

    Provider = Supabase (final_docs/12). Supabase's current signing key is
    asymmetric (ECC P-256), so we verify via the project JWKS — derived from
    ``SUPABASE_URL`` unless ``SUPABASE_JWKS_URL`` overrides it. ``SUPABASE_JWT_
    SECRET`` is an optional HS256 fallback. If neither is configured while
    ``AUTH_ENABLED`` is set, the verifier stays unset and
    ``get_current_principal`` fails closed (501).
    """
    if not settings.AUTH_ENABLED:
        return
    from src.auth import SupabaseVerifier, set_verifier

    jwks_url = settings.SUPABASE_JWKS_URL or (
        f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
        if settings.SUPABASE_URL
        else ""
    )
    if jwks_url or settings.SUPABASE_JWT_SECRET:
        set_verifier(
            SupabaseVerifier(
                jwks_url=jwks_url or None,
                jwt_secret=settings.SUPABASE_JWT_SECRET or None,
                audience=settings.SUPABASE_JWT_AUD,
            )
        )
        logger.info(
            "Auth enabled — Supabase verifier installed (jwks=%s, hs256_fallback=%s).",
            bool(jwks_url),
            bool(settings.SUPABASE_JWT_SECRET),
        )
    else:
        logger.error(
            "AUTH_ENABLED=true but neither SUPABASE_URL nor SUPABASE_JWT_SECRET "
            "is set — auth will fail closed."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB pools + ModelRouter on startup; dispose cleanly on shutdown."""
    _configure_adk_env()
    _configure_auth()
    init_engine()

    # Read-only engine for the investment DB (AWS RDS) — powers the Stock
    # Dashboard (/api/v1/stocks/*). Skipped silently if not configured; a
    # failure here logs + continues (the stocks routes 503 if accessed).
    if is_investment_configured():
        try:
            init_investment_engine()
            logger.info("Investment DB engine initialized (read-only).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Investment DB engine failed to initialize: %s", exc)

    # Read-only engine for the SEBI DB — powers Regulatory Lens
    # (/api/v1/regulatory/*). Skipped silently if not configured; a failure
    # here logs + continues (the regulatory routes 503 if accessed).
    if is_sebi_configured():
        try:
            init_sebi_engine()
            logger.info("SEBI DB engine initialized (read-only).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("SEBI DB engine failed to initialize: %s", exc)

    # Build the ModelRouter if enabled — collects all GEMINI_API_KEY* values
    # from settings and hands them to the singleton. Disabling lets us run
    # without LLM access (e.g., CI runs of non-LLM endpoints).
    if settings.MODEL_ROUTER_ENABLED:
        api_keys = settings.gemini_api_keys
        if api_keys:
            init_router(api_keys=api_keys)
            logger.info("ModelRouter initialized with %d Gemini API key(s).", len(api_keys))
        else:
            logger.warning(
                "MODEL_ROUTER_ENABLED=true but no GEMINI_API_KEY found. "
                "Agent endpoints will fail until a key is configured."
            )

    # Build the integration registry AFTER the router (agent-source integrations
    # need the router to build their sub-agents). Failures are isolated per
    # integration and surfaced via GET /integrations — never block startup.
    try:
        await init_registry(settings.INTEGRATIONS_REGISTRY_PATH)
    except Exception as exc:  # noqa: BLE001 — registry must never block boot
        logger.warning("Integration registry failed to initialize: %s", exc)

    try:
        yield
    finally:
        dispose_registry()
        dispose_router()
        await dispose_sebi_engine()
        await dispose_investment_engine()
        await dispose_engine()


app = FastAPI(
    title="PRISM Analyst Services",
    description="AI-powered equity research platform — backend API.",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── System endpoints (no prefix — load balancer health checks) ──
@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "ok",
        "service": "prism-analyst-services",
        "version": "0.1.0",
    }


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Service metadata."""
    return {
        "service": "PRISM Analyst Services",
        "version": "0.1.0",
        "docs": "/docs" if settings.DEBUG else "Disabled in production",
        "health": "/health",
    }


# ── Versioned API routers ──
app.include_router(chat_router, prefix=settings.API_PREFIX)
app.include_router(bmc_router, prefix=settings.API_PREFIX)
app.include_router(news_router, prefix=settings.API_PREFIX)
app.include_router(stocks_router, prefix=settings.API_PREFIX)
app.include_router(regulatory_router, prefix=settings.API_PREFIX)
app.include_router(portfolio_router, prefix=settings.API_PREFIX)
app.include_router(integrations_router, prefix=settings.API_PREFIX)
app.include_router(me_router, prefix=settings.API_PREFIX)
# Debug router — actual access is gated inside the handler (404 in prod).
# We mount unconditionally so the route table is consistent.
app.include_router(router_health_router, prefix=settings.API_PREFIX)
