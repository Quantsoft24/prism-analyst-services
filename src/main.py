"""PRISM Analyst Services — FastAPI Application.

Minimal entrypoint: health check, CORS, metadata.
Routes, agents, and tools will be registered in later phases.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

app = FastAPI(
    title="PRISM Analyst Services",
    description="AI-powered equity research platform — backend API",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "ok",
        "service": "prism-analyst-services",
        "version": "0.1.0",
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with service metadata."""
    return {
        "service": "PRISM Analyst Services",
        "version": "0.1.0",
        "docs": "/docs" if settings.DEBUG else "Disabled in production",
        "health": "/health",
    }
