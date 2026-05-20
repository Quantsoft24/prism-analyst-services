"""FastAPI routers — one module per resource.

Each router is mounted under ``settings.API_PREFIX`` (``/api/v1``) in
``src/main.py``. All routes are documented via FastAPI's auto-generated
OpenAPI spec at ``/docs`` (dev) — that spec is the contract third-party
consumers depend on.
"""

from src.routers.chat import router as chat_router
from src.routers.companies import router as companies_router
from src.routers.filings import router as filings_router
from src.routers.router_health import router as router_health_router

__all__ = ["companies_router", "chat_router", "filings_router", "router_health_router"]
