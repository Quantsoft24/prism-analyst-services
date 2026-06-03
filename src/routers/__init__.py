"""FastAPI routers — one module per resource.

Each router is mounted under ``settings.API_PREFIX`` (``/api/v1``) in
``src/main.py``. All routes are documented via FastAPI's auto-generated
OpenAPI spec at ``/docs`` (dev) — that spec is the contract third-party
consumers depend on.
"""

from src.routers.bmc import router as bmc_router
from src.routers.chat import router as chat_router
from src.routers.companies import router as companies_router
from src.routers.integrations import router as integrations_router
from src.routers.news import router as news_router
from src.routers.portfolio import router as portfolio_router
from src.routers.router_health import router as router_health_router
from src.routers.stocks import router as stocks_router

__all__ = [
    "companies_router",
    "chat_router",
    "bmc_router",
    "news_router",
    "portfolio_router",
    "integrations_router",
    "router_health_router",
    "stocks_router",
]
