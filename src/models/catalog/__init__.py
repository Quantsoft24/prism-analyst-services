"""ORM models for the read-only catalog DB (stock_chat Postgres).

These models live on ``CatalogBase`` (see ``src/core/catalog_database.py``) so
PRISM's primary Alembic chain never touches them. Treat as read-only — the
stock-chat / bmc services own these tables.
"""

from src.models.catalog.company_alias import CompanyAlias
from src.models.catalog.company_industry import CompanyIndustry

__all__ = ["CompanyAlias", "CompanyIndustry"]
