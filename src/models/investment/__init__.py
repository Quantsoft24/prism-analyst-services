"""ORM models for the read-only investment DB (AWS RDS).

These models live on ``InvestmentBase`` (see ``src/core/investment_database.py``)
so PRISM's primary Alembic chain never touches them. Treat as read-only — the
tables are owned externally.
"""

from src.models.investment.annual_data import AnnualData
from src.models.investment.index_tables import IndexConstituent, IndexData, IndicesList
from src.models.investment.master_security import MasterSecurity
from src.models.investment.price_row import PriceRow

__all__ = [
    "AnnualData",
    "IndexConstituent",
    "IndexData",
    "IndicesList",
    "MasterSecurity",
    "PriceRow",
]
