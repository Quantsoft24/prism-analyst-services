"""SQLAlchemy ORM models for PRISM.

Import order matters for Alembic autogeneration: ``Base`` first, then every
model module so that all tables are registered against ``Base.metadata``.
"""

from src.models.agent_run import AgentRun
from src.models.base import Base

# Catalog DB models (read-only, separate Base — kept out of Alembic's metadata).
from src.models.catalog import CompanyIndustry  # noqa: F401
from src.models.firm import Firm
from src.models.integration import FirmIntegration
from src.models.user import FirmMembership, User

__all__ = [
    "Base",
    "Firm",
    "User",
    "FirmMembership",
    "AgentRun",
    "FirmIntegration",
    "CompanyIndustry",
]
