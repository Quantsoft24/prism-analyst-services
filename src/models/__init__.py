"""SQLAlchemy ORM models for PRISM.

Import order matters for Alembic autogeneration: ``Base`` first, then every
model module so that all tables are registered against ``Base.metadata``.
"""

from src.models.agent_run import AgentRun
from src.models.base import Base
from src.models.billing import Entitlement, Plan, Subscription

# Catalog DB models (read-only, separate Base — kept out of Alembic's metadata).
from src.models.catalog import CompanyIndustry  # noqa: F401
from src.models.chat_conversation import ChatConversation
from src.models.firm import Firm
from src.models.integration import FirmIntegration
from src.models.portfolio import (
    PortfolioBacktest,
    PortfolioCustomFactor,
    PortfolioStrategy,
)
from src.models.user import FirmMembership, User
from src.models.user_preferences import UserPreference

__all__ = [
    "Base",
    "Firm",
    "User",
    "FirmMembership",
    "AgentRun",
    "FirmIntegration",
    "PortfolioBacktest",
    "PortfolioCustomFactor",
    "PortfolioStrategy",
    "CompanyIndustry",
    "UserPreference",
    "Plan",
    "Subscription",
    "Entitlement",
    "ChatConversation",
]
