"""SQLAlchemy ORM models for PRISM.

Import order matters for Alembic autogeneration: ``Base`` first, then every
model module so that all tables are registered against ``Base.metadata``.
"""

from src.models.agent_run import AgentRun
from src.models.base import Base
from src.models.bmc import BMCAnalysis, BMCBlock, BMCEvidence
from src.models.company import Company, CompanyAlias
from src.models.filing import Filing, FilingChunk
from src.models.firm import Firm
from src.models.user import FirmMembership, User

__all__ = [
    "Base",
    "Firm",
    "User",
    "FirmMembership",
    "Company",
    "CompanyAlias",
    "AgentRun",
    "Filing",
    "FilingChunk",
    "BMCAnalysis",
    "BMCBlock",
    "BMCEvidence",
]
