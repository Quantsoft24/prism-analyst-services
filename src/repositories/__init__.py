"""Data access layer — thin async repositories over SQLAlchemy 2.x.

Repositories are responsible for queries, ordering, and pagination — never
for HTTP concerns (those belong in routers) and never for business policy
(those belong in services / agents).
"""

from src.repositories.bmc_repo import BMCRepository
from src.repositories.company_repo import CompanyRepository
from src.repositories.filing_repo import (
    ChunkHit,
    FilingChunkRepository,
    FilingRepository,
)

__all__ = [
    "CompanyRepository",
    "FilingRepository",
    "FilingChunkRepository",
    "ChunkHit",
    "BMCRepository",
]
