"""Data access layer — thin async repositories over SQLAlchemy 2.x.

Repositories are responsible for queries, ordering, and pagination — never
for HTTP concerns (those belong in routers) and never for business policy
(those belong in services / agents).
"""

from src.repositories.integration_repo import IntegrationRepository

__all__ = [
    "IntegrationRepository",
]
