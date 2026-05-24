"""Core infrastructure — database engine/session, middleware, auth dependencies.

Exports:
    - ``init_engine`` / ``dispose_engine`` — app lifecycle hooks
    - ``get_session`` — FastAPI dependency for request-scoped DB sessions
    - ``session_scope`` — context manager for scripts and background tasks
"""

from src.core.database import (
    dispose_engine,
    get_session,
    get_sessionmaker,
    init_engine,
    session_scope,
)

__all__ = [
    "init_engine",
    "dispose_engine",
    "get_session",
    "get_sessionmaker",
    "session_scope",
]
