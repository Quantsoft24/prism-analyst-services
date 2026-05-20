"""Shared response shapes — pagination, errors."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

ItemT = TypeVar("ItemT")


class PageMeta(BaseModel):
    """Metadata block included on every paginated response."""

    total: int = Field(description="Total number of items matching the query (before pagination).")
    limit: int = Field(description="Maximum items returned in this page.")
    offset: int = Field(description="Zero-based offset of the first item in this page.")

    @property
    def has_more(self) -> bool:
        return self.offset + self.limit < self.total


class Paginated(BaseModel, Generic[ItemT]):
    """Generic envelope for list endpoints.

    Wire shape::

        {
          "items": [ ... ],
          "page": {"total": 123, "limit": 25, "offset": 0}
        }
    """

    items: list[ItemT]
    page: PageMeta
