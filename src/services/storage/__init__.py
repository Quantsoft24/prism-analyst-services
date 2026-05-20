"""Object storage abstraction — one API over local disk, S3, GCS, etc.

Built on ``fsspec`` (the PyData filesystem-spec used by pandas, dask, polars,
HF datasets). Switching from local-disk dev to S3 prod is a single env var:

    FILINGS_STORAGE_URL=file://./.data/filings      # dev
    FILINGS_STORAGE_URL=s3://prism-filings/filings   # prod

No application code changes — ``open_storage(url)`` dispatches on the scheme.

Why an interface on top of fsspec at all (rather than calling fsspec directly):
  * Keeps our call sites tiny + intention-revealing (``store.put(key, data)``).
  * Lets us add cross-cutting concerns (metrics, retries, content-type
    handling) in one place later.
  * Makes the dependency mockable in tests without monkeypatching fsspec.
"""

from src.services.storage.base import ObjectStore, StoredObject
from src.services.storage.fsspec_store import FsspecObjectStore, open_storage

__all__ = ["ObjectStore", "StoredObject", "FsspecObjectStore", "open_storage"]
