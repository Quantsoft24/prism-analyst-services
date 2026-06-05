"""Auth dependency — the firm slug for the current request.

``get_current_firm_id`` now delegates to the provider-agnostic
``get_current_principal`` (``src/auth/principal.py``) and returns its
``firm_id``. This is the single resolution path:

  * ``AUTH_ENABLED=false`` → dev stub: ``X-Dev-Firm`` header or ``DEV_FIRM_ID``
    (unchanged behaviour — every existing router keeps working as before).
  * ``AUTH_ENABLED=true``  → the firm comes from the verified Supabase token
    (+ JIT-provisioned personal firm). Anonymous callers get the sentinel
    ``__anonymous__`` firm; endpoints that must be gated use
    ``src.auth.policy.require(...)`` rather than relying on this slug.

Returning a ``str`` slug keeps the contract identical for all callers, so no
router changed when real auth was wired.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.auth.principal import Principal, get_current_principal


async def get_current_firm_id(
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> str:
    """Resolve the firm slug for the current request (see module docstring)."""
    return principal.firm_id
