"""``prism.auth`` — the provider-agnostic authentication seam.

P0 (this slice) ships the *shape* of real auth without committing to a provider:

  * ``verifier``  — a ``TokenVerifier`` Protocol + a registry. No concrete
    provider yet; ``get_verifier()`` returns ``None`` until P1 wires one
    (Supabase / Clerk / Cognito / … — all issue OIDC JWTs, so the concrete
    impl is small and swappable).
  * ``principal`` — ``Principal`` (firm / user / role / is_anonymous) and the
    ``get_current_principal`` dependency. In dev (``AUTH_ENABLED=false``) it
    mirrors today's ``get_current_firm_id`` exactly, so behaviour is unchanged.
  * ``policy``    — a single, configurable, server-enforced access policy
    (``config/access_policy.yml``) + a ``require(feature)`` dependency.

Existing routers keep using ``src.core.auth.get_current_firm_id`` untouched in
P0; the swap to ``get_current_principal`` happens in P1 when a verifier exists.
"""

from src.auth.policy import require, required_level
from src.auth.principal import ANONYMOUS_FIRM, Principal, get_current_principal
from src.auth.verifier import (
    SupabaseVerifier,
    TokenClaims,
    TokenVerifier,
    get_verifier,
    set_verifier,
)

__all__ = [
    "Principal",
    "get_current_principal",
    "ANONYMOUS_FIRM",
    "TokenClaims",
    "TokenVerifier",
    "SupabaseVerifier",
    "get_verifier",
    "set_verifier",
    "require",
    "required_level",
]
