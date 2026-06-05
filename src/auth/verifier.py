"""Token verification seam — provider-agnostic.

Every managed auth provider we'd consider (Supabase, Clerk, Auth0, AWS Cognito,
GCP Identity Platform) issues a standard **OIDC JWT**. So the backend only needs
to (a) verify a bearer token against the provider's public JWKS keys and (b) map
its claims into a ``TokenClaims``. That keeps the provider a swappable box behind
this Protocol — choosing/changing it later is a small, contained change.

P0 ships **no concrete verifier**: ``get_verifier()`` returns ``None``. P1 wires
one (e.g. ``SupabaseVerifier``) and calls ``set_verifier(...)`` at app startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TokenClaims:
    """The subset of verified JWT claims PRISM cares about. Providers differ in
    claim names; the concrete verifier (P1) maps them into this shape."""

    sub: str                      # provider user id → maps to User.external_id
    email: str | None = None
    full_name: str | None = None
    org_id: str | None = None     # provider organization id
    org_slug: str | None = None   # → Firm.slug for org users
    org_role: str | None = None   # → FirmMembership.role


@runtime_checkable
class TokenVerifier(Protocol):
    """Verify a bearer token and return its claims, or ``None`` if invalid.

    Implementations MUST validate signature, audience, and expiry. They MUST
    NOT raise on an ordinary invalid/expired token — return ``None`` so the
    caller can treat the request as anonymous and let the access policy decide.
    """

    async def verify(self, bearer_token: str) -> TokenClaims | None: ...


class SupabaseVerifier:
    """Verify a Supabase access token, mapping its claims to ``TokenClaims``.

    Supabase signs tokens with the project's *current* JWT signing key. We
    dispatch on the token's ``alg`` header so both schemes work:

      * **ES256 / RS256** (Supabase's current asymmetric signing keys) → verify
        against the project **JWKS** endpoint. No secret lives on our server.
      * **HS256** (legacy shared secret) → verify with ``jwt_secret`` if set —
        a fallback for un-migrated projects / the key-rotation window.

    Claim mapping: ``sub`` → external_id, ``email``, ``user_metadata.full_name``
    /``name`` → full_name. Supabase has no org concept, so provisioning gives
    each user a personal (single-member) firm.
    """

    def __init__(
        self,
        *,
        jwks_url: str | None = None,
        jwt_secret: str | None = None,
        audience: str = "authenticated",
    ) -> None:
        self._jwks_url = jwks_url or None
        self._secret = jwt_secret or None
        self._audience = audience
        self._jwk_client = None  # lazily built PyJWKClient (caches fetched keys)

    def _client(self):
        import jwt

        if self._jwk_client is None and self._jwks_url:
            self._jwk_client = jwt.PyJWKClient(self._jwks_url)
        return self._jwk_client

    async def verify(self, bearer_token: str) -> TokenClaims | None:
        import asyncio

        import jwt  # local import keeps PyJWT optional until auth is enabled

        try:
            alg = jwt.get_unverified_header(bearer_token).get("alg")
        except Exception:  # noqa: BLE001 — malformed token → anonymous
            return None

        try:
            if alg in ("ES256", "RS256") and self._jwks_url:
                client = self._client()
                # PyJWKClient.get_signing_key_from_jwt does a (cached) network
                # fetch — run it off the event loop.
                signing_key = await asyncio.to_thread(client.get_signing_key_from_jwt, bearer_token)
                payload = jwt.decode(
                    bearer_token,
                    signing_key.key,
                    algorithms=["ES256", "RS256"],
                    audience=self._audience,
                )
            elif alg == "HS256" and self._secret:
                payload = jwt.decode(
                    bearer_token,
                    self._secret,
                    algorithms=["HS256"],
                    audience=self._audience,
                )
            else:
                return None
        except Exception:  # noqa: BLE001 — any verify failure → anonymous
            return None

        sub = payload.get("sub")
        if not sub:
            return None
        meta = payload.get("user_metadata") or {}
        return TokenClaims(
            sub=str(sub),
            email=payload.get("email"),
            full_name=meta.get("full_name") or meta.get("name"),
        )


# Module-level singleton — set at app startup in P1, ``None`` until then.
_verifier: TokenVerifier | None = None


def get_verifier() -> TokenVerifier | None:
    """The configured token verifier, or ``None`` if no provider is wired yet."""
    return _verifier


def set_verifier(verifier: TokenVerifier | None) -> None:
    """Install (or clear) the active token verifier. Called once at startup in P1."""
    global _verifier
    _verifier = verifier
