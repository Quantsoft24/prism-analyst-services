"""Unit tests for SupabaseVerifier (no network).

Covers the HS256 fallback path (shared secret) and the ES256 path (Supabase's
current asymmetric signing key), with the JWKS lookup stubbed so no network is
hit. Verifies that only a correctly-signed, in-audience, unexpired token maps to
claims; everything else → None (anonymous)."""

from __future__ import annotations

import types

import jwt
from cryptography.hazmat.primitives.asymmetric import ec

from src.auth.verifier import SupabaseVerifier

# ≥32 bytes so PyJWT doesn't warn about short HS keys (real secrets are long).
SECRET = "test-jwt-secret-0123456789abcdef0123456789"
_WRONG = "wrong-jwt-secret-0123456789abcdef0123456789"
_FUTURE = 4102444800  # 2100-01-01, comfortably unexpired


# ── HS256 fallback path ──────────────────────────────────────────────────────


def _hs_verifier() -> SupabaseVerifier:
    return SupabaseVerifier(jwt_secret=SECRET)


async def test_hs256_valid_token_maps_claims():
    token = jwt.encode(
        {
            "sub": "user-abc",
            "email": "analyst@example.com",
            "aud": "authenticated",
            "exp": _FUTURE,
            "user_metadata": {"full_name": "Test Analyst"},
        },
        SECRET,
        algorithm="HS256",
    )
    claims = await _hs_verifier().verify(token)
    assert claims is not None
    assert claims.sub == "user-abc"
    assert claims.email == "analyst@example.com"
    assert claims.full_name == "Test Analyst"


async def test_hs256_bad_signature_returns_none():
    token = jwt.encode({"sub": "x", "aud": "authenticated", "exp": _FUTURE}, _WRONG, algorithm="HS256")
    assert await _hs_verifier().verify(token) is None


async def test_hs256_wrong_audience_returns_none():
    token = jwt.encode({"sub": "x", "aud": "other", "exp": _FUTURE}, SECRET, algorithm="HS256")
    assert await _hs_verifier().verify(token) is None


async def test_hs256_expired_returns_none():
    token = jwt.encode({"sub": "x", "aud": "authenticated", "exp": 1}, SECRET, algorithm="HS256")
    assert await _hs_verifier().verify(token) is None


async def test_garbage_returns_none():
    assert await _hs_verifier().verify("not-a-jwt") is None


async def test_hs256_token_rejected_when_only_jwks_configured():
    """An HS256 token must NOT verify on a JWKS-only verifier (no secret)."""
    token = jwt.encode({"sub": "x", "aud": "authenticated", "exp": _FUTURE}, SECRET, algorithm="HS256")
    v = SupabaseVerifier(jwks_url="https://example.supabase.co/auth/v1/.well-known/jwks.json")
    assert await v.verify(token) is None


# ── ES256 (asymmetric / JWKS) path — JWKS lookup stubbed ─────────────────────


async def test_es256_valid_token_via_jwks(monkeypatch):
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    token = jwt.encode(
        {
            "sub": "user-ecc",
            "email": "ecc@example.com",
            "aud": "authenticated",
            "exp": _FUTURE,
            "user_metadata": {"name": "ECC User"},
        },
        private_key,
        algorithm="ES256",
        headers={"kid": "key-1"},
    )

    v = SupabaseVerifier(jwks_url="https://example.supabase.co/auth/v1/.well-known/jwks.json")
    # Stub the JWKS client so no network call happens; return the matching key.
    v._jwk_client = types.SimpleNamespace(
        get_signing_key_from_jwt=lambda _t: types.SimpleNamespace(key=public_key)
    )

    claims = await v.verify(token)
    assert claims is not None
    assert claims.sub == "user-ecc"
    assert claims.email == "ecc@example.com"
    assert claims.full_name == "ECC User"


async def test_es256_wrong_key_returns_none(monkeypatch):
    signing_key = ec.generate_private_key(ec.SECP256R1())
    other_public = ec.generate_private_key(ec.SECP256R1()).public_key()
    token = jwt.encode(
        {"sub": "x", "aud": "authenticated", "exp": _FUTURE},
        signing_key,
        algorithm="ES256",
        headers={"kid": "key-1"},
    )
    v = SupabaseVerifier(jwks_url="https://example.supabase.co/auth/v1/.well-known/jwks.json")
    v._jwk_client = types.SimpleNamespace(
        get_signing_key_from_jwt=lambda _t: types.SimpleNamespace(key=other_public)
    )
    assert await v.verify(token) is None
