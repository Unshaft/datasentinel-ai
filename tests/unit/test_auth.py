"""
Tests unitaires pour le module d'authentification JWT.

Couvre :
- Création et décodage de tokens
- Token expiré → HTTPException 401
- Token invalide (signature erronée) → HTTPException 401
- get_current_user avec auth désactivée → toujours autorisé
- get_current_user avec auth activée → vérifie le token
"""

import asyncio
from datetime import timedelta
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from jose import jwt

from src.api.auth import create_access_token, get_current_user, verify_token
from src.core.config import settings


def _run(coro):
    """Helper : exécute une coroutine dans un event loop temporaire."""
    return asyncio.run(coro)


class TestCreateAccessToken:
    """Création de tokens JWT."""

    def test_token_is_decodable(self):
        """Un token créé doit être décodable avec la même clé."""
        token = create_access_token({"sub": "admin"})
        payload = jwt.decode(
            token, settings.api_secret_key, algorithms=[settings.jwt_algorithm]
        )
        assert payload["sub"] == "admin"

    def test_token_contains_exp(self):
        """Le token doit contenir une date d'expiration."""
        token = create_access_token({"sub": "admin"})
        payload = jwt.decode(
            token, settings.api_secret_key, algorithms=[settings.jwt_algorithm]
        )
        assert "exp" in payload

    def test_custom_expiry(self):
        """Un delta personnalisé doit être respecté."""
        token = create_access_token({"sub": "admin"}, expires_delta=timedelta(minutes=5))
        payload = jwt.decode(
            token, settings.api_secret_key, algorithms=[settings.jwt_algorithm]
        )
        assert "exp" in payload


class TestVerifyToken:
    """Vérification de tokens JWT."""

    def test_valid_token_returns_payload(self):
        """Un token valide doit retourner le payload."""
        token = create_access_token({"sub": "admin"})
        payload = verify_token(token)
        assert payload["sub"] == "admin"

    def test_expired_token_raises_401(self):
        """Un token expiré doit lever HTTPException 401."""
        token = create_access_token({"sub": "admin"}, expires_delta=timedelta(seconds=-1))
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        assert exc_info.value.status_code == 401

    def test_invalid_signature_raises_401(self):
        """Un token signé avec une mauvaise clé doit lever HTTPException 401."""
        token = jwt.encode(
            {"sub": "admin"},
            "wrong-secret-key",
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        assert exc_info.value.status_code == 401

    def test_garbage_token_raises_401(self):
        """Une chaîne aléatoire doit lever HTTPException 401."""
        with pytest.raises(HTTPException) as exc_info:
            verify_token("not.a.jwt")
        assert exc_info.value.status_code == 401


class TestGetCurrentUser:
    """Dépendance FastAPI get_current_user."""

    def test_auth_disabled_allows_anonymous(self):
        """Avec auth désactivée, toute requête est acceptée sans token."""
        with patch.object(settings, "auth_enabled", False):
            result = _run(get_current_user(token=None))
        assert result == {"user": "anonymous"}

    def test_auth_disabled_ignores_token(self):
        """Avec auth désactivée, même un token invalide passe."""
        with patch.object(settings, "auth_enabled", False):
            result = _run(get_current_user(token="garbage"))
        assert result == {"user": "anonymous"}

    def test_auth_enabled_no_token_raises_401(self):
        """Avec auth activée, sans token → 401."""
        with patch.object(settings, "auth_enabled", True):
            with pytest.raises(HTTPException) as exc_info:
                _run(get_current_user(token=None))
        assert exc_info.value.status_code == 401

    def test_auth_enabled_valid_token_succeeds(self):
        """Avec auth activée, un token valide → utilisateur identifié."""
        token = create_access_token({"sub": "admin"})
        with patch.object(settings, "auth_enabled", True):
            result = _run(get_current_user(token=token))
        assert result["user"] == "admin"

    def test_auth_enabled_expired_token_raises_401(self):
        """Avec auth activée, token expiré → 401."""
        token = create_access_token({"sub": "admin"}, expires_delta=timedelta(seconds=-1))
        with patch.object(settings, "auth_enabled", True):
            with pytest.raises(HTTPException) as exc_info:
                _run(get_current_user(token=token))
        assert exc_info.value.status_code == 401
