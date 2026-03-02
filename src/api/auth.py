"""
Authentification JWT pour DataSentinel AI.

Ce module gère la création et la vérification des tokens JWT.
L'authentification est opt-in : désactivée par défaut en développement
(AUTH_ENABLED=false dans .env), activable en production.

Usage:
    # Protéger un endpoint
    from src.api.auth import get_current_user

    @router.post("/endpoint")
    async def my_endpoint(current_user = Depends(get_current_user)):
        ...
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from src.core.config import settings

# Schéma OAuth2 — pointe vers le endpoint de login
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    auto_error=False,  # False pour ne pas lever 401 si auth désactivée
)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """
    Crée un token JWT signé.

    Args:
        data: Payload à encoder (ex: {"sub": "admin"})
        expires_delta: Durée de vie. Si None, utilise jwt_expire_minutes depuis settings.

    Returns:
        Token JWT encodé (string)
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta is not None
        else timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.api_secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict[str, Any]:
    """
    Décode et vérifie un token JWT.

    Args:
        token: Token JWT à vérifier

    Returns:
        Payload décodé

    Raises:
        HTTPException 401 si token invalide ou expiré
    """
    try:
        payload = jwt.decode(
            token,
            settings.api_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict[str, Any]:
    """
    Dépendance FastAPI : retourne l'utilisateur courant.

    Comportement selon AUTH_ENABLED :
    - False (dev) : toujours autorisé, retourne {"user": "anonymous"}
    - True (prod) : vérifie le token Bearer, lève 401 si absent ou invalide

    Args:
        token: Token Bearer extrait automatiquement par FastAPI

    Returns:
        Dict avec au moins la clé "user"

    Raises:
        HTTPException 401 si auth activée et token absent/invalide
    """
    if not settings.auth_enabled:
        return {"user": "anonymous"}

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token d'authentification manquant",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(token)
    user = payload.get("sub")
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide : champ 'sub' manquant",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user": user}
