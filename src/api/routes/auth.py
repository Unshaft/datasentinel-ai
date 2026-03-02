"""
Route /auth - Authentification et gestion des tokens.

Endpoint OAuth2 compatible pour l'obtention de tokens JWT.
Utilise les credentials configurés dans Settings (API_USERNAME / API_PASSWORD).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from src.api.auth import create_access_token
from src.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


class TokenResponse(BaseModel):
    """Réponse contenant le token JWT."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Obtenir un token JWT",
    description="""
    Authentifie un utilisateur et retourne un token JWT Bearer.

    Le token doit ensuite être passé dans le header `Authorization: Bearer <token>`
    sur les endpoints protégés (quand `AUTH_ENABLED=true`).

    **En développement** (`AUTH_ENABLED=false`) : cette route reste disponible
    mais les endpoints ne vérifient pas le token.
    """,
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """
    Génère un token JWT après vérification des credentials.

    Args:
        form_data: username + password (form-data OAuth2)

    Returns:
        Token JWT + durée de vie

    Raises:
        HTTPException 401 si credentials incorrects
    """
    if (
        form_data.username != settings.api_username
        or form_data.password != settings.api_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": form_data.username})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_expire_minutes * 60,
    )
