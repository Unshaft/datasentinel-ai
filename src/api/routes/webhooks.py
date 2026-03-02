"""
Routes /webhooks — Gestion des webhooks de notification.

Permet d'enregistrer une URL qui recevra un POST JSON automatiquement
quand une analyse se termine.

Événements disponibles :
- analysis.completed  : POST /analyze ou POST /upload terminé
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.core.webhook_manager import add_webhook, get_webhooks, remove_webhook

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


class WebhookRequest(BaseModel):
    """Corps de la requête d'enregistrement."""

    url: str
    events: list[str] = ["analysis.completed"]
    description: str = ""


class WebhookResponse(BaseModel):
    """Réponse après enregistrement ou dans la liste."""

    webhook_id: str
    url: str
    events: list[str]
    description: str
    active: bool


@router.post(
    "",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enregistrer un webhook",
    description="""
    Enregistre une URL cible qui recevra un POST JSON à chaque événement.

    **Payload reçu par le webhook** :
    ```json
    {
      "event": "analysis.completed",
      "timestamp": "2026-03-02T10:00:00+00:00",
      "session_id": "session_abc123",
      "quality_score": 87.5,
      "status": "completed",
      "issues_count": 3,
      "needs_human_review": false
    }
    ```
    """,
)
async def register_webhook(request: WebhookRequest) -> WebhookResponse:
    """Enregistre un nouveau webhook."""
    webhook_id = add_webhook(request.url, request.events, request.description)
    # Récupérer le webhook fraîchement créé
    all_wh = {w["webhook_id"]: w for w in get_webhooks()}
    w = all_wh[webhook_id]
    return WebhookResponse(
        webhook_id=w["webhook_id"],
        url=w["url"],
        events=w["events"],
        description=w["description"],
        active=w["active"],
    )


@router.get(
    "",
    response_model=list[WebhookResponse],
    summary="Lister les webhooks",
    description="Retourne tous les webhooks enregistrés.",
)
async def list_webhooks() -> list[WebhookResponse]:
    """Liste tous les webhooks actifs."""
    return [
        WebhookResponse(
            webhook_id=w["webhook_id"],
            url=w["url"],
            events=w["events"],
            description=w["description"],
            active=w["active"],
        )
        for w in get_webhooks()
    ]


@router.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Supprimer un webhook",
    description="Supprime un webhook par son ID.",
)
async def delete_webhook(webhook_id: str) -> None:
    """Supprime un webhook enregistré."""
    if not remove_webhook(webhook_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook '{webhook_id}' introuvable.",
        )
