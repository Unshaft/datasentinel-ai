"""
Gestionnaire de webhooks — enregistrement et envoi de notifications async.

Quand une analyse se termine, l'endpoint appelle fire_webhooks() en background.
Chaque abonné enregistré reçoit un POST JSON avec le résultat.

Design :
- Stockage in-memory + persistence JSON sur disque (v0.4)
- fire_webhooks() est async et non-bloquant (asyncio.gather + return_exceptions)
- Un webhook qui échoue ne bloque pas les autres
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Fichier de persistence (créé dans le répertoire data/)
_PERSIST_PATH = Path("./data/webhooks.json")

# Stockage in-memory des webhooks enregistrés
_webhooks: dict[str, dict[str, Any]] = {}


def _load_from_disk() -> None:
    """Charge les webhooks persistés depuis le fichier JSON (au démarrage)."""
    try:
        if _PERSIST_PATH.exists():
            data = json.loads(_PERSIST_PATH.read_text(encoding="utf-8"))
            _webhooks.update(data)
            logger.info("Webhooks chargés depuis %s (%d entrées)", _PERSIST_PATH, len(data))
    except Exception as e:
        logger.warning("Impossible de charger les webhooks depuis le disque : %s", e)


def _save_to_disk() -> None:
    """Persiste les webhooks dans le fichier JSON."""
    try:
        _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PERSIST_PATH.write_text(
            json.dumps(_webhooks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Impossible de persister les webhooks : %s", e)


# Chargement automatique au démarrage du module
_load_from_disk()


def add_webhook(url: str, events: list[str], description: str = "") -> str:
    """
    Enregistre un webhook et retourne son ID.

    Args:
        url: URL cible (POST JSON)
        events: Liste d'événements à écouter (ex: ["analysis.completed"])
        description: Description libre

    Returns:
        webhook_id généré
    """
    webhook_id = f"wh_{uuid.uuid4().hex[:12]}"
    _webhooks[webhook_id] = {
        "webhook_id": webhook_id,
        "url": url,
        "events": events,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }
    _save_to_disk()
    logger.info("Webhook enregistré : %s → %s", webhook_id, url)
    return webhook_id


def remove_webhook(webhook_id: str) -> bool:
    """
    Supprime un webhook.

    Returns:
        True si trouvé et supprimé, False sinon
    """
    if webhook_id in _webhooks:
        del _webhooks[webhook_id]
        _save_to_disk()
        logger.info("Webhook supprimé : %s", webhook_id)
        return True
    return False


def get_webhooks() -> list[dict[str, Any]]:
    """Retourne tous les webhooks enregistrés."""
    return list(_webhooks.values())


async def fire_webhooks(event: str, payload: dict[str, Any]) -> None:
    """
    Déclenche tous les webhooks abonnés à l'événement.

    Envoie un POST JSON en parallèle à chaque URL abonnée.
    Les erreurs individuelles sont loguées mais n'arrêtent pas les autres.

    Args:
        event: Nom de l'événement (ex: "analysis.completed")
        payload: Données à envoyer (session_id, quality_score, etc.)
    """
    matching = [
        w for w in _webhooks.values()
        if w["active"] and event in w["events"]
    ]

    if not matching:
        return

    notification = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            _send_one(client, w["url"], notification, w["webhook_id"])
            for w in matching
        ]
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    webhook_id: str,
) -> None:
    """Envoie la notification à une URL. Logge l'erreur si échec."""
    try:
        resp = await client.post(url, json=payload)
        logger.info("Webhook %s → %s [HTTP %d]", webhook_id, url, resp.status_code)
    except Exception as exc:
        logger.warning("Webhook %s → %s FAILED : %s", webhook_id, url, exc)
