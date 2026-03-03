"""
Gestionnaire de jobs asynchrones (v0.6 — F21).

Permet de soumettre des analyses longues (fichiers volumineux) de façon
non bloquante via asyncio.create_task.

Les états des jobs sont persistés dans le SessionStore sous clé job:{job_id}.
TTL : 2h (7200s).
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from src.memory.session_store import get_session_store

_JOB_TTL = 7200  # 2 heures


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


class JobManager:
    """
    Gestionnaire de jobs asyncio.

    Ne gère pas lui-même les tâches asyncio (pas de boucle event),
    fournit uniquement les primitives de persistance d'état.
    La tâche est créée par la route via asyncio.create_task.
    """

    _instance: "JobManager | None" = None

    def __new__(cls) -> "JobManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Primitives de persistance
    # ------------------------------------------------------------------

    def _store(self) -> Any:
        return get_session_store()

    def create_job(self, filename: str) -> str:
        """Crée un job en état pending et retourne son job_id."""
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        data = {
            "job_id": job_id,
            "filename": filename,
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._store()._client.set(_job_key(job_id), json.dumps(data), ex=_JOB_TTL)
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: float = 0.0,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Met à jour l'état d'un job existant."""
        raw = self._store()._client.get(_job_key(job_id))
        if raw is None:
            return
        data = json.loads(raw)
        data["status"] = status
        data["progress"] = progress
        if result is not None:
            data["result"] = result
        if error is not None:
            data["error"] = error
        self._store()._client.set(_job_key(job_id), json.dumps(data), ex=_JOB_TTL)

    def get_job(self, job_id: str) -> dict | None:
        """Charge l'état d'un job (None si expiré ou inexistant)."""
        raw = self._store()._client.get(_job_key(job_id))
        if raw is None:
            return None
        return json.loads(raw)


def get_job_manager() -> JobManager:
    """Retourne le singleton JobManager."""
    return JobManager()
