"""
Persistance des sessions AgentContext via Redis.

Permet de reprendre une session entre deux appels API sans renvoyer
les données. Exemple : POST /analyze → session_id → GET /analyze/{session_id}.

Design :
- Redis comme backend principal
- Dict in-memory comme fallback si Redis est indisponible
- TTL configurable (défaut 1h via settings.session_ttl)
- Sérialisation JSON via Pydantic .model_dump() / .model_validate()
- DataFrame stocké en base64+parquet sous la clé df:{session_id}

Singleton via get_session_store() — une seule instance par process.
"""

import base64
import io
import json
import logging
from typing import Any

from src.core.config import settings
from src.core.models import AgentContext

logger = logging.getLogger(__name__)


class InMemoryFallback:
    """
    Fallback in-memory quand Redis est indisponible.

    Stocke les sessions dans un dict process-level.
    Pas de TTL réel (les sessions vivent le temps du process).
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def ping(self) -> bool:
        return True


class SessionStore:
    """
    Store de sessions AgentContext.

    Utilise Redis comme backend principal, avec fallback automatique
    vers un dict in-memory si Redis n'est pas disponible.
    """

    _instance: "SessionStore | None" = None

    def __init__(self, redis_client=None) -> None:
        """
        Initialise le store.

        Args:
            redis_client: Client Redis (ou mock pour les tests).
                          Si None, tente de se connecter via settings.redis_url.
        """
        if redis_client is not None:
            self._client = redis_client
            self._using_fallback = False
        else:
            self._client = self._connect()

    def _connect(self):
        """Tente la connexion Redis, retourne le fallback si échec."""
        try:
            import redis

            client = redis.from_url(settings.redis_url, decode_responses=True)
            client.ping()
            logger.info("SessionStore : connecté à Redis (%s)", settings.redis_url)
            self._using_fallback = False
            return client
        except Exception as e:
            logger.info(
                "SessionStore : Redis indisponible (%s). Fallback in-memory activé.", e
            )
            self._using_fallback = True
            return InMemoryFallback()

    @property
    def using_fallback(self) -> bool:
        """True si Redis est indisponible et qu'on utilise le fallback."""
        return getattr(self, "_using_fallback", True)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def save(
        self,
        session_id: str,
        context: AgentContext,
        ttl: int | None = None,
    ) -> None:
        """
        Persiste un AgentContext en JSON.

        Args:
            session_id: Clé de session
            context: Contexte à sauvegarder
            ttl: Durée de vie en secondes (None → settings.session_ttl)
        """
        effective_ttl = ttl if ttl is not None else settings.session_ttl
        serialized = json.dumps(context.model_dump(mode="json"))
        self._client.set(f"session:{session_id}", serialized, ex=effective_ttl)

    def load(self, session_id: str) -> AgentContext | None:
        """
        Charge un AgentContext depuis le store.

        Args:
            session_id: Clé de session

        Returns:
            AgentContext ou None si session absente / expirée
        """
        raw = self._client.get(f"session:{session_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        return AgentContext.model_validate(data)

    def delete(self, session_id: str) -> None:
        """Supprime une session (contexte + DataFrame si présent)."""
        self._client.delete(f"session:{session_id}")
        self._client.delete(f"df:{session_id}")

    def exists(self, session_id: str) -> bool:
        """Vérifie si une session existe."""
        return self._client.get(f"session:{session_id}") is not None

    def save_dataframe(
        self,
        session_id: str,
        df: "pd.DataFrame",
        ttl: int | None = None,
    ) -> None:
        """
        Persiste un DataFrame en base64+parquet.

        Args:
            session_id: Clé de session
            df: DataFrame à sauvegarder
            ttl: Durée de vie en secondes (None → settings.session_ttl)
        """
        import pandas as pd  # noqa: F401 — guard lazy import

        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        effective_ttl = ttl if ttl is not None else settings.session_ttl
        self._client.set(f"df:{session_id}", encoded, ex=effective_ttl)

    def load_dataframe(self, session_id: str) -> "pd.DataFrame | None":
        """
        Charge un DataFrame depuis le store.

        Returns:
            DataFrame ou None si absent / expiré
        """
        import pandas as pd

        raw = self._client.get(f"df:{session_id}")
        if raw is None:
            return None
        buf = io.BytesIO(base64.b64decode(raw))
        return pd.read_parquet(buf)


def get_session_store() -> SessionStore:
    """
    Retourne le SessionStore singleton.

    Crée l'instance au premier appel (lazy init).
    Les appels suivants retournent la même instance.
    """
    if SessionStore._instance is None:
        SessionStore._instance = SessionStore()
    return SessionStore._instance
