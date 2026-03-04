"""
Gestionnaire des profils de domaine métier — F32 (v1.0).

Un DomainProfile définit :
- trigger_types  : types sémantiques qui déclenchent l'activation du profil
- required_types : types obligatoires (absence → CONSTRAINT_VIOLATION CRITICAL)
- rules          : règles descriptives (texte libre, affichées dans les détails)
- severity_overrides : upgrade de sévérité par type sémantique

Persistance : ./data/domain_agents.json (même pattern que stats.json / webhooks.json).
Détection   : après F27, ratio = |col_types ∩ trigger_types| / |trigger_types|.
              Le profil actif avec le meilleur ratio ≥ min_match_ratio est activé.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_FILE = Path("./data/domain_agents.json")


# ---------------------------------------------------------------------------
# Modèles de données
# ---------------------------------------------------------------------------


@dataclass
class DomainRule:
    """Règle descriptive associée à un profil de domaine."""

    text: str
    applies_to_types: list[str] = field(default_factory=list)
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class DomainProfile:
    """Profil de validation spécialisé par domaine métier."""

    name: str
    description: str = ""
    trigger_types: list[str] = field(default_factory=list)
    min_match_ratio: float = 0.3
    required_types: list[str] = field(default_factory=list)
    rules: list[DomainRule] = field(default_factory=list)
    severity_overrides: dict[str, str] = field(default_factory=dict)
    active: bool = True
    domain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _profile_to_dict(p: DomainProfile) -> dict[str, Any]:
    d = asdict(p)
    return d


def _profile_from_dict(d: dict[str, Any]) -> DomainProfile:
    rules = [DomainRule(**r) for r in d.pop("rules", [])]
    return DomainProfile(**d, rules=rules)


# ---------------------------------------------------------------------------
# DomainManager singleton
# ---------------------------------------------------------------------------


class DomainManager:
    """
    Singleton — CRUD + détection des profils de domaine métier.

    Thread-safety : opérations synchrones légères, pas de verrou nécessaire
    (même pattern que StatsManager / FeedbackProcessor).
    """

    _instance: "DomainManager | None" = None

    def __new__(cls) -> "DomainManager":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._profiles: list[DomainProfile] = []
            inst._load()
            cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Charge les profils depuis le fichier JSON (silencieux si absent)."""
        if not _DATA_FILE.exists():
            return
        try:
            data = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
            self._profiles = [_profile_from_dict(d) for d in data.get("profiles", [])]
            logger.debug("DomainManager: %d profil(s) chargé(s)", len(self._profiles))
        except Exception as exc:
            logger.warning("DomainManager: erreur chargement JSON — %s", exc)

    def _save(self) -> None:
        """Persiste les profils dans le fichier JSON (best-effort)."""
        try:
            _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {"profiles": [_profile_to_dict(p) for p in self._profiles]}
            _DATA_FILE.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("DomainManager: erreur sauvegarde JSON — %s", exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, profile: DomainProfile) -> DomainProfile:
        """Ajoute un profil et persiste."""
        self._profiles.append(profile)
        self._save()
        logger.info("DomainManager: profil '%s' créé (%s)", profile.name, profile.domain_id)
        return profile

    def list_profiles(self, active_only: bool = True) -> list[DomainProfile]:
        """Renvoie les profils (actifs uniquement par défaut)."""
        if active_only:
            return [p for p in self._profiles if p.active]
        return list(self._profiles)

    def get(self, domain_id: str) -> DomainProfile | None:
        """Renvoie le profil par ID, ou None."""
        for p in self._profiles:
            if p.domain_id == domain_id:
                return p
        return None

    def delete(self, domain_id: str) -> bool:
        """Supprime définitivement un profil. Renvoie True si trouvé."""
        before = len(self._profiles)
        self._profiles = [p for p in self._profiles if p.domain_id != domain_id]
        if len(self._profiles) < before:
            self._save()
            logger.info("DomainManager: profil %s supprimé", domain_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Détection de domaine
    # ------------------------------------------------------------------

    def detect_domain(self, semantic_types: dict[str, Any]) -> DomainProfile | None:
        """
        Renvoie le profil actif avec le meilleur ratio de types déclencheurs.

        Args:
            semantic_types: dict col_name → {semantic_type, confidence, ...}
                            (output de SemanticProfilerAgent F27)

        Returns:
            DomainProfile si ratio ≥ min_match_ratio, sinon None.
        """
        if not semantic_types:
            return None

        col_types: set[str] = {
            info.get("semantic_type", "")
            for info in semantic_types.values()
            if info.get("semantic_type")
        }

        best: DomainProfile | None = None
        best_ratio = 0.0

        for profile in self.list_profiles(active_only=True):
            trigger_set = set(profile.trigger_types)
            if not trigger_set:
                continue
            ratio = len(col_types & trigger_set) / len(trigger_set)
            if ratio >= profile.min_match_ratio and ratio > best_ratio:
                best = profile
                best_ratio = ratio

        if best:
            logger.debug(
                "DomainManager: profil '%s' détecté (ratio=%.0f%%)",
                best.name,
                best_ratio * 100,
            )
        return best
