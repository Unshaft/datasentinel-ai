"""
Processeur de feedback qui améliore le comportement du système (v0.7 — F26).

Chaque feedback utilisateur déclenche :
- False positive : règle d'exception dans ChromaDB + décrémentation de confidence
- Confirmation : incrément de confidence + renforcement règle
- Correction custom : ajout d'une règle positive dans ChromaDB

Persistance dans ./data/feedback_stats.json
"""

import json
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.chroma_store import ChromaStore

_STATS_FILE = Path("./data/feedback_stats.json")
_CONF_FLOOR = 0.5
_CONF_CEIL = 0.99
_FP_DECREMENT = 0.05
_CONF_INCREMENT = 0.02
_FP_THRESHOLD = 5  # nbre de FP avant de baisser la confiance


class FeedbackProcessor:
    """
    Traite les feedbacks utilisateur et ajuste les paramètres du système.

    Singleton — thread-safe pour les opérations concurrentes.
    """

    _instance: "FeedbackProcessor | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "FeedbackProcessor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._data_lock = threading.Lock()
                    inst._stats: dict[str, Any] = {}
                    inst._load()
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if _STATS_FILE.exists():
                self._stats = json.loads(_STATS_FILE.read_text(encoding="utf-8"))
            else:
                self._stats = {}
        except Exception:
            self._stats = {}

    def _save(self) -> None:
        try:
            _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATS_FILE.write_text(
                json.dumps(self._stats, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, feedback: Any, chroma_store: "ChromaStore") -> None:
        """
        Traite un feedback utilisateur de façon asynchrone (best-effort).

        Args:
            feedback: FeedbackRequest ou tout objet avec is_correct, target_id,
                      target_type, session_id, user_correction.
            chroma_store: Instance ChromaStore pour les règles d'exception.
        """
        try:
            is_correct: bool | None = getattr(feedback, "is_correct", None)
            target_id: str = getattr(feedback, "target_id", "unknown")
            target_type: str = getattr(feedback, "target_type", "unknown")
            user_correction: str | None = getattr(feedback, "user_correction", None)

            # Dériver le check_type depuis target_type
            check_type = target_type  # issue / proposal / decision

            if is_correct is False:
                self._handle_false_positive(check_type, target_id, chroma_store)
            elif is_correct is True:
                self._handle_confirmation(check_type, chroma_store)

            if user_correction:
                self._handle_custom_correction(
                    check_type, target_id, user_correction, chroma_store
                )

            self._save()
        except Exception:
            pass  # Never block the API response

    def get_adjustments(self) -> dict[str, float]:
        """Retourne les ajustements de confiance courants."""
        with self._data_lock:
            return dict(self._stats.get("confidence_adjustments", {}))

    def get_summary(self) -> dict[str, Any]:
        """Retourne un résumé pour GET /stats."""
        with self._data_lock:
            fp = self._stats.get("false_positive_stats", {})
            conf = self._stats.get("confidence_adjustments", {})
            examples = self._stats.get("examples_added", 0)
            confirmations = self._stats.get("confirmations", 0)
            return {
                "false_positives_corrected": sum(fp.values()),
                "confirmations": confirmations,
                "examples_added": examples,
                "checks_adjusted": len(conf),
            }

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_false_positive(
        self, check_type: str, target_id: str, store: "ChromaStore"
    ) -> None:
        with self._data_lock:
            fp = self._stats.setdefault("false_positive_stats", {})
            fp[check_type] = fp.get(check_type, 0) + 1
            count = fp[check_type]

            # Ajuste la confiance si seuil dépassé
            if count > _FP_THRESHOLD:
                adj = self._stats.setdefault("confidence_adjustments", {})
                current = adj.get(check_type, 1.0)
                adj[check_type] = round(max(_CONF_FLOOR, current - _FP_DECREMENT), 4)

        # Écrit une règle d'exception dans ChromaDB (best-effort)
        try:
            rule_id = f"fp_exception_{uuid.uuid4().hex[:12]}"
            store.add_rule(
                rule_id=rule_id,
                rule_text=f"Feedback: '{target_id}' a été signalé comme faux positif pour {check_type}",
                rule_type="exception",
                metadata={"severity": "low", "category": "feedback", "source": "feedback_processor"},
            )
        except Exception:
            pass

    def _handle_confirmation(self, check_type: str, store: "ChromaStore") -> None:
        with self._data_lock:
            self._stats["confirmations"] = self._stats.get("confirmations", 0) + 1
            adj = self._stats.setdefault("confidence_adjustments", {})
            current = adj.get(check_type, 1.0)
            adj[check_type] = round(min(_CONF_CEIL, current + _CONF_INCREMENT), 4)

    def _handle_custom_correction(
        self, check_type: str, target_id: str, correction: str, store: "ChromaStore"
    ) -> None:
        with self._data_lock:
            self._stats["examples_added"] = self._stats.get("examples_added", 0) + 1

        try:
            rule_id = f"example_{uuid.uuid4().hex[:12]}"
            store.add_rule(
                rule_id=rule_id,
                rule_text=f"Correction utilisateur pour {check_type}: {correction}",
                rule_type="example",
                metadata={"severity": "low", "category": "feedback", "source": "feedback_processor"},
            )
        except Exception:
            pass


def get_feedback_processor() -> FeedbackProcessor:
    """Retourne le singleton FeedbackProcessor."""
    return FeedbackProcessor()
