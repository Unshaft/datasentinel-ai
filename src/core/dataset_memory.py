"""
Mémoire inter-sessions par dataset (v1.2 — F30).

Persiste un profil de qualité par dataset_id (fingerprint du schéma).
À chaque analyse, l'historique est enrichi : score moyen, issues récurrentes,
tendance qualité, suggestions de règles pro-actives.

Singleton process-level, thread-safe.
"""

import hashlib
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


_MEMORY_FILE = Path("./data/dataset_memory.json")
_MAX_SESSIONS_PER_DATASET = 10
_SUGGESTION_THRESHOLD = 0.6   # >60% des sessions → suggestion
_TREND_MIN_SESSIONS = 3       # sessions min pour calculer une tendance
_TREND_DELTA = 5.0            # points de différence pour "improving"/"degrading"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_dataset_id(df: pd.DataFrame) -> str:
    """
    Retourne un fingerprint déterministe basé sur le schéma (colonnes + types).

    Stable même si le nombre de lignes change entre deux analyses.
    """
    schema_str = "|".join(
        f"{col}:{str(dtype)}"
        for col, dtype in sorted(df.dtypes.items())
    )
    return "dataset_" + hashlib.md5(schema_str.encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DatasetMemoryManager
# ---------------------------------------------------------------------------


class DatasetMemoryManager:
    """
    Gestionnaire de mémoire inter-sessions par dataset.

    Chaque dataset_id (fingerprint du schéma) accumule :
    - l'historique des scores des 10 dernières analyses
    - le comptage des issues récurrentes par type et par colonne
    - des suggestions de règles générées automatiquement
    """

    _instance: "DatasetMemoryManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "DatasetMemoryManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._data_lock = threading.Lock()
                    instance._data: dict = {"datasets": {}}
                    instance._load_from_disk()
                    cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        try:
            if _MEMORY_FILE.exists():
                with open(_MEMORY_FILE, encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._data = {"datasets": {}}
        except Exception:
            self._data = {"datasets": {}}

    def _save_to_disk(self) -> None:
        try:
            _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_entry(self, dataset_id: str) -> dict:
        """Retourne (et crée si absent) l'entrée pour un dataset_id."""
        datasets = self._data.setdefault("datasets", {})
        if dataset_id not in datasets:
            datasets[dataset_id] = {
                "dataset_id": dataset_id,
                "first_seen": _now_iso(),
                "last_seen": _now_iso(),
                "session_count": 0,
                "score_sum": 0.0,
                "recurring_issues": {},        # issue_type → count
                "problematic_columns": {},     # col_name → count
                "sessions": [],                # list of DatasetSession dicts
                "suggested_rules": [],
            }
        return datasets[dataset_id]

    @staticmethod
    def _compute_trend(sessions: list[dict]) -> str:
        """
        Calcule la tendance à partir des scores des sessions récentes.

        Requiert au moins _TREND_MIN_SESSIONS sessions.
        Compare la dernière session contre la moyenne des précédentes.
        """
        if len(sessions) < _TREND_MIN_SESSIONS:
            return "new" if len(sessions) <= 1 else "stable"

        last_score = sessions[-1]["quality_score"]
        previous_avg = sum(s["quality_score"] for s in sessions[:-1]) / len(sessions[:-1])

        if last_score > previous_avg + _TREND_DELTA:
            return "improving"
        if last_score < previous_avg - _TREND_DELTA:
            return "degrading"
        return "stable"

    @staticmethod
    def _generate_suggestions(entry: dict) -> list[str]:
        """
        Génère des suggestions de règles si un problème apparaît dans >60% des sessions.

        Appelé après chaque mise à jour de l'entrée.
        """
        session_count = entry["session_count"]
        if session_count == 0:
            return []

        suggestions: list[str] = []

        # Issues récurrentes par type
        for issue_type, count in entry["recurring_issues"].items():
            ratio = count / session_count
            if ratio <= _SUGGESTION_THRESHOLD:
                continue
            if issue_type == "missing_values":
                # Cherche quelle(s) colonne(s) sont concernées
                for col, col_count in entry["problematic_columns"].items():
                    col_ratio = col_count / session_count
                    if col_ratio > _SUGGESTION_THRESHOLD:
                        rule = (
                            f"Colonne '{col}' ne devrait pas contenir "
                            f"de valeurs nulles (absente dans {col_count}/{session_count} sessions)"
                        )
                        if rule not in suggestions:
                            suggestions.append(rule)
            elif issue_type == "format_error":
                for col, col_count in entry["problematic_columns"].items():
                    col_ratio = col_count / session_count
                    if col_ratio > _SUGGESTION_THRESHOLD:
                        rule = (
                            f"Colonne '{col}' doit respecter un format strict "
                            f"(erreurs dans {col_count}/{session_count} sessions)"
                        )
                        if rule not in suggestions:
                            suggestions.append(rule)

        # Tendance dégradante
        trend = DatasetMemoryManager._compute_trend(entry["sessions"])
        if trend == "degrading":
            rule = (
                "Score en baisse sur les dernières analyses — "
                "lancer un audit complet recommandé"
            )
            if rule not in suggestions:
                suggestions.append(rule)

        return suggestions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_session(
        self,
        dataset_id: str,
        session_id: str,
        quality_score: float,
        issues: list,
    ) -> None:
        """
        Enregistre une session dans la mémoire du dataset (best-effort).

        Args:
            dataset_id: Fingerprint du schéma
            session_id: ID de la session analytique
            quality_score: Score 0–100
            issues: Liste de QualityIssue (ou dicts avec issue_type/column)
        """
        try:
            with self._data_lock:
                entry = self._get_or_create_entry(dataset_id)

                # Comptage issues par type et par colonne
                issue_types: list[str] = []
                top_columns: list[str] = []
                issue_type_counts: dict[str, int] = {}

                for issue in issues:
                    # Supporte QualityIssue (objet) ou dict
                    if hasattr(issue, "issue_type"):
                        itype = issue.issue_type.value if hasattr(issue.issue_type, "value") else str(issue.issue_type)
                        col = issue.column or ""
                    else:
                        itype = str(issue.get("issue_type", ""))
                        col = str(issue.get("column", ""))

                    issue_types.append(itype)
                    issue_type_counts[itype] = issue_type_counts.get(itype, 0) + 1
                    if col:
                        top_columns.append(col)
                        entry["problematic_columns"][col] = (
                            entry["problematic_columns"].get(col, 0) + 1
                        )

                # Mise à jour des compteurs globaux
                for itype in issue_types:
                    entry["recurring_issues"][itype] = (
                        entry["recurring_issues"].get(itype, 0) + 1
                    )

                # Session record
                session_record = {
                    "session_id": session_id,
                    "timestamp": _now_iso(),
                    "quality_score": quality_score,
                    "issue_counts": issue_type_counts,
                    "top_columns": list(dict.fromkeys(top_columns))[:5],  # unique, max 5
                }
                entry["sessions"].append(session_record)

                # Fenêtre glissante : conserver seulement les N dernières sessions
                if len(entry["sessions"]) > _MAX_SESSIONS_PER_DATASET:
                    entry["sessions"] = entry["sessions"][-_MAX_SESSIONS_PER_DATASET:]

                # Métriques globales
                entry["session_count"] = entry.get("session_count", 0) + 1
                entry["score_sum"] = entry.get("score_sum", 0.0) + quality_score
                entry["last_seen"] = _now_iso()

                # Suggestions pro-actives
                entry["suggested_rules"] = self._generate_suggestions(entry)

            self._save_to_disk()
        except Exception:
            pass

    def get_entry(self, dataset_id: str) -> dict | None:
        """Retourne l'entrée brute d'un dataset, ou None si inconnu."""
        with self._data_lock:
            return self._data.get("datasets", {}).get(dataset_id)

    def get_memory_info(self, dataset_id: str, was_known_before: bool) -> dict:
        """
        Retourne un résumé DatasetMemoryInfo prêt pour AnalyzeResponse.

        Args:
            dataset_id: Fingerprint du schéma
            was_known_before: True si le dataset était déjà connu avant cet appel
        """
        with self._data_lock:
            entry = self._data.get("datasets", {}).get(dataset_id)

        if entry is None:
            return {
                "dataset_id": dataset_id,
                "is_known": False,
                "session_count": 0,
                "avg_quality_score": 0.0,
                "trend": "new",
                "recurring_issues": [],
                "suggested_rules": [],
            }

        session_count = entry.get("session_count", 0)
        score_sum = entry.get("score_sum", 0.0)
        avg_score = round(score_sum / session_count, 2) if session_count > 0 else 0.0

        # Top 3 issue types récurrentes
        ri: dict[str, int] = entry.get("recurring_issues", {})
        top3 = [k for k, _ in Counter(ri).most_common(3)]

        sessions = entry.get("sessions", [])
        trend = self._compute_trend(sessions)

        return {
            "dataset_id": dataset_id,
            "is_known": was_known_before,
            "session_count": session_count,
            "avg_quality_score": avg_score,
            "trend": trend,
            "recurring_issues": top3,
            "suggested_rules": entry.get("suggested_rules", []),
        }

    def get_history(self, dataset_id: str) -> dict | None:
        """
        Retourne l'historique complet d'un dataset pour GET /datasets/{id}/history.

        Returns None si le dataset est inconnu.
        """
        with self._data_lock:
            entry = self._data.get("datasets", {}).get(dataset_id)

        if entry is None:
            return None

        session_count = entry.get("session_count", 0)
        score_sum = entry.get("score_sum", 0.0)
        avg_score = round(score_sum / session_count, 2) if session_count > 0 else 0.0
        sessions = entry.get("sessions", [])

        return {
            "dataset_id": dataset_id,
            "first_seen": entry.get("first_seen", ""),
            "last_seen": entry.get("last_seen", ""),
            "session_count": session_count,
            "avg_quality_score": avg_score,
            "trend": self._compute_trend(sessions),
            "recurring_issues": entry.get("recurring_issues", {}),
            "problematic_columns": entry.get("problematic_columns", {}),
            "suggested_rules": entry.get("suggested_rules", []),
            "sessions": sessions,
        }


def get_dataset_memory_manager() -> DatasetMemoryManager:
    """Retourne le singleton DatasetMemoryManager."""
    return DatasetMemoryManager()
