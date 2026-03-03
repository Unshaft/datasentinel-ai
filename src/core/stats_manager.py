"""
Gestionnaire de statistiques agrégées (v0.6 — F22).

Collecte des métriques d'utilisation et les persiste en JSON.
Singleton process-level, thread-safe pour les opérations courantes.
"""

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


_STATS_FILE = Path("./data/stats.json")
_SCORE_BUCKETS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


def _bucket(score: float) -> str:
    if score < 20:
        return "0-20"
    if score < 40:
        return "20-40"
    if score < 60:
        return "40-60"
    if score < 80:
        return "60-80"
    return "80-100"


class StatsManager:
    """
    Gestionnaire de statistiques agrégées avec persistance JSON.

    Persiste dans ./data/stats.json au même format que webhooks.json.
    Thread-safe pour les incréments concurrents.
    """

    _instance: "StatsManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "StatsManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._data_lock = threading.Lock()
                    instance._data: dict = {}
                    instance._load_from_disk()
                    cls._instance = instance
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Charge les stats depuis le fichier JSON (best-effort)."""
        try:
            if _STATS_FILE.exists():
                with open(_STATS_FILE, encoding="utf-8") as f:
                    self._data = json.load(f)
            else:
                self._data = self._empty_data()
        except Exception:
            self._data = self._empty_data()

    def _save_to_disk(self) -> None:
        """Persiste les stats sur disque (best-effort)."""
        try:
            _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @staticmethod
    def _empty_data() -> dict:
        return {
            "total_sessions": 0,
            "score_sum": 0.0,
            "issue_type_counts": {},
            "sessions_by_day": {},
            "score_distribution": {b: 0 for b in _SCORE_BUCKETS},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_session(self, quality_score: float, issue_types: list[str]) -> None:
        """Enregistre une session analysée (best-effort, ne bloque pas)."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with self._data_lock:
                d = self._data
                d["total_sessions"] = d.get("total_sessions", 0) + 1
                d["score_sum"] = d.get("score_sum", 0.0) + quality_score

                # Sessions par jour
                sbd = d.setdefault("sessions_by_day", {})
                sbd[today] = sbd.get(today, 0) + 1

                # Distribution des scores
                dist = d.setdefault("score_distribution", {b: 0 for b in _SCORE_BUCKETS})
                bkt = _bucket(quality_score)
                dist[bkt] = dist.get(bkt, 0) + 1

                # Top issue types
                itc = d.setdefault("issue_type_counts", {})
                for it in issue_types:
                    itc[it] = itc.get(it, 0) + 1

            self._save_to_disk()
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Retourne les statistiques agrégées."""
        with self._data_lock:
            d = self._data
            total = d.get("total_sessions", 0)
            score_sum = d.get("score_sum", 0.0)
            avg_score = round(score_sum / total, 2) if total > 0 else 0.0

            # Top 5 issue types
            itc: dict[str, int] = d.get("issue_type_counts", {})
            top5 = dict(
                Counter(itc).most_common(5)
            )

            # Sessions des 7 derniers jours seulement
            sbd: dict[str, int] = d.get("sessions_by_day", {})
            sorted_days = sorted(sbd.keys(), reverse=True)[:7]
            last7 = {k: sbd[k] for k in sorted(sorted_days)}

            dist = d.get("score_distribution", {b: 0 for b in _SCORE_BUCKETS})

        return {
            "total_sessions": total,
            "avg_quality_score": avg_score,
            "top_issue_types": top5,
            "sessions_by_day": last7,
            "score_distribution": dist,
        }

    def reset(self) -> None:
        """Remet les statistiques à zéro."""
        with self._data_lock:
            self._data = self._empty_data()
        self._save_to_disk()


def get_stats_manager() -> StatsManager:
    """Retourne le singleton StatsManager."""
    return StatsManager()
