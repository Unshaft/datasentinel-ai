"""
Tests unitaires pour DatasetMemoryManager (F30 — v1.2).

Couvre :
- Déterminisme du dataset_id
- Première session → is_known=False
- Deuxième session → is_known=True, session_count=2
- avg_quality_score correct
- recurring_issues comptés
- Suggestion déclenchée après >60% sessions
- Fenêtre glissante (10 sessions max)
- Persistance round-trip
- Tendance dégradante détectée
"""

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.core.dataset_memory import DatasetMemoryManager, compute_dataset_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(cols: dict) -> pd.DataFrame:
    """Crée un DataFrame minimal avec les colonnes données."""
    return pd.DataFrame({k: [v] for k, v in cols.items()})


def _fresh_manager(tmp_path: Path) -> DatasetMemoryManager:
    """Retourne un manager avec état vide (contourne le singleton)."""
    manager = DatasetMemoryManager.__new__(DatasetMemoryManager)
    manager._data_lock = threading.Lock()
    manager._data = {"datasets": {}}
    # Rediriger la persistence vers tmp_path pour ne pas polluer les vrais fichiers
    object.__setattr__(manager, "_memory_file_override", tmp_path / "dataset_memory.json")
    return manager


# Patch _MEMORY_FILE pour tous les tests
@pytest.fixture(autouse=True)
def isolate_memory_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.dataset_memory._MEMORY_FILE", tmp_path / "dataset_memory.json")
    # Reset le singleton entre chaque test
    DatasetMemoryManager._instance = None
    yield
    DatasetMemoryManager._instance = None


# ---------------------------------------------------------------------------
# compute_dataset_id
# ---------------------------------------------------------------------------


def test_dataset_id_deterministic():
    """Même schéma → même dataset_id."""
    df1 = _make_df({"age": 25, "name": "Alice"})
    df2 = _make_df({"age": 30, "name": "Bob"})  # mêmes colonnes, valeurs différentes
    assert compute_dataset_id(df1) == compute_dataset_id(df2)


def test_dataset_id_different_schema():
    """Schéma différent → dataset_id différent."""
    df1 = _make_df({"age": 25, "name": "Alice"})
    df2 = _make_df({"salary": 50000, "department": "RH"})
    assert compute_dataset_id(df1) != compute_dataset_id(df2)


def test_dataset_id_prefix():
    """Le dataset_id commence par 'dataset_'."""
    df = _make_df({"col": 1})
    did = compute_dataset_id(df)
    assert did.startswith("dataset_")


# ---------------------------------------------------------------------------
# Première session
# ---------------------------------------------------------------------------


def test_first_session_is_not_known():
    """Première analyse → is_known=False."""
    mgr = DatasetMemoryManager()
    df = _make_df({"age": 25})
    dataset_id = compute_dataset_id(df)

    was_known = mgr.get_entry(dataset_id) is not None
    mgr.record_session(dataset_id, "session_001", 80.0, [])
    info = mgr.get_memory_info(dataset_id, was_known)

    assert info["is_known"] is False
    assert info["session_count"] == 1
    assert info["avg_quality_score"] == 80.0
    assert info["trend"] == "new"


# ---------------------------------------------------------------------------
# Deuxième session
# ---------------------------------------------------------------------------


def test_second_session_is_known():
    """Deuxième analyse du même dataset → is_known=True."""
    mgr = DatasetMemoryManager()
    df = _make_df({"age": 25})
    dataset_id = compute_dataset_id(df)

    mgr.record_session(dataset_id, "session_001", 80.0, [])

    was_known = mgr.get_entry(dataset_id) is not None  # True après la 1ère
    mgr.record_session(dataset_id, "session_002", 70.0, [])
    info = mgr.get_memory_info(dataset_id, was_known)

    assert info["is_known"] is True
    assert info["session_count"] == 2


def test_avg_quality_score_updates():
    """avg_quality_score est la moyenne des sessions."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_test001"

    mgr.record_session(dataset_id, "s1", 80.0, [])
    mgr.record_session(dataset_id, "s2", 60.0, [])
    info = mgr.get_memory_info(dataset_id, True)

    assert info["avg_quality_score"] == 70.0


# ---------------------------------------------------------------------------
# Recurring issues
# ---------------------------------------------------------------------------


class _FakeIssue:
    """Issue minimale pour les tests."""
    def __init__(self, issue_type: str, column: str | None = None):
        self.issue_type = issue_type
        self.column = column


def test_recurring_issues_counted():
    """recurring_issues est incrémenté à chaque session."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_recurring"

    issue = _FakeIssue("missing_values", "email")
    mgr.record_session(dataset_id, "s1", 80.0, [issue])
    mgr.record_session(dataset_id, "s2", 75.0, [issue])

    entry = mgr.get_entry(dataset_id)
    assert entry["recurring_issues"]["missing_values"] == 2


# ---------------------------------------------------------------------------
# Suggestions pro-actives
# ---------------------------------------------------------------------------


def test_suggestion_triggered_above_threshold():
    """Suggestion générée si issue présente dans >60% des sessions."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_suggest"
    issue = _FakeIssue("missing_values", "salary")

    # 4 sessions avec l'issue sur 5 → 80% > 60%
    for i in range(4):
        mgr.record_session(dataset_id, f"s{i}", 70.0, [issue])
    mgr.record_session(dataset_id, "s4", 70.0, [])  # 1 session sans

    entry = mgr.get_entry(dataset_id)
    assert len(entry["suggested_rules"]) > 0
    assert any("salary" in r for r in entry["suggested_rules"])


def test_no_suggestion_below_threshold():
    """Pas de suggestion si issue < 60% des sessions."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_nosuggest"
    issue = _FakeIssue("missing_values", "salary")

    # 1 session avec l'issue sur 3 → 33% < 60%
    mgr.record_session(dataset_id, "s0", 70.0, [issue])
    mgr.record_session(dataset_id, "s1", 70.0, [])
    mgr.record_session(dataset_id, "s2", 70.0, [])

    entry = mgr.get_entry(dataset_id)
    assert len(entry["suggested_rules"]) == 0


# ---------------------------------------------------------------------------
# Fenêtre glissante
# ---------------------------------------------------------------------------


def test_sliding_window_keeps_last_10():
    """Seulement les 10 dernières sessions sont conservées."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_window"

    for i in range(15):
        mgr.record_session(dataset_id, f"s{i}", float(i), [])

    entry = mgr.get_entry(dataset_id)
    assert len(entry["sessions"]) == 10
    # Les 10 dernières = s5..s14
    assert entry["sessions"][0]["session_id"] == "s5"
    assert entry["sessions"][-1]["session_id"] == "s14"


# ---------------------------------------------------------------------------
# Persistance round-trip
# ---------------------------------------------------------------------------


def test_persist_and_reload(tmp_path, monkeypatch):
    """L'état est correctement sauvegardé puis rechargé."""
    mem_file = tmp_path / "dataset_memory.json"
    monkeypatch.setattr("src.core.dataset_memory._MEMORY_FILE", mem_file)
    DatasetMemoryManager._instance = None

    mgr = DatasetMemoryManager()
    mgr.record_session("dataset_persist", "s1", 85.0, [])

    # Simuler un rechargement en réinitialisant le singleton
    DatasetMemoryManager._instance = None
    mgr2 = DatasetMemoryManager()

    entry = mgr2.get_entry("dataset_persist")
    assert entry is not None
    assert entry["session_count"] == 1
    assert entry["sessions"][0]["quality_score"] == 85.0


# ---------------------------------------------------------------------------
# Tendance
# ---------------------------------------------------------------------------


def test_degrading_trend_detected():
    """Tendance 'degrading' si dernier score < moyenne précédente - 5pts."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_degrade"

    mgr.record_session(dataset_id, "s1", 90.0, [])
    mgr.record_session(dataset_id, "s2", 88.0, [])
    mgr.record_session(dataset_id, "s3", 80.0, [])  # 80 vs avg(89)=9pt drop

    info = mgr.get_memory_info(dataset_id, True)
    assert info["trend"] == "degrading"


def test_improving_trend_detected():
    """Tendance 'improving' si dernier score > moyenne précédente + 5pts."""
    mgr = DatasetMemoryManager()
    dataset_id = "dataset_improve"

    mgr.record_session(dataset_id, "s1", 60.0, [])
    mgr.record_session(dataset_id, "s2", 62.0, [])
    mgr.record_session(dataset_id, "s3", 75.0, [])  # 75 vs avg(61)=14pt gain

    info = mgr.get_memory_info(dataset_id, True)
    assert info["trend"] == "improving"
