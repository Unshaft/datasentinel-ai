"""
Tests unitaires pour ChromaStore.

Couvre les deux bugs corrigés :
- Bug 2 : was_correct=None interdite dans les métadonnées ChromaDB
- Bug 3 : crash quand on interroge une collection vide
"""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Les variables d'env sont déjà posées dans conftest.py


class TestChromaStoreEmptyCollection:
    """
    Bug 3 — search_rules / find_similar_decisions / search_similar_feedback
    ne doivent pas planter sur une collection vide.
    """

    @pytest.fixture(autouse=True)
    def chroma_store(self, tmp_path):
        """Crée un ChromaStore isolé dans un répertoire temporaire."""
        # Réinitialise le singleton entre chaque test
        from src.memory import chroma_store as cs_module
        cs_module.ChromaStore._instance = None

        with patch("src.memory.chroma_store.settings") as mock_settings:
            mock_settings.chroma_persist_path = tmp_path / "chroma"
            mock_settings.chroma_rules_collection = "test_rules"
            mock_settings.chroma_decisions_collection = "test_decisions"
            mock_settings.chroma_feedback_collection = "test_feedback"

            from src.memory.chroma_store import ChromaStore
            store = ChromaStore(persist_path=tmp_path / "chroma")
            yield store

        # Nettoyage du singleton après le test
        cs_module.ChromaStore._instance = None

    def test_search_rules_empty_collection_returns_empty_list(self, chroma_store):
        """search_rules() sur collection vide doit retourner [] sans exception."""
        result = chroma_store.search_rules(query="uniqueness constraint")
        assert result == []

    def test_find_similar_decisions_empty_returns_empty_list(self, chroma_store):
        """find_similar_decisions() sur collection vide doit retourner []."""
        result = chroma_store.find_similar_decisions(query="detect anomaly in salary")
        assert result == []

    def test_search_similar_feedback_empty_returns_empty_list(self, chroma_store):
        """search_similar_feedback() sur collection vide doit retourner []."""
        result = chroma_store.search_similar_feedback(query="imputation was wrong")
        assert result == []

    def test_search_rules_after_adding_one_rule(self, chroma_store):
        """Après ajout d'une règle, search_rules() doit la trouver."""
        chroma_store.add_rule(
            rule_id="rule_001",
            rule_text="All ID columns must be unique and not null.",
            rule_type="constraint",
            metadata={"severity": "high", "category": "uniqueness"},
        )
        results = chroma_store.search_rules(query="unique id constraint")
        assert len(results) >= 1
        assert results[0]["id"] == "rule_001"

    def test_n_results_capped_at_collection_size(self, chroma_store):
        """n_results > collection.count() ne doit pas planter."""
        chroma_store.add_rule(
            rule_id="rule_only_one",
            rule_text="Salary must be positive.",
            rule_type="constraint",
            metadata={"severity": "medium", "category": "validity"},
        )
        # On demande 10 résultats mais il n'y en a qu'1 → ne doit pas crasher
        results = chroma_store.search_rules(query="salary positive", n_results=10)
        assert len(results) == 1


class TestChromaStoreNoneMetadata:
    """
    Bug 2 — log_decision() ne doit plus stocker was_correct=None,
    ce qui causait un crash ChromaDB (NoneType interdit en métadonnée).
    """

    @pytest.fixture(autouse=True)
    def chroma_store(self, tmp_path):
        from src.memory import chroma_store as cs_module
        cs_module.ChromaStore._instance = None

        with patch("src.memory.chroma_store.settings") as mock_settings:
            mock_settings.chroma_persist_path = tmp_path / "chroma"
            mock_settings.chroma_rules_collection = "test_rules"
            mock_settings.chroma_decisions_collection = "test_decisions"
            mock_settings.chroma_feedback_collection = "test_feedback"

            from src.memory.chroma_store import ChromaStore
            store = ChromaStore(persist_path=tmp_path / "chroma")
            yield store

        cs_module.ChromaStore._instance = None

    def test_log_decision_does_not_crash(self, chroma_store):
        """log_decision() ne doit pas lever d'exception (plus de None en metadata)."""
        decision_id = chroma_store.log_decision(
            decision_id="dec_test_001",
            agent_type="quality",
            action="detect_quality_issues",
            reasoning="Found 3 null values in salary column",
            context={"session_id": "sess_001"},
            confidence=0.85,
        )
        assert decision_id == "dec_test_001"

    def test_log_decision_was_correct_absent_from_metadata(self, chroma_store):
        """was_correct ne doit pas être présent dans les métadonnées initiales."""
        chroma_store.log_decision(
            decision_id="dec_test_002",
            agent_type="profiler",
            action="profile_dataset",
            reasoning="Profiled 100 rows",
            context={"dataset_id": "ds_001"},
            confidence=0.95,
        )
        stored = chroma_store.decisions_collection.get(
            ids=["dec_test_002"],
            include=["metadatas"],
        )
        meta = stored["metadatas"][0]
        assert "was_correct" not in meta, (
            "was_correct ne doit pas être stocké à la création — "
            "il sera ajouté seulement après feedback utilisateur"
        )

    def test_feedback_updates_was_correct(self, chroma_store):
        """Un feedback utilisateur doit mettre à jour was_correct sur la décision."""
        chroma_store.log_decision(
            decision_id="dec_test_003",
            agent_type="corrector",
            action="propose_corrections",
            reasoning="Proposed median imputation",
            context={"session_id": "sess_002"},
            confidence=0.8,
        )
        chroma_store.add_feedback(
            feedback_id="fb_001",
            target_id="dec_test_003",
            target_type="decision",
            is_correct=True,
            comments="Median imputation was the right choice",
        )
        stored = chroma_store.decisions_collection.get(
            ids=["dec_test_003"],
            include=["metadatas"],
        )
        meta = stored["metadatas"][0]
        assert meta.get("was_correct") is True

    def test_get_decision_accuracy_without_feedback(self, chroma_store):
        """Sans feedback, accuracy doit être None (pas de dénominateur)."""
        chroma_store.log_decision(
            decision_id="dec_acc_001",
            agent_type="quality",
            action="detect_quality_issues",
            reasoning="Found anomalies",
            context={},
            confidence=0.7,
        )
        stats = chroma_store.get_decision_accuracy(agent_type="quality")
        assert stats["total_decisions"] == 1
        assert stats["unknown"] == 1
        assert stats["accuracy"] is None  # Pas encore de feedback
