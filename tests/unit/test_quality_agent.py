"""
Tests unitaires pour QualityAgent.

Couvre le bug 4 corrigé :
- _validate_against_rules niveau 1 (heuristique ID) — toujours actif
- _validate_against_rules niveau 2 (ChromaDB RAG) — actif si règles disponibles
- Dégradation gracieuse si ChromaDB est indisponible
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext, IssueType, Severity


@pytest.fixture
def agent_context():
    return AgentContext(session_id="test_session", dataset_id="test_dataset")


@pytest.fixture
def quality_agent(mock_chroma_store):
    """QualityAgent avec ChromaDB mocké (pas de DB réelle)."""
    with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma_store):
        with patch("src.agents.base.get_decision_logger") as mock_logger:
            mock_logger.return_value = MagicMock(
                log=MagicMock(return_value=None),
                get_historical_accuracy=MagicMock(return_value=0.8),
                find_similar=MagicMock(return_value=[]),
            )
            from src.agents.quality import QualityAgent
            return QualityAgent()


class TestValidateAgainstRulesLevel1:
    """Niveau 1 : heuristique d'unicité sur colonnes *_id."""

    def test_detects_duplicate_id(self, quality_agent, agent_context):
        """Une colonne customer_id avec doublons doit générer une CONSTRAINT_VIOLATION."""
        df = pd.DataFrame({
            "customer_id": [1, 2, 2, 4, 5],  # doublon sur 2
            "name": ["A", "B", "C", "D", "E"],
        })
        issues = quality_agent._validate_against_rules(df, agent_context)

        constraint_issues = [
            i for i in issues if i.issue_type == IssueType.CONSTRAINT_VIOLATION
        ]
        assert len(constraint_issues) >= 1
        assert constraint_issues[0].column == "customer_id"
        assert constraint_issues[0].severity == Severity.HIGH
        assert constraint_issues[0].affected_count == 1

    def test_no_issue_on_unique_id(self, quality_agent, agent_context):
        """Une colonne *_id sans doublons ne doit pas générer d'issue."""
        df = pd.DataFrame({
            "order_id": [10, 20, 30, 40, 50],
            "value": [1, 2, 3, 4, 5],
        })
        issues = quality_agent._validate_against_rules(df, agent_context)
        assert all(i.issue_type != IssueType.CONSTRAINT_VIOLATION for i in issues)

    def test_id_column_detected_by_substring(self, quality_agent, agent_context):
        """La détection doit s'appliquer sur 'user_id', 'product_id', 'id', etc."""
        df = pd.DataFrame({
            "product_id": [1, 1, 3],  # doublon
            "name": ["X", "Y", "Z"],
        })
        issues = quality_agent._validate_against_rules(df, agent_context)
        assert any(i.issue_type == IssueType.CONSTRAINT_VIOLATION for i in issues)

    def test_level1_works_without_chromadb(self, agent_context):
        """Niveau 1 doit fonctionner même si ChromaDB est complètement absent."""
        broken_store = MagicMock()
        broken_store.search_rules.side_effect = Exception("ChromaDB unavailable")

        with patch("src.agents.quality.get_chroma_store", return_value=broken_store):
            with patch("src.agents.base.get_decision_logger") as mock_logger:
                mock_logger.return_value = MagicMock(
                    log=MagicMock(return_value=None),
                    get_historical_accuracy=MagicMock(return_value=0.8),
                    find_similar=MagicMock(return_value=[]),
                )
                from src.agents.quality import QualityAgent
                agent = QualityAgent()

        df = pd.DataFrame({"user_id": [1, 1, 3], "val": [10, 20, 30]})
        # Ne doit pas planter, et doit quand même détecter le doublon
        issues = agent._validate_against_rules(df, agent_context)
        assert any(i.issue_type == IssueType.CONSTRAINT_VIOLATION for i in issues)
        # L'erreur ChromaDB doit être loguée dans les métadonnées du contexte
        assert "rules_validation_error" in agent_context.metadata


class TestValidateAgainstRulesLevel2:
    """Niveau 2 : règles sémantiques via ChromaDB RAG."""

    def test_uniqueness_rule_triggers_constraint_issue(self, agent_context):
        """Une règle 'unique' dans ChromaDB doit déclencher une CONSTRAINT_VIOLATION."""
        mock_store = MagicMock()
        mock_store.search_rules.return_value = [
            {
                "id": "rule_unique_email",
                "text": "The email column must be unique across all records.",
                "metadata": {"severity": "high", "category": "uniqueness"},
                "similarity": 0.75,
            }
        ]

        with patch("src.agents.quality.get_chroma_store", return_value=mock_store):
            with patch("src.agents.base.get_decision_logger") as mock_logger:
                mock_logger.return_value = MagicMock(
                    log=MagicMock(return_value=None),
                    get_historical_accuracy=MagicMock(return_value=0.8),
                    find_similar=MagicMock(return_value=[]),
                )
                from src.agents.quality import QualityAgent
                agent = QualityAgent()

        df = pd.DataFrame({
            "email": ["a@test.com", "a@test.com", "c@test.com"],  # doublon
        })
        issues = agent._validate_against_rules(df, agent_context)
        chroma_issues = [
            i for i in issues
            if i.issue_type == IssueType.CONSTRAINT_VIOLATION
            and "rule_unique_email" in str(i.details)
        ]
        assert len(chroma_issues) >= 1
        assert chroma_issues[0].severity == Severity.HIGH

    def test_not_null_rule_triggers_constraint_issue(self, agent_context):
        """Une règle 'not null' dans ChromaDB doit déclencher une CONSTRAINT_VIOLATION."""
        mock_store = MagicMock()
        mock_store.search_rules.return_value = [
            {
                "id": "rule_salary_not_null",
                "text": "Salary is obligatoire and must not be null.",
                "metadata": {"severity": "critical", "category": "completeness"},
                "similarity": 0.8,
            }
        ]

        with patch("src.agents.quality.get_chroma_store", return_value=mock_store):
            with patch("src.agents.base.get_decision_logger") as mock_logger:
                mock_logger.return_value = MagicMock(
                    log=MagicMock(return_value=None),
                    get_historical_accuracy=MagicMock(return_value=0.8),
                    find_similar=MagicMock(return_value=[]),
                )
                from src.agents.quality import QualityAgent
                agent = QualityAgent()

        df = pd.DataFrame({
            "salary": [50000, None, 60000, None, 70000],
        })
        issues = agent._validate_against_rules(df, agent_context)
        null_constraint_issues = [
            i for i in issues
            if i.issue_type == IssueType.CONSTRAINT_VIOLATION
            and "rule_salary_not_null" in str(i.details)
        ]
        assert len(null_constraint_issues) >= 1
        assert null_constraint_issues[0].severity == Severity.CRITICAL
        assert null_constraint_issues[0].affected_count == 2

    def test_low_similarity_rule_is_ignored(self, agent_context):
        """Une règle avec similarité < 0.55 ne doit pas générer d'issue."""
        mock_store = MagicMock()
        mock_store.search_rules.return_value = [
            {
                "id": "rule_irrelevant",
                "text": "All prices must be unique.",
                "metadata": {"severity": "high", "category": "uniqueness"},
                "similarity": 0.40,  # En dessous du seuil
            }
        ]

        with patch("src.agents.quality.get_chroma_store", return_value=mock_store):
            with patch("src.agents.base.get_decision_logger") as mock_logger:
                mock_logger.return_value = MagicMock(
                    log=MagicMock(return_value=None),
                    get_historical_accuracy=MagicMock(return_value=0.8),
                    find_similar=MagicMock(return_value=[]),
                )
                from src.agents.quality import QualityAgent
                agent = QualityAgent()

        df = pd.DataFrame({"price": [10, 10, 30]})
        issues = agent._validate_against_rules(df, agent_context)
        # Aucune issue depuis ChromaDB car similarité trop basse
        chroma_issues = [i for i in issues if "rule_irrelevant" in str(i.details)]
        assert len(chroma_issues) == 0

    def test_no_duplicate_issue_when_level1_and_level2_both_detect(self, agent_context):
        """
        Si niveau 1 (heuristique *_id) et niveau 2 (ChromaDB) détectent tous les deux
        la même violation, on obtient deux issues distinctes (traçabilité différente).
        Vérifie qu'il n'y a pas de suppression silencieuse.
        """
        mock_store = MagicMock()
        mock_store.search_rules.return_value = [
            {
                "id": "rule_id_unique",
                "text": "The user_id must be unique.",
                "metadata": {"severity": "high", "category": "uniqueness"},
                "similarity": 0.9,
            }
        ]

        with patch("src.agents.quality.get_chroma_store", return_value=mock_store):
            with patch("src.agents.base.get_decision_logger") as mock_logger:
                mock_logger.return_value = MagicMock(
                    log=MagicMock(return_value=None),
                    get_historical_accuracy=MagicMock(return_value=0.8),
                    find_similar=MagicMock(return_value=[]),
                )
                from src.agents.quality import QualityAgent
                agent = QualityAgent()

        df = pd.DataFrame({"user_id": [1, 1, 3]})
        issues = agent._validate_against_rules(df, agent_context)
        # Au moins 2 issues : une du niveau 1, une du niveau 2
        assert len(issues) >= 2
