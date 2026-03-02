"""
Tests pour la parallélisation du QualityAgent (execute_async).

Vérifie que :
- execute_async détecte les mêmes types de problèmes qu'execute (sync)
- execute_async se termine sans erreur sur un DataFrame propre
- execute_async supporte detect_anomalies=False
"""

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame avec des problèmes de qualité détectables (≥10 lignes)."""
    return pd.DataFrame({
        "customer_id": [1, 2, 2, 4, 5, 6, 7, 8, 9, 10],   # doublon sur 2
        "name": ["Alice", "Bob", None, "Diana", "", "Frank", "Grace", None, "Ivan", "Julia"],
        "age": [25, 30, 35, 200, 28, 32, 45, 29, 31, -1],
        "salary": [50000, -5000, 55000, 70000, None, 65000, 80000, None, 45000, 75000],
    })


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """DataFrame sans problèmes."""
    return pd.DataFrame({
        "id": range(1, 11),
        "name": [f"User{i}" for i in range(1, 11)],
        "value": [float(i * 10) for i in range(1, 11)],
    })


@pytest.fixture
def mock_chroma():
    """Mock ChromaDB pour éviter la connexion réelle."""
    mock = MagicMock()
    mock.search_rules.return_value = []
    return mock


def _make_agent(mock_chroma):
    """Construit un QualityAgent avec ChromaDB mocké."""
    with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma):
        with patch("src.agents.base.get_decision_logger") as mock_logger:
            mock_logger.return_value = MagicMock(
                log=MagicMock(return_value=None),
                get_historical_accuracy=MagicMock(return_value=0.8),
                find_similar=MagicMock(return_value=[]),
            )
            from src.agents.quality import QualityAgent
            return QualityAgent()


class TestExecuteAsync:
    """Vérifications de comportement de execute_async."""

    def test_async_detects_issues(self, sample_df, mock_chroma):
        """execute_async doit détecter au moins 1 problème sur le DataFrame sale."""
        agent = _make_agent(mock_chroma)
        context = AgentContext(session_id="async_test", dataset_id="ds_test")

        result = asyncio.run(
            agent.execute_async(context, sample_df, detect_anomalies=False)
        )

        assert len(result.issues) > 0

    def test_async_same_issue_types_as_sync(self, sample_df, mock_chroma):
        """Les types d'issues détectés doivent être identiques en sync et async."""
        agent = _make_agent(mock_chroma)

        ctx_sync = AgentContext(session_id="sync_test", dataset_id="ds_sync")
        ctx_async = AgentContext(session_id="async_test", dataset_id="ds_async")

        sync_result = agent.execute(ctx_sync, sample_df, detect_anomalies=False)
        async_result = asyncio.run(
            agent.execute_async(ctx_async, sample_df, detect_anomalies=False)
        )

        sync_types = {i.issue_type for i in sync_result.issues}
        async_types = {i.issue_type for i in async_result.issues}
        assert sync_types == async_types

    def test_async_no_issues_on_clean_data(self, clean_df, mock_chroma):
        """execute_async sur données propres ne doit pas détecter d'issues nulls."""
        agent = _make_agent(mock_chroma)
        context = AgentContext(session_id="clean_test", dataset_id="ds_clean")

        result = asyncio.run(
            agent.execute_async(context, clean_df, detect_anomalies=False)
        )

        # Pas de valeurs manquantes ni de violations de contraintes
        from src.core.models import IssueType
        null_issues = [i for i in result.issues if i.issue_type == IssueType.MISSING_VALUES]
        assert len(null_issues) == 0
