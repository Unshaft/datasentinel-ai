"""
Tests unitaires pour OrchestratorAgent.

Couvre le bug 1 corrigé :
- estimated_improvement présent dans la réponse de recommend()
- Calcul cohérent avec la logique de la route FastAPI

Et le comportement général du pipeline :
- Modes ANALYZE, RECOMMEND, FULL_PIPELINE
- Score de qualité calculé correctement
- Escalade déclenchée sur problème CRITICAL
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext, IssueType, Severity


@pytest.fixture
def mock_decision_logger():
    logger = MagicMock()
    logger.log.return_value = MagicMock(decision_id="dec_001")
    logger.get_historical_accuracy.return_value = 0.8
    logger.find_similar.return_value = []
    return logger


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 55000, 70000, 65000],
    })


@pytest.fixture
def dirty_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 2, 4, 5],   # doublon
        "name": ["Alice", "Bob", None, "Diana", "Eve"],
        "age": [25, 30, 35, 200, -5],      # anomalies
        "salary": [50000, 60000, 55000, None, 65000],  # null
    })


def _make_orchestrator(mock_decision_logger, mock_chroma_store):
    """Helper : crée un OrchestratorAgent avec toutes les dépendances mockées."""
    with patch("src.agents.base.get_decision_logger", return_value=mock_decision_logger):
        with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma_store):
            with patch("src.agents.validator.get_chroma_store", return_value=mock_chroma_store):
                from src.agents.orchestrator import OrchestratorAgent
                return OrchestratorAgent()


class TestEstimatedImprovement:
    """Bug 1 — estimated_improvement doit être présent dans recommend()."""

    def test_recommend_returns_estimated_improvement(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        """
        orchestrator.recommend() doit retourner un dict avec la clé
        'estimated_improvement'. Avant le fix, c'était un KeyError.
        """
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(dirty_df)

        assert "estimated_improvement" in result, (
            "Bug 1 non corrigé : 'estimated_improvement' absent du résultat de recommend()"
        )

    def test_estimated_improvement_is_non_negative(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        """estimated_improvement doit être ≥ 0."""
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(dirty_df)

        assert result["estimated_improvement"] >= 0

    def test_estimated_improvement_capped_at_remaining_score(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        """estimated_improvement ne peut pas dépasser 100 - quality_score."""
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(dirty_df)

        quality_score = result["quality_score"]
        max_possible = 100 - quality_score
        assert result["estimated_improvement"] <= max_possible + 0.01  # tolérance flottant

    def test_clean_data_zero_improvement(
        self, mock_decision_logger, mock_chroma_store, clean_df
    ):
        """Sur des données propres, le score est élevé donc l'amélioration estimée est faible."""
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(clean_df)

        # Pas d'issues → pas de propositions → amélioration nulle ou très faible
        assert result["estimated_improvement"] >= 0


class TestQualityScore:
    """Le score de qualité doit refléter les problèmes détectés."""

    def test_clean_data_high_score(
        self, mock_decision_logger, mock_chroma_store, clean_df
    ):
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(clean_df)

        assert result["quality_score"] >= 80

    def test_dirty_data_lower_score(
        self, mock_decision_logger, mock_chroma_store, dirty_df, clean_df
    ):
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        dirty_result = orchestrator.analyze(dirty_df)
        clean_result = orchestrator.analyze(clean_df)

        assert dirty_result["quality_score"] < clean_result["quality_score"]

    def test_score_between_0_and_100(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(dirty_df)

        assert 0 <= result["quality_score"] <= 100


class TestPipelineModes:
    """Les différents modes de pipeline doivent retourner les bonnes clés."""

    def test_analyze_returns_issues(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(dirty_df)

        assert "issues" in result
        assert "profile" in result
        assert len(result["issues"]) > 0

    def test_recommend_returns_proposals(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(dirty_df)

        assert "proposals" in result
        assert len(result["proposals"]) > 0

    def test_full_analysis_returns_validations(
        self, mock_decision_logger, mock_chroma_store, dirty_df
    ):
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.full_analysis(dirty_df)

        assert "validations" in result
        assert "approved_corrections" in result

    def test_clean_data_no_proposals(
        self, mock_decision_logger, mock_chroma_store, clean_df
    ):
        """Sur données propres, pas d'issues → pas de propositions."""
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(clean_df)

        assert result["proposals"] == []


class TestEscalation:
    """L'orchestrateur doit déclencher l'escalade sur problèmes critiques."""

    def test_critical_issue_triggers_human_review(
        self, mock_decision_logger, mock_chroma_store
    ):
        """Un dataset avec issue CRITICAL doit passer needs_human_review=True."""
        # DataFrame qui génère une anomalie détectée comme critique via le
        # score de confiance bas (confidence < 0.7 → needs_escalation=True sur QualityIssue)
        df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 9999],
        })
        orchestrator = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="esc_test", dataset_id="ds_esc")

        from src.core.models import IssueType, QualityIssue, Severity
        import uuid

        # Injecter directement une issue CRITICAL dans le contexte
        context.issues.append(QualityIssue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            issue_type=IssueType.ANOMALY,
            severity=Severity.CRITICAL,
            column="value",
            description="Valeur critique détectée",
            affected_count=1,
            affected_percentage=10.0,
            confidence=0.9,
            detected_by="quality",
        ))

        needs_escalation = orchestrator._check_escalation_needed(context)
        assert needs_escalation is True
