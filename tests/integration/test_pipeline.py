"""
Tests d'intégration — Pipeline complet sans appel LLM.

Ces tests exercent le pipeline entier (Profiler → Quality → Corrector → Validator)
sur des DataFrames représentatifs. Le LLM (Claude) n'est PAS appelé car les méthodes
execute() des agents sont 100 % Python/Pandas/scikit-learn.

Les seules dépendances mockées sont :
- DecisionLogger → pour éviter les appels ChromaDB
- ChromaStore (get_chroma_store) → pour isoler les tests de la DB réelle

Ce design valide que l'intégration *mécanique* du pipeline est correcte :
chaînage des agents, accumulation dans AgentContext, calcul du score qualité, etc.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext, IssueType, Severity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_decision_logger():
    logger = MagicMock()
    logger.log.return_value = MagicMock(decision_id="dec_integration")
    logger.get_historical_accuracy.return_value = 0.8
    logger.find_similar.return_value = []
    return logger


@pytest.fixture
def df_with_known_issues():
    """
    Dataset avec des problèmes connus et quantifiables :
    - customer_id : 1 doublon (id=2 présent deux fois)
    - name : 1 valeur nulle
    - salary : 1 valeur nulle
    - age : 1 anomalie manifeste (200)
    """
    return pd.DataFrame({
        "customer_id": [1, 2, 2, 4, 5, 6, 7, 8, 9, 10],
        "name":        ["Alice", "Bob", None, "Diana", "Eve",
                        "Frank", "Grace", "Henry", "Ivan", "Julia"],
        "age":         [25, 30, 35, 200, 28, 32, 45, 29, 31, 38],
        "salary":      [50000, 60000, 55000, 70000, None,
                        65000, 80000, None, 45000, 75000],
        "department":  ["IT", "HR", "IT", "Finance", "HR",
                        "IT", "Finance", "HR", "IT", "Finance"],
    })


@pytest.fixture
def df_clean():
    return pd.DataFrame({
        "customer_id": list(range(1, 21)),
        "age":   [25 + i for i in range(20)],
        "salary": [50000 + i * 1000 for i in range(20)],
    })


def _patched_orchestrator(mock_decision_logger, mock_chroma_store):
    with patch("src.agents.base.get_decision_logger", return_value=mock_decision_logger):
        with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma_store):
            with patch("src.agents.validator.get_chroma_store", return_value=mock_chroma_store):
                from src.agents.orchestrator import OrchestratorAgent
                return OrchestratorAgent()


# ---------------------------------------------------------------------------
# Suite d'intégration
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Pipeline Profiler → Quality → Corrector → Validator."""

    def test_analyze_pipeline_runs_without_error(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Le pipeline d'analyse doit s'exécuter sans exception."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(df_with_known_issues)

        assert result is not None
        assert "session_id" in result
        assert "quality_score" in result
        assert "issues" in result

    def test_profiler_captures_dataset_dimensions(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Le profil doit refléter les dimensions réelles du dataset."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(df_with_known_issues)

        profile = result["profile"]
        assert profile["rows"] == 10
        assert profile["columns"] == 5

    def test_quality_detects_null_values(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Des valeurs nulles dans name et salary doivent générer des MISSING_VALUES issues."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(df_with_known_issues)

        null_issues = [
            i for i in result["issues"]
            if i["type"] == IssueType.MISSING_VALUES.value
        ]
        affected_columns = {i["column"] for i in null_issues}
        assert "name" in affected_columns or "salary" in affected_columns

    def test_quality_detects_duplicate_id(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Le doublon sur customer_id doit générer une CONSTRAINT_VIOLATION."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(df_with_known_issues)

        constraint_issues = [
            i for i in result["issues"]
            if i["type"] == IssueType.CONSTRAINT_VIOLATION.value
        ]
        assert len(constraint_issues) >= 1
        assert any(i["column"] == "customer_id" for i in constraint_issues)

    def test_recommend_generates_proposals_for_null_issues(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Des issues MISSING_VALUES doivent produire des propositions d'imputation."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(df_with_known_issues)

        assert "proposals" in result
        assert len(result["proposals"]) > 0
        # Au moins une proposition d'imputation
        imputation_types = {"impute_mean", "impute_median", "impute_mode"}
        proposal_types = {p["type"] for p in result["proposals"]}
        assert len(imputation_types & proposal_types) > 0

    def test_full_pipeline_validates_proposals(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Le pipeline complet doit inclure des validations."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.full_analysis(df_with_known_issues)

        assert "validations" in result
        assert len(result["validations"]) > 0

    def test_estimated_improvement_present_and_consistent(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """
        estimated_improvement doit être présent (bug 1 corrigé)
        et cohérent avec quality_score.
        """
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.recommend(df_with_known_issues)

        assert "estimated_improvement" in result
        assert 0 <= result["estimated_improvement"] <= 100 - result["quality_score"] + 0.01

    def test_session_id_propagated(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Un session_id fourni doit être préservé dans la réponse."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(df_with_known_issues, session_id="my_session_123")

        assert result["session_id"] == "my_session_123"

    def test_clean_data_no_issues(
        self, mock_decision_logger, mock_chroma_store, df_clean
    ):
        """Un dataset propre ne doit générer aucun problème."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result = orchestrator.analyze(df_clean)

        assert result["quality_score"] >= 85
        # Pas de MISSING_VALUES sur un dataset propre
        null_issues = [
            i for i in result["issues"]
            if i["type"] == IssueType.MISSING_VALUES.value
        ]
        assert len(null_issues) == 0


class TestAgentContextAccumulation:
    """Vérifie que l'AgentContext accumule bien les résultats à chaque étape."""

    def test_context_has_profile_after_profiling(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        from src.agents.orchestrator import TaskType

        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="ctx_test", dataset_id="ds_ctx")
        context = orchestrator.run_pipeline(
            context, df_with_known_issues, TaskType.PROFILE_ONLY
        )

        assert context.profile is not None
        assert context.profile.row_count == 10
        assert context.profile.column_count == 5

    def test_context_accumulates_issues_after_quality(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        from src.agents.orchestrator import TaskType

        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="ctx_quality", dataset_id="ds_q")
        context = orchestrator.run_pipeline(
            context, df_with_known_issues, TaskType.ANALYZE
        )

        assert len(context.issues) > 0

    def test_context_has_proposals_after_recommend(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        from src.agents.orchestrator import TaskType

        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="ctx_rec", dataset_id="ds_rec")
        context = orchestrator.run_pipeline(
            context, df_with_known_issues, TaskType.RECOMMEND
        )

        assert len(context.proposals) > 0
        # Chaque proposition doit référencer un issue existant
        issue_ids = {i.issue_id for i in context.issues}
        for proposal in context.proposals:
            assert proposal.issue_id in issue_ids

    def test_proposals_reference_valid_issues(
        self, mock_decision_logger, mock_chroma_store, df_with_known_issues
    ):
        """Intégrité référentielle : chaque proposal.issue_id doit exister dans context.issues."""
        orchestrator = _patched_orchestrator(mock_decision_logger, mock_chroma_store)
        result_context = AgentContext(session_id="ref_test", dataset_id="ds_ref")

        from src.agents.orchestrator import TaskType
        result_context = orchestrator.run_pipeline(
            result_context, df_with_known_issues, TaskType.RECOMMEND
        )

        issue_ids = {i.issue_id for i in result_context.issues}
        for proposal in result_context.proposals:
            assert proposal.issue_id in issue_ids, (
                f"Proposition {proposal.proposal_id} référence l'issue inconnue "
                f"{proposal.issue_id}"
            )
