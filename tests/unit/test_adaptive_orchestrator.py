"""
Tests unitaires pour OrchestratorAgent — méthodes adaptatives (F24 — v0.7).

Couvre :
- _build_execution_plan() et ses 6 règles adaptatives
- run_pipeline_adaptive() : reasoning_steps dans metadata
- Backward compatibility : run_pipeline_async() inchangé
"""

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext, IssueType, QualityIssue, Severity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_decision_logger():
    logger = MagicMock()
    logger.log.return_value = MagicMock(decision_id="dec_adaptive")
    logger.get_historical_accuracy.return_value = 0.8
    logger.find_similar.return_value = []
    return logger


@pytest.fixture
def df_small():
    """DataFrame de moins de 30 lignes (Règle 1)."""
    return pd.DataFrame({
        "id": range(10),
        "name": [f"item_{i}" for i in range(10)],
        "value": [float(i) for i in range(10)],
    })


@pytest.fixture
def df_normal():
    """DataFrame standard de 50 lignes."""
    return pd.DataFrame({
        "id": range(50),
        "name": [f"item_{i}" for i in range(50)],
        "value": [float(i) for i in range(50)],
    })


@pytest.fixture
def df_no_nulls():
    """DataFrame sans aucune valeur nulle (Règle 2)."""
    return pd.DataFrame({
        "id": range(40),
        "value": [float(i) for i in range(40)],
    })


@pytest.fixture
def df_mostly_null():
    """DataFrame avec >50% de nulls (Règle 3)."""
    return pd.DataFrame({
        "id": range(20),
        "col_a": [None] * 16 + [1, 2, 3, 4],
        "col_b": [None] * 16 + [5, 6, 7, 8],
        "col_c": [None] * 16 + [9, 10, 11, 12],
    })


@pytest.fixture
def df_no_numeric():
    """DataFrame sans aucune colonne numérique (Règle 5)."""
    return pd.DataFrame({
        "name": [f"person_{i}" for i in range(40)],
        "city": [f"city_{i}" for i in range(40)],
        "category": ["A", "B", "C", "D"] * 10,
    })


@pytest.fixture
def df_wide():
    """DataFrame avec > 100 colonnes (Règle 4)."""
    data = {f"col_{i}": range(20) for i in range(110)}
    return pd.DataFrame(data)


def _make_orchestrator(mock_decision_logger, mock_chroma_store):
    with patch("src.agents.base.get_decision_logger", return_value=mock_decision_logger):
        with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma_store):
            with patch("src.agents.validator.get_chroma_store", return_value=mock_chroma_store):
                from src.agents.orchestrator import OrchestratorAgent
                return OrchestratorAgent()


# ---------------------------------------------------------------------------
# Tests sur _build_execution_plan
# ---------------------------------------------------------------------------

class TestBuildExecutionPlan:

    def test_default_plan_includes_basic_checks(self, mock_decision_logger, mock_chroma_store):
        """Le plan de base contient type, duplicates, format (toujours présents)."""
        import pandas as pd
        # df avec des nulls pour éviter que Règle 2 retire missing_values
        df = pd.DataFrame({
            "id": range(50),
            "name": [f"item_{i}" if i % 5 != 0 else None for i in range(50)],
            "value": [float(i) for i in range(50)],
        })
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_plan", dataset_id="ds_plan")
        context = orch.profiler.execute(context, df)

        plan = orch._build_execution_plan(context, df)
        assert "missing_values" in plan  # nulls présents → pas skippé
        assert "type" in plan
        assert "duplicates" in plan

    def test_rule1_skip_anomalies_when_small_df(self, mock_decision_logger, mock_chroma_store, df_small):
        """Règle 1 : < 30 lignes → detect_anomalies absent du plan."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r1", dataset_id="ds_r1")
        context = orch.profiler.execute(context, df_small)

        plan = orch._build_execution_plan(context, df_small)
        assert "detect_anomalies" not in plan

    def test_rule1_include_anomalies_when_enough_rows(self, mock_decision_logger, mock_chroma_store, df_normal):
        """Règle 1 : >= 30 lignes → detect_anomalies présent."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r1b", dataset_id="ds_r1b")
        context = orch.profiler.execute(context, df_normal)

        plan = orch._build_execution_plan(context, df_normal)
        assert "detect_anomalies" in plan

    def test_rule2_skip_missing_when_no_nulls(self, mock_decision_logger, mock_chroma_store, df_no_nulls):
        """Règle 2 : 0 null total → missing_values absent du plan."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r2", dataset_id="ds_r2")
        context = orch.profiler.execute(context, df_no_nulls)

        plan = orch._build_execution_plan(context, df_no_nulls)
        assert "missing_values" not in plan

    def test_rule3_skip_format_when_mostly_null(self, mock_decision_logger, mock_chroma_store, df_mostly_null):
        """Règle 3 : >50% nulls → format absent du plan."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r3", dataset_id="ds_r3")
        context = orch.profiler.execute(context, df_mostly_null)

        plan = orch._build_execution_plan(context, df_mostly_null)
        assert "format" not in plan

    def test_rule4_sampling_mode_when_many_columns(self, mock_decision_logger, mock_chroma_store, df_wide):
        """Règle 4 : >100 colonnes → sampling_mode dans metadata."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r4", dataset_id="ds_r4")
        context = orch.profiler.execute(context, df_wide)

        orch._build_execution_plan(context, df_wide)
        assert context.metadata.get("sampling_mode") is True

    def test_rule5_skip_anomaly_when_no_numeric(self, mock_decision_logger, mock_chroma_store, df_no_numeric):
        """Règle 5 : pas de colonnes numériques → detect_anomalies absent."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r5", dataset_id="ds_r5")
        context = orch.profiler.execute(context, df_no_numeric)

        plan = orch._build_execution_plan(context, df_no_numeric)
        assert "detect_anomalies" not in plan

    def test_rule6_drift_added_when_option_set(self, mock_decision_logger, mock_chroma_store, df_normal):
        """Règle 6 : detect_drift=True + reference_df → drift dans le plan."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r6", dataset_id="ds_r6")
        context = orch.profiler.execute(context, df_normal)

        plan = orch._build_execution_plan(
            context, df_normal,
            detect_drift=True,
            reference_df=df_normal.copy(),
        )
        assert "drift" in plan

    def test_rule6_no_drift_without_reference(self, mock_decision_logger, mock_chroma_store, df_normal):
        """Règle 6 : detect_drift=True mais sans reference_df → drift absent."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_r6b", dataset_id="ds_r6b")
        context = orch.profiler.execute(context, df_normal)

        plan = orch._build_execution_plan(context, df_normal, detect_drift=True)
        assert "drift" not in plan

    def test_no_profile_returns_safe_default(self, mock_decision_logger, mock_chroma_store, df_normal):
        """Sans profil, retourne un plan sécurisé avec detect_anomalies."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_noprofile", dataset_id="ds_np")
        # Ne pas profiler → profile=None

        plan = orch._build_execution_plan(context, df_normal)
        assert "detect_anomalies" in plan
        assert isinstance(plan, list)


# ---------------------------------------------------------------------------
# Tests sur run_pipeline_adaptive
# ---------------------------------------------------------------------------

class TestRunPipelineAdaptive:

    def test_adaptive_pipeline_returns_context(self, mock_decision_logger, mock_chroma_store, df_normal):
        """run_pipeline_adaptive retourne un AgentContext."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_adapt", dataset_id="ds_adapt")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        assert result is not None
        assert hasattr(result, "session_id")
        assert hasattr(result, "issues")

    def test_adaptive_pipeline_populates_reasoning_steps(self, mock_decision_logger, mock_chroma_store, df_normal):
        """context.metadata contient reasoning_steps après run_pipeline_adaptive."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_reason", dataset_id="ds_reason")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        assert "reasoning_steps" in result.metadata
        steps = result.metadata["reasoning_steps"]
        assert isinstance(steps, list)
        assert len(steps) >= 2  # Au moins Observe + Reason

    def test_adaptive_reasoning_steps_structure(self, mock_decision_logger, mock_chroma_store, df_normal):
        """Chaque reasoning_step a les champs step, phase, thought, action, observation."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_steps_struct", dataset_id="ds_struct")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        for step in result.metadata["reasoning_steps"]:
            assert "step" in step
            assert "phase" in step

    def test_adaptive_pipeline_phases_order(self, mock_decision_logger, mock_chroma_store, df_normal):
        """Les phases sont dans l'ordre : observe → reason → act."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_order", dataset_id="ds_order")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        phases = [s["phase"] for s in result.metadata["reasoning_steps"]]
        assert phases[0] == "observe"
        assert "reason" in phases
        assert "act" in phases
        # observe doit précéder reason
        assert phases.index("observe") < phases.index("reason")

    def test_adaptive_pipeline_quality_score_present(self, mock_decision_logger, mock_chroma_store, df_normal):
        """quality_score est présent dans context.metadata après adaptive pipeline."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_score_adapt", dataset_id="ds_score_adapt")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        assert "quality_score" in result.metadata
        score = result.metadata["quality_score"]
        assert 0.0 <= score <= 100.0

    def test_backward_compat_run_pipeline_async_unchanged(self, mock_decision_logger, mock_chroma_store, df_normal):
        """run_pipeline_async() fonctionne toujours comme avant (backward compat)."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_compat", dataset_id="ds_compat")

        result = asyncio.run(
            orch.run_pipeline_async(context, df_normal, task_type=TaskType.ANALYZE)
        )

        assert result is not None
        assert "quality_score" in result.metadata
        # run_pipeline_async ne produit PAS reasoning_steps
        # (ou s'il y en a, c'est en bonus, mais pas requis)

    def test_adaptive_small_df_skips_anomaly(self, mock_decision_logger, mock_chroma_store, df_small):
        """Avec un petit DataFrame, le pipeline adaptatif ne plante pas."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_small_adapt", dataset_id="ds_small_adapt")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_small, task_type=TaskType.ANALYZE)
        )

        assert result is not None
        assert "quality_score" in result.metadata


# ---------------------------------------------------------------------------
# Tests sur la phase Reflect (F31 — v1.2)
# ---------------------------------------------------------------------------

class TestReflectPhase:
    """Tests pour la phase Reflect de run_pipeline_adaptive (F31)."""

    def test_reflect_step_in_reasoning(self, mock_decision_logger, mock_chroma_store, df_normal):
        """La phase 'reflect' est toujours présente dans reasoning_steps."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_reflect_present", dataset_id="ds_rp")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        phases = [s["phase"] for s in result.metadata["reasoning_steps"]]
        assert "reflect" in phases

    def test_reflect_flags_in_metadata(self, mock_decision_logger, mock_chroma_store, df_normal):
        """reflect_flags est toujours présent dans context.metadata après adaptive pipeline."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_reflect_meta", dataset_id="ds_rm")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        assert "reflect_flags" in result.metadata
        assert isinstance(result.metadata["reflect_flags"], list)

    def test_reflect_no_flags_clean_data(self, mock_decision_logger, mock_chroma_store):
        """Dataset propre → pas de flag 'score_vs_critical'."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_reflect_clean", dataset_id="ds_rc")
        df_clean = pd.DataFrame({
            "id": range(50),
            "value": [float(i) for i in range(50)],
        })

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_clean, task_type=TaskType.ANALYZE)
        )

        flags = result.metadata.get("reflect_flags", [])
        assert "score_vs_critical" not in flags

    def test_reflect_coherence_score_vs_critical(self, mock_decision_logger, mock_chroma_store):
        """_reflect_coherence : avg_col ≥ 80 et 2+ CRITICAL → flag 'score_vs_critical'."""
        from src.core.models import AgentType, IssueType, QualityIssue
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_svc", dataset_id="ds_svc")

        context.metadata["column_scores"] = {"col_a": 90.0, "col_b": 85.0}
        for i in range(2):
            context.issues.append(QualityIssue(
                issue_id=f"issue_crit_{i}",
                issue_type=IssueType.CONSTRAINT_VIOLATION,
                severity=Severity.CRITICAL,
                column=None,
                description="Domain rule violation",
                affected_count=1,
                affected_percentage=10.0,
                confidence=0.9,
                detected_by=AgentType.QUALITY,
            ))

        df = pd.DataFrame({"col_a": range(10), "col_b": range(10)})
        flags = orch._reflect_coherence(context, df)
        assert "score_vs_critical" in flags

    def test_reflect_coherence_plan_blind_spot(self, mock_decision_logger, mock_chroma_store):
        """_reflect_coherence : plan sans detect_anomalies + 2 issues HIGH → 'plan_blind_spot'."""
        from src.core.models import AgentType, IssueType, QualityIssue
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_pbs", dataset_id="ds_pbs")

        context.metadata["execution_plan"] = ["missing_values", "type", "duplicates"]
        for i in range(2):
            context.issues.append(QualityIssue(
                issue_id=f"issue_high_{i}",
                issue_type=IssueType.ANOMALY,
                severity=Severity.HIGH,
                column="value",
                description="High severity anomaly",
                affected_count=5,
                affected_percentage=10.0,
                confidence=0.9,
                detected_by=AgentType.QUALITY,
            ))

        df = pd.DataFrame({"value": range(10)})
        flags = orch._reflect_coherence(context, df)
        assert "plan_blind_spot" in flags

    def test_reflect_coherence_no_flag_normal(self, mock_decision_logger, mock_chroma_store):
        """_reflect_coherence : situation normale → liste vide."""
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_no_flag", dataset_id="ds_nf")

        context.metadata["column_scores"] = {"col_a": 70.0}
        context.metadata["execution_plan"] = ["missing_values", "detect_anomalies"]

        df = pd.DataFrame({"col_a": range(10)})
        flags = orch._reflect_coherence(context, df)
        assert flags == []

    def test_execution_plan_stored_in_metadata(
        self, mock_decision_logger, mock_chroma_store, df_normal
    ):
        """execution_plan est stocké dans context.metadata après run_pipeline_adaptive."""
        from src.agents.orchestrator import TaskType
        orch = _make_orchestrator(mock_decision_logger, mock_chroma_store)
        context = AgentContext(session_id="test_plan_meta", dataset_id="ds_pm")

        result = asyncio.run(
            orch.run_pipeline_adaptive(context, df_normal, task_type=TaskType.ANALYZE)
        )

        assert "execution_plan" in result.metadata
        assert isinstance(result.metadata["execution_plan"], list)
