"""
Tests unitaires pour SemanticProfilerAgent (F27) et
_validate_semantic_types() dans QualityAgent (F28) — v0.8.

Couvre :
- enrich_async() : désactivé, populé, structure, batch, limite max_columns
- Fallback silencieux sur erreur API et timeout
- _validate_semantic_types() : monetary_amount, percentage, age, confidence basse
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext, IssueType


# ---------------------------------------------------------------------------
# Helpers — mock Anthropic response
# ---------------------------------------------------------------------------

def _make_tool_use_block(col_name: str, semantic_type: str, confidence: float = 0.9):
    """Crée un faux ToolUseBlock (structure minimale attendue par le parser)."""
    block = SimpleNamespace(
        type="tool_use",
        name="classify_column",
        input={
            "column_name": col_name,
            "semantic_type": semantic_type,
            "confidence": confidence,
            "language": "fr",
            "pattern": None,
            "notes": None,
        },
    )
    return block


def _make_anthropic_response(*blocks):
    """Crée un faux objet response Anthropic avec des tool_use blocks."""
    response = MagicMock()
    response.content = list(blocks)
    return response


# ---------------------------------------------------------------------------
# Fixture — DataFrame avec colonnes variées
# ---------------------------------------------------------------------------

@pytest.fixture
def df_mixed():
    """DataFrame avec email, age et montant pour tester F27 + F28."""
    return pd.DataFrame({
        "email_client": [
            "alice@test.com", "bob@test.com", "charlie@test.com",
            "diana@test.com", "eve@test.com",
        ],
        "age_client": [25, 30, 35, 28, 32],
        "ca_mensuel": [12500.0, 8900.5, -200.0, 15000.0, 7800.0],
        "note_satisfaction": [4.5, 3.0, 5.0, 2.5, 4.0],
        "categorie": ["A", "B", "A", "C", "B"],
    })


@pytest.fixture
def df_percentage():
    """DataFrame avec une colonne pourcentage hors-plage."""
    return pd.DataFrame({
        "taux_completion": [80.0, 95.0, 150.0, 70.0, 105.0],  # 150 et 105 hors [0,100]
    })


@pytest.fixture
def df_age_outlier():
    """DataFrame avec une colonne age contenant des valeurs aberrantes."""
    return pd.DataFrame({
        "age": [25, 30, 200, 28, -5],  # 200 et -5 hors [0,150]
    })


# ---------------------------------------------------------------------------
# Tests — SemanticProfilerAgent.enrich_async()
# ---------------------------------------------------------------------------

class TestSemanticProfilerEnrich:

    def test_returns_context_unchanged_when_disabled(self, df_mixed):
        """ENABLE_LLM_CHECKS=False → semantic_types absent du context."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_disabled", dataset_id="ds_disabled")

        with patch("src.agents.semantic_profiler.settings") as mock_settings:
            mock_settings.enable_llm_checks = False
            result = asyncio.run(agent.enrich_async(context, df_mixed))

        assert "semantic_types" not in result.metadata

    def test_populates_semantic_types_in_metadata(self, df_mixed):
        """enrich_async() stocke semantic_types dans context.metadata."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        response = _make_anthropic_response(
            _make_tool_use_block("email_client", "email", 0.97),
            _make_tool_use_block("age_client", "age", 0.95),
        )

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_pop", dataset_id="ds_pop")

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed))

        assert "semantic_types" in result.metadata
        assert isinstance(result.metadata["semantic_types"], dict)

    def test_semantic_type_has_required_keys(self, df_mixed):
        """Chaque entrée de semantic_types a semantic_type et confidence."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        response = _make_anthropic_response(
            _make_tool_use_block("email_client", "email", 0.97),
        )

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_keys", dataset_id="ds_keys")

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed))

        sem = result.metadata["semantic_types"]
        assert "email_client" in sem
        entry = sem["email_client"]
        assert "semantic_type" in entry
        assert "confidence" in entry

    def test_batch_classifies_multiple_columns(self, df_mixed):
        """N colonnes retournées par le LLM → N entrées dans semantic_types."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        response = _make_anthropic_response(
            _make_tool_use_block("email_client", "email", 0.97),
            _make_tool_use_block("age_client", "age", 0.95),
            _make_tool_use_block("ca_mensuel", "monetary_amount", 0.92),
        )

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_batch", dataset_id="ds_batch")

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed))

        sem = result.metadata["semantic_types"]
        assert len(sem) == 3

    def test_max_columns_limit_respected(self, df_mixed):
        """max_columns=2 → le message batch ne traite que 2 colonnes max."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        response = _make_anthropic_response(
            _make_tool_use_block("email_client", "email", 0.97),
            _make_tool_use_block("age_client", "age", 0.95),
        )

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_maxcols", dataset_id="ds_maxcols")

        call_args_captured = {}

        async def fake_create(**kwargs):
            call_args_captured.update(kwargs)
            return response

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = fake_create
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed, max_columns=2))

        # Le message ne doit mentionner que les 2 premières colonnes
        user_message = call_args_captured["messages"][0]["content"]
        assert "email_client" in user_message
        assert "age_client" in user_message
        # Les colonnes au-delà de max_columns ne doivent pas être dans le message
        assert "ca_mensuel" not in user_message

    def test_fallback_on_api_error(self, df_mixed):
        """Exception API → context inchangé, pas de crash."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_err", dataset_id="ds_err")

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(side_effect=Exception("API error"))
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed))

        # Pas de crash, context intact
        assert result is not None
        assert "semantic_types" not in result.metadata

    def test_fallback_on_timeout(self, df_mixed):
        """asyncio.TimeoutError → context inchangé, pas de crash."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_timeout", dataset_id="ds_timeout")

        async def slow_create(**kwargs):
            raise asyncio.TimeoutError()

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = slow_create
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed))

        assert result is not None
        assert "semantic_types" not in result.metadata

    def test_unknown_column_ignored_gracefully(self, df_mixed):
        """tool_use pour une colonne inexistante → ignoré, pas d'erreur."""
        from src.agents.semantic_profiler import SemanticProfilerAgent

        response = _make_anthropic_response(
            _make_tool_use_block("email_client", "email", 0.97),
            _make_tool_use_block("colonne_fantome", "free_text", 0.5),  # n'existe pas dans df
        )

        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_unknown", dataset_id="ds_unknown")

        with patch("src.agents.semantic_profiler.settings") as mock_settings, \
             patch("anthropic.AsyncAnthropic") as MockClient:
            mock_settings.enable_llm_checks = True
            mock_settings.llm_check_model = "claude-haiku-4-5-20251001"
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(return_value=response)
            MockClient.return_value = mock_instance

            result = asyncio.run(agent.enrich_async(context, df_mixed))

        sem = result.metadata["semantic_types"]
        assert "email_client" in sem
        assert "colonne_fantome" not in sem  # ignoré car pas dans df


# ---------------------------------------------------------------------------
# Tests — QualityAgent._validate_semantic_types()  (F28)
# ---------------------------------------------------------------------------

def _make_quality_agent(mock_chroma_store):
    """Crée un QualityAgent avec ChromaStore mocké."""
    with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma_store):
        from src.agents.quality import QualityAgent
        return QualityAgent()


class TestSemanticQualityValidation:

    def test_monetary_negative_creates_issue(self, mock_chroma_store):
        """semantic_type=monetary_amount avec valeur < 0 → issue ANOMALY MEDIUM."""
        agent = _make_quality_agent(mock_chroma_store)
        df = pd.DataFrame({"ca_mensuel": [12500.0, -200.0, 8900.5]})
        context = AgentContext(session_id="test_money", dataset_id="ds_money")
        context.metadata["semantic_types"] = {
            "ca_mensuel": {"semantic_type": "monetary_amount", "confidence": 0.92},
        }

        issues = agent._validate_semantic_types(df, context)

        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.ANOMALY
        assert issues[0].column == "ca_mensuel"
        assert issues[0].details["semantic_type"] == "monetary_amount"

    def test_percentage_out_of_range_creates_issue(self, mock_chroma_store, df_percentage):
        """semantic_type=percentage avec valeur hors [0,100] → issue ANOMALY MEDIUM."""
        agent = _make_quality_agent(mock_chroma_store)
        context = AgentContext(session_id="test_pct", dataset_id="ds_pct")
        context.metadata["semantic_types"] = {
            "taux_completion": {"semantic_type": "percentage", "confidence": 0.9},
        }

        issues = agent._validate_semantic_types(df_percentage, context)

        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.ANOMALY
        assert issues[0].column == "taux_completion"
        assert issues[0].details["out_of_range_count"] == 2

    def test_age_out_of_range_creates_issue(self, mock_chroma_store, df_age_outlier):
        """semantic_type=age avec valeur hors [0,150] → issue ANOMALY MEDIUM."""
        agent = _make_quality_agent(mock_chroma_store)
        context = AgentContext(session_id="test_age", dataset_id="ds_age")
        context.metadata["semantic_types"] = {
            "age": {"semantic_type": "age", "confidence": 0.95},
        }

        issues = agent._validate_semantic_types(df_age_outlier, context)

        assert len(issues) == 1
        assert issues[0].column == "age"
        assert issues[0].details["out_of_range_count"] == 2

    def test_low_confidence_semantic_type_skipped(self, mock_chroma_store):
        """confidence < 0.7 → pas d'issue créée."""
        agent = _make_quality_agent(mock_chroma_store)
        df = pd.DataFrame({"montant": [100.0, -50.0, 200.0]})
        context = AgentContext(session_id="test_lowconf", dataset_id="ds_lowconf")
        context.metadata["semantic_types"] = {
            "montant": {"semantic_type": "monetary_amount", "confidence": 0.5},
        }

        issues = agent._validate_semantic_types(df, context)

        assert len(issues) == 0

    def test_no_semantic_types_returns_empty(self, mock_chroma_store):
        """Pas de semantic_types dans metadata → retourne liste vide."""
        agent = _make_quality_agent(mock_chroma_store)
        df = pd.DataFrame({"col": [1, 2, 3]})
        context = AgentContext(session_id="test_empty", dataset_id="ds_empty")
        # Pas de semantic_types dans metadata

        issues = agent._validate_semantic_types(df, context)

        assert issues == []
