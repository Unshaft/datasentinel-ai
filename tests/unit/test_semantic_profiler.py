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

    def test_returns_heuristic_types_when_llm_disabled(self, df_mixed):
        """ENABLE_LLM_CHECKS=False → semantic_types populé par l'heuristique (F27v2)."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        agent = SemanticProfilerAgent()
        context = AgentContext(session_id="test_disabled", dataset_id="ds_disabled")

        with patch("src.agents.semantic_profiler.settings") as mock_settings:
            mock_settings.enable_llm_checks = False
            result = asyncio.run(agent.enrich_async(context, df_mixed))

        # v2 : l'heuristique tourne toujours, même sans LLM
        assert "semantic_types" in result.metadata
        assert all(
            v.get("method") == "heuristic"
            for v in result.metadata["semantic_types"].values()
        )

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
        """Toutes les colonnes sont dans semantic_types après merge heuristique+LLM.

        v2 : l'heuristique pré-classifie toutes les colonnes. Le LLM enrichit
        certaines d'entre elles. Le merge retourne autant d'entrées que de colonnes.
        """
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
        # Toutes les colonnes du DataFrame sont présentes (heuristique + LLM merge)
        assert len(sem) == len(df_mixed.columns)
        # Les colonnes enrichies par le LLM (confidence > heuristique) ont method=llm
        assert sem["email_client"]["method"] == "llm"
        assert sem["email_client"]["semantic_type"] == "email"

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
        """Exception API → heuristique conservé, pas de crash (F27v2)."""
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

        # v2 : heuristique conservé même si LLM échoue
        assert result is not None
        assert "semantic_types" in result.metadata
        # Toutes les entrées restent en mode heuristique
        assert all(
            v.get("method") == "heuristic"
            for v in result.metadata["semantic_types"].values()
        )

    def test_fallback_on_timeout(self, df_mixed):
        """asyncio.TimeoutError → heuristique conservé, pas de crash (F27v2)."""
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

        # v2 : heuristique conservé même si LLM timeout
        assert result is not None
        assert "semantic_types" in result.metadata

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


# ---------------------------------------------------------------------------
# Tests — Classificateur heuristique (F27 v2)
# ---------------------------------------------------------------------------


class TestHeuristicClassifier:
    """Vérifie le classificateur heuristique sans LLM."""

    def _agent(self):
        from src.agents.semantic_profiler import SemanticProfilerAgent
        return SemanticProfilerAgent()

    def test_email_detected_by_regex(self):
        """Colonne avec des emails valides → semantic_type=email."""
        agent = self._agent()
        df = pd.DataFrame({"email_client": [
            "alice@test.com", "bob@example.com", "carol@test.fr",
            "diana@mail.com", "eve@test.org",
        ]})
        result = agent._heuristic_classify(df, 5)
        assert "email_client" in result
        assert result["email_client"]["semantic_type"] == "email"
        assert result["email_client"]["method"] == "heuristic"

    def test_age_clean_column_confidence_075(self):
        """Colonne 'age' avec toutes valeurs en [0,150] → confidence 0.75 (déclenche F28)."""
        agent = self._agent()
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 28, 32, 27, 38, 42]})
        result = agent._heuristic_classify(df, 5)
        assert result["age"]["semantic_type"] == "age"
        assert result["age"]["confidence"] == 0.75

    def test_age_dirty_column_below_f28_threshold(self):
        """Colonne 'age' avec 80 % des valeurs dans la plage → confidence < 0.75."""
        agent = self._agent()
        # 8/10 = 80 % en [0,150] — ne doit PAS déclencher les validators F28
        df = pd.DataFrame({"age": [25, 30, 35, 200, 28, 32, 45, 29, 31, -1]})
        result = agent._heuristic_classify(df, 5)
        assert result["age"]["semantic_type"] == "age"
        assert result["age"]["confidence"] < 0.75

    def test_salary_detected_as_monetary_amount(self):
        """Colonne 'salary' → monetary_amount."""
        agent = self._agent()
        df = pd.DataFrame({"salary": [50000, 60000, 55000, 70000, 65000]})
        result = agent._heuristic_classify(df, 5)
        assert result["salary"]["semantic_type"] == "monetary_amount"

    def test_percentage_clean_column(self):
        """Colonne 'taux' avec valeurs en [0,100] → percentage avec confidence 0.75."""
        agent = self._agent()
        df = pd.DataFrame({"taux_completion": [80.0, 95.0, 70.0, 60.0, 85.0,
                                               75.0, 90.0, 55.0, 65.0, 78.0]})
        result = agent._heuristic_classify(df, 5)
        assert result["taux_completion"]["semantic_type"] == "percentage"
        assert result["taux_completion"]["confidence"] == 0.75

    def test_free_text_fallback(self):
        """Colonne sans signal → free_text.

        Les valeurs ont trop de cardinalité pour être 'category' (>40%) mais ne sont
        pas toutes uniques (évite 'identifier'). Aucun regex ni mot-clé ne correspond.
        """
        agent = self._agent()
        # 3 valeurs uniques sur 5 lignes = 60% cardinalité → pas category (>40%)
        # Non toutes uniques → pas identifier
        df = pd.DataFrame({"random_col": ["abc", "def", "ghi", "abc", "ghi"]})
        result = agent._heuristic_classify(df, 5)
        assert result["random_col"]["semantic_type"] == "free_text"

    def test_identifier_by_id_suffix(self):
        """Colonne finissant par _id → identifier."""
        agent = self._agent()
        df = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5]})
        result = agent._heuristic_classify(df, 5)
        assert result["customer_id"]["semantic_type"] == "identifier"

    def test_category_low_cardinality(self):
        """Colonne avec faible cardinalité → category."""
        agent = self._agent()
        df = pd.DataFrame({"dept": ["IT", "HR", "IT", "Finance", "HR",
                                    "IT", "Finance", "HR", "IT", "Finance"]})
        result = agent._heuristic_classify(df, 5)
        assert result["dept"]["semantic_type"] == "category"

    def test_max_columns_respected(self):
        """max_columns=2 → seules 2 colonnes classifiées."""
        agent = self._agent()
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = agent._heuristic_classify(df, 2)
        assert len(result) == 2

    def test_no_false_positive_stage_for_age(self):
        """'stage' ne doit PAS être classifié comme 'age' (évite substring 'age' dans 'stage')."""
        agent = self._agent()
        df = pd.DataFrame({"stage": ["A", "B", "C", "A", "B"]})
        result = agent._heuristic_classify(df, 5)
        assert result["stage"]["semantic_type"] != "age"

    def test_all_columns_have_method_heuristic(self):
        """Toutes les entrées heuristiques portent method='heuristic'."""
        agent = self._agent()
        df = pd.DataFrame({
            "email": ["a@b.com", "c@d.com"],
            "age": [25, 30],
            "unknown": ["x", "y"],
        })
        result = agent._heuristic_classify(df, 10)
        assert all(v["method"] == "heuristic" for v in result.values())


# ---------------------------------------------------------------------------
# Tests — enrich_sync (F27 v2)
# ---------------------------------------------------------------------------


class TestEnrichSync:
    """Vérifie enrich_sync() — toujours disponible sans LLM."""

    def test_always_populates_semantic_types(self):
        """enrich_sync popule semantic_types même quand LLM désactivé."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        agent = SemanticProfilerAgent()
        df = pd.DataFrame({"email": ["alice@test.com", "bob@test.com", "carol@test.fr"]})
        context = AgentContext(session_id="s1", dataset_id="d1")

        result = agent.enrich_sync(context, df)

        assert "semantic_types" in result.metadata
        assert "email" in result.metadata["semantic_types"]

    def test_returns_context_unchanged_if_empty_df(self):
        """DataFrame vide → semantic_types vide, pas de crash."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        agent = SemanticProfilerAgent()
        df = pd.DataFrame()
        context = AgentContext(session_id="s2", dataset_id="d2")

        result = agent.enrich_sync(context, df)

        assert "semantic_types" in result.metadata
        assert result.metadata["semantic_types"] == {}

    def test_does_not_call_llm(self):
        """enrich_sync ne doit jamais appeler anthropic."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        agent = SemanticProfilerAgent()
        df = pd.DataFrame({"age": [25, 30, 35]})
        context = AgentContext(session_id="s3", dataset_id="d3")

        with patch("anthropic.AsyncAnthropic") as MockClient:
            agent.enrich_sync(context, df)
            MockClient.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — _merge_results (F27 v2)
# ---------------------------------------------------------------------------


class TestMergeResults:
    """Vérifie la fusion heuristique + LLM."""

    def test_llm_wins_when_higher_confidence(self):
        """LLM avec confidence plus haute remplace l'heuristique."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        heuristic = {"col1": {"semantic_type": "free_text", "confidence": 0.62, "method": "heuristic"}}
        llm = {"col1": {"semantic_type": "email", "confidence": 0.95}}
        merged = SemanticProfilerAgent._merge_results(heuristic, llm)
        assert merged["col1"]["semantic_type"] == "email"
        assert merged["col1"]["method"] == "llm"

    def test_heuristic_kept_when_llm_lower(self):
        """Heuristique conservé si LLM moins confiant."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        heuristic = {"col1": {"semantic_type": "age", "confidence": 0.75, "method": "heuristic"}}
        llm = {"col1": {"semantic_type": "free_text", "confidence": 0.60}}
        merged = SemanticProfilerAgent._merge_results(heuristic, llm)
        assert merged["col1"]["semantic_type"] == "age"

    def test_llm_adds_missing_columns(self):
        """LLM peut ajouter des colonnes non détectées par l'heuristique."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        heuristic = {"col1": {"semantic_type": "age", "confidence": 0.75, "method": "heuristic"}}
        llm = {"col2": {"semantic_type": "email", "confidence": 0.95}}
        merged = SemanticProfilerAgent._merge_results(heuristic, llm)
        assert "col1" in merged
        assert "col2" in merged

    def test_heuristic_columns_preserved_if_llm_misses_them(self):
        """Colonnes non retournées par LLM → classification heuristique conservée."""
        from src.agents.semantic_profiler import SemanticProfilerAgent
        heuristic = {
            "col1": {"semantic_type": "age", "confidence": 0.75, "method": "heuristic"},
            "col2": {"semantic_type": "email", "confidence": 0.65, "method": "heuristic"},
        }
        llm = {"col1": {"semantic_type": "age", "confidence": 0.90}}
        merged = SemanticProfilerAgent._merge_results(heuristic, llm)
        assert "col2" in merged
        assert merged["col2"]["method"] == "heuristic"
