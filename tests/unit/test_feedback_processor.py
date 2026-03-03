"""
Tests unitaires pour FeedbackProcessor (F26 — v0.7).

Couvre :
- Traitement des faux positifs (false positive)
- Traitement des confirmations
- Traitement des corrections personnalisées
- Bornes de confiance [0.5, 0.99]
- get_adjustments()
- get_summary()
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_feedback_processor():
    """Remet le singleton FeedbackProcessor à zéro avant chaque test."""
    from src.core.feedback_processor import FeedbackProcessor
    # Réinitialise l'état interne du singleton
    instance = FeedbackProcessor()
    with instance._data_lock:
        instance._stats = {}
    yield
    with instance._data_lock:
        instance._stats = {}


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.add_rule.return_value = "rule_test"
    return store


def _make_feedback(
    is_correct: bool | None = None,
    target_id: str = "issue_abc",
    target_type: str = "issue",
    user_correction: str | None = None,
    session_id: str = "session_test",
) -> SimpleNamespace:
    return SimpleNamespace(
        is_correct=is_correct,
        target_id=target_id,
        target_type=target_type,
        user_correction=user_correction,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Tests — False Positive
# ---------------------------------------------------------------------------

class TestFalsePositive:

    def test_false_positive_increments_fp_stats(self, mock_store):
        """Un faux positif incrémente le compteur false_positive_stats."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        feedback = _make_feedback(is_correct=False, target_type="issue")
        fp.process(feedback, mock_store)

        assert fp._stats["false_positive_stats"]["issue"] == 1

    def test_false_positive_multiple_increments(self, mock_store):
        """Plusieurs FP s'accumulent dans le compteur."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        for _ in range(3):
            fp.process(_make_feedback(is_correct=False, target_type="proposal"), mock_store)

        assert fp._stats["false_positive_stats"]["proposal"] == 3

    def test_false_positive_writes_exception_rule(self, mock_store):
        """Un faux positif écrit une règle d'exception dans ChromaDB."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(_make_feedback(is_correct=False, target_id="issue_xyz"), mock_store)

        mock_store.add_rule.assert_called_once()
        call_kwargs = mock_store.add_rule.call_args.kwargs
        assert call_kwargs["rule_type"] == "exception"

    def test_false_positive_no_confidence_adjustment_before_threshold(self, mock_store):
        """Moins de 5 FP : pas d'ajustement de confiance."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        for _ in range(5):  # _FP_THRESHOLD = 5 → ajustement à >5
            fp.process(_make_feedback(is_correct=False, target_type="issue"), mock_store)

        adj = fp.get_adjustments()
        # Exactement 5 FP (pas > 5) → pas encore d'ajustement
        assert "issue" not in adj

    def test_false_positive_adjusts_confidence_after_threshold(self, mock_store):
        """Plus de 5 FP → confidence_adjustments décrémenté."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        for _ in range(6):  # 6 > _FP_THRESHOLD(5)
            fp.process(_make_feedback(is_correct=False, target_type="issue"), mock_store)

        adj = fp.get_adjustments()
        assert "issue" in adj
        assert adj["issue"] < 1.0

    def test_false_positive_confidence_floor_respected(self, mock_store):
        """La confiance ne descend jamais en dessous de 0.5."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        # Forcer beaucoup de FP (100) pour saturer la borne basse
        for _ in range(100):
            fp.process(_make_feedback(is_correct=False, target_type="issue"), mock_store)

        adj = fp.get_adjustments()
        assert adj.get("issue", 1.0) >= 0.5


# ---------------------------------------------------------------------------
# Tests — Confirmation
# ---------------------------------------------------------------------------

class TestConfirmation:

    def test_confirmation_increments_counter(self, mock_store):
        """Une confirmation incrémente le compteur confirmations."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(_make_feedback(is_correct=True, target_type="issue"), mock_store)

        assert fp._stats.get("confirmations", 0) >= 1

    def test_confirmation_multiple_increments(self, mock_store):
        """Plusieurs confirmations s'accumulent."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        for _ in range(3):
            fp.process(_make_feedback(is_correct=True, target_type="issue"), mock_store)

        assert fp._stats.get("confirmations", 0) == 3

    def test_confirmation_adjusts_confidence_upward(self, mock_store):
        """Une confirmation ajuste confidence_adjustments (capped à 0.99)."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(_make_feedback(is_correct=True, target_type="proposal"), mock_store)

        adj = fp.get_adjustments()
        assert "proposal" in adj
        # Initial value=1.0, min(0.99, 1.0+0.02)=0.99 (plafond)
        assert adj["proposal"] >= 0.99

    def test_confirmation_confidence_ceiling_respected(self, mock_store):
        """La confiance ne dépasse jamais 0.99."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        # Beaucoup de confirmations pour atteindre le plafond
        for _ in range(100):
            fp.process(_make_feedback(is_correct=True, target_type="decision"), mock_store)

        adj = fp.get_adjustments()
        assert adj.get("decision", 1.0) <= 0.99

    def test_confirmation_does_not_write_rule(self, mock_store):
        """Une confirmation simple ne crée pas de règle ChromaDB."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(_make_feedback(is_correct=True), mock_store)

        # Confirmation seule → pas d'add_rule
        mock_store.add_rule.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — Custom Correction
# ---------------------------------------------------------------------------

class TestCustomCorrection:

    def test_custom_correction_increments_examples(self, mock_store):
        """Une correction personnalisée incrémente examples_added."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(
            _make_feedback(user_correction="La valeur correcte est 25"),
            mock_store,
        )

        assert fp._stats.get("examples_added", 0) >= 1

    def test_custom_correction_writes_example_rule(self, mock_store):
        """Une correction personnalisée écrit une règle example dans ChromaDB."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(
            _make_feedback(user_correction="Age should be between 0 and 120"),
            mock_store,
        )

        calls = mock_store.add_rule.call_args_list
        example_calls = [c for c in calls if c.kwargs.get("rule_type") == "example"]
        assert len(example_calls) >= 1

    def test_custom_correction_combined_with_false_positive(self, mock_store):
        """is_correct=False + user_correction → FP handler ET correction handler."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        fp.process(
            _make_feedback(
                is_correct=False,
                user_correction="La valeur devrait être positive",
            ),
            mock_store,
        )

        # Deux appels add_rule : un pour l'exception FP, un pour la correction
        assert mock_store.add_rule.call_count == 2


# ---------------------------------------------------------------------------
# Tests — get_adjustments & get_summary
# ---------------------------------------------------------------------------

class TestGetAdjustmentsAndSummary:

    def test_get_adjustments_returns_dict(self, mock_store):
        """get_adjustments() retourne un dict."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        result = fp.get_adjustments()
        assert isinstance(result, dict)

    def test_get_adjustments_empty_initially(self, mock_store):
        """get_adjustments() est vide au démarrage."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        assert fp.get_adjustments() == {}

    def test_get_summary_returns_dict_with_keys(self, mock_store):
        """get_summary() retourne un dict avec les clés attendues."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        summary = fp.get_summary()
        assert "false_positives_corrected" in summary
        assert "confirmations" in summary
        assert "examples_added" in summary
        assert "checks_adjusted" in summary

    def test_get_summary_reflects_fp_count(self, mock_store):
        """get_summary().false_positives_corrected reflète le nombre de FP traités."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        for _ in range(3):
            fp.process(_make_feedback(is_correct=False, target_type="issue"), mock_store)

        summary = fp.get_summary()
        assert summary["false_positives_corrected"] == 3

    def test_get_summary_reflects_confirmations(self, mock_store):
        """get_summary().confirmations reflète le nombre de confirmations."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        for _ in range(2):
            fp.process(_make_feedback(is_correct=True, target_type="issue"), mock_store)

        summary = fp.get_summary()
        assert summary["confirmations"] == 2

    def test_process_does_not_raise_on_store_error(self):
        """process() ne lève pas d'exception si ChromaStore échoue."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        broken_store = MagicMock()
        broken_store.add_rule.side_effect = Exception("DB down")

        # Ne doit pas lever d'exception
        fp.process(_make_feedback(is_correct=False), broken_store)

    def test_process_unknown_feedback_type_does_not_crash(self):
        """process() avec des attributs manquants ne lève pas d'exception."""
        from src.core.feedback_processor import get_feedback_processor
        fp = get_feedback_processor()

        store = MagicMock()
        # Objet minimal sans attributs
        fp.process(object(), store)
