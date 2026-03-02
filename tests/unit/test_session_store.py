"""
Tests unitaires pour SessionStore.

Utilise fakeredis pour simuler Redis sans serveur réel.
Couvre :
- save / load round-trip (contenu préservé)
- Session absente → None
- TTL expiré → None
- Fallback in-memory si Redis indisponible
- delete() supprime la session
"""

import time
from unittest.mock import MagicMock, patch

import fakeredis
import pytest

from src.core.models import AgentContext, IssueType, QualityIssue, Severity
from src.memory.session_store import InMemoryFallback, SessionStore


@pytest.fixture
def fake_redis():
    """Client fakeredis isolé par test (pas de serveur réel)."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def store(fake_redis):
    """SessionStore utilisant fakeredis."""
    s = SessionStore(redis_client=fake_redis)
    s._using_fallback = False
    return s


@pytest.fixture
def simple_context():
    """AgentContext minimal pour les tests."""
    return AgentContext(session_id="sess_001", dataset_id="ds_001")


@pytest.fixture
def context_with_issues():
    """AgentContext avec issues pour tester la sérialisation complète."""
    ctx = AgentContext(session_id="sess_with_issues", dataset_id="ds_issues")
    ctx.issues.append(
        QualityIssue(
            issue_id="issue_001",
            issue_type=IssueType.MISSING_VALUES,
            severity=Severity.HIGH,
            column="salary",
            description="Valeurs manquantes dans salary",
            affected_count=3,
            affected_percentage=10.0,
            confidence=0.95,
            detected_by="quality",
        )
    )
    ctx.metadata["quality_score"] = 75.0
    return ctx


class TestSaveLoad:
    """save() et load() — round-trip de sérialisation."""

    def test_save_and_load_simple_context(self, store, simple_context):
        """Un contexte simple doit être récupéré intact."""
        store.save("sess_001", simple_context)
        loaded = store.load("sess_001")

        assert loaded is not None
        assert loaded.session_id == "sess_001"
        assert loaded.dataset_id == "ds_001"

    def test_save_and_load_preserves_issues(self, store, context_with_issues):
        """Les issues doivent être préservées après sérialisation."""
        store.save("sess_with_issues", context_with_issues)
        loaded = store.load("sess_with_issues")

        assert loaded is not None
        assert len(loaded.issues) == 1
        assert loaded.issues[0].issue_id == "issue_001"
        assert loaded.issues[0].severity == Severity.HIGH

    def test_save_and_load_preserves_metadata(self, store, context_with_issues):
        """Les métadonnées doivent être préservées."""
        store.save("sess_with_issues", context_with_issues)
        loaded = store.load("sess_with_issues")

        assert loaded.metadata["quality_score"] == 75.0

    def test_load_absent_session_returns_none(self, store):
        """Charger une session inexistante doit retourner None."""
        result = store.load("nonexistent_session")
        assert result is None


class TestTTL:
    """Expiration des sessions."""

    def test_session_expires_after_ttl(self, fake_redis, simple_context):
        """Après expiration du TTL, la session doit être introuvable."""
        store = SessionStore(redis_client=fake_redis)
        store.save("expiring_session", simple_context, ttl=1)

        # Simuler l'expiration en supprimant manuellement (fakeredis gère le TTL)
        time.sleep(1.1)
        result = store.load("expiring_session")
        assert result is None


class TestDelete:
    """delete() — suppression de session."""

    def test_delete_removes_session(self, store, simple_context):
        """delete() doit rendre la session introuvable."""
        store.save("sess_to_delete", simple_context)
        store.delete("sess_to_delete")
        assert store.load("sess_to_delete") is None

    def test_delete_nonexistent_does_not_crash(self, store):
        """delete() sur une session inexistante ne doit pas lever d'exception."""
        store.delete("nonexistent")  # Pas d'exception


class TestExists:
    """exists() — vérification d'existence."""

    def test_exists_returns_true_for_saved_session(self, store, simple_context):
        store.save("sess_exists", simple_context)
        assert store.exists("sess_exists") is True

    def test_exists_returns_false_for_absent_session(self, store):
        assert store.exists("absent") is False


class TestInMemoryFallback:
    """Fallback in-memory si Redis est indisponible."""

    def test_fallback_save_and_load(self, simple_context):
        """Le fallback doit sauvegarder et restituer les sessions."""
        # Simuler Redis indisponible
        broken_redis = MagicMock()
        broken_redis.ping.side_effect = ConnectionError("Redis down")

        with patch("src.memory.session_store.settings") as mock_settings:
            mock_settings.redis_url = "redis://nonexistent:9999"
            mock_settings.session_ttl = 3600
            store = SessionStore()

        assert store.using_fallback is True
        store.save("fallback_sess", simple_context)
        loaded = store.load("fallback_sess")
        assert loaded is not None
        assert loaded.session_id == "sess_001"

    def test_in_memory_fallback_set_get(self):
        """InMemoryFallback doit se comporter comme un Redis minimal."""
        fallback = InMemoryFallback()
        fallback.set("key", "value")
        assert fallback.get("key") == "value"

    def test_in_memory_fallback_delete(self):
        fallback = InMemoryFallback()
        fallback.set("key", "value")
        fallback.delete("key")
        assert fallback.get("key") is None

    def test_in_memory_fallback_missing_key_returns_none(self):
        fallback = InMemoryFallback()
        assert fallback.get("absent") is None


class TestSaveLoadDataframe:
    """save_dataframe() et load_dataframe() — round-trip parquet/base64 (v0.5)."""

    import pandas as pd  # noqa: E402 — import au niveau classe pour lisibilité

    def test_save_and_load_dataframe_roundtrip(self, store):
        """Le DataFrame sauvegardé doit être récupéré intact."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        store.save_dataframe("sess_df_01", df)
        loaded = store.load_dataframe("sess_df_01")

        assert loaded is not None
        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 3
        assert loaded["a"].tolist() == [1, 2, 3]

    def test_load_dataframe_absent_returns_none(self, store):
        """Charger un DataFrame inexistant doit retourner None."""
        result = store.load_dataframe("nonexistent_df")
        assert result is None

    def test_save_dataframe_does_not_affect_context(self, store, simple_context):
        """save_dataframe() ne doit pas écraser le contexte de session."""
        import pandas as pd

        store.save("sess_both", simple_context)
        df = pd.DataFrame({"x": [10, 20]})
        store.save_dataframe("sess_both", df)

        ctx = store.load("sess_both")
        assert ctx is not None
        assert ctx.session_id == "sess_001"

    def test_save_dataframe_with_nulls(self, store):
        """Les DataFrames avec NaN sont correctement sérialisés/désérialisés."""
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
        store.save_dataframe("sess_df_null", df)
        loaded = store.load_dataframe("sess_df_null")

        assert loaded is not None
        assert loaded["a"].isna().sum() == 1

    def test_delete_removes_dataframe_too(self, store):
        """delete() doit supprimer le DataFrame stocké en même temps que la session."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2]})
        store.save("sess_del_df", simple_context := AgentContext(session_id="s", dataset_id="d"))
        store.save_dataframe("sess_del_df", df)
        store.delete("sess_del_df")

        assert store.load_dataframe("sess_del_df") is None
