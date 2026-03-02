"""
Tests unitaires pour webhook_manager.

Couvre :
- add_webhook / remove_webhook / get_webhooks
- fire_webhooks envoie les notifications aux abonnés
- fire_webhooks ignore les webhooks d'autres événements
- Échec HTTP → logué, pas d'exception levée
- Pas d'abonné → fire_webhooks est no-op
- Persistance JSON (v0.4) : add/remove → fichier écrit, _load_from_disk recharge
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.core.webhook_manager as wm


@pytest.fixture(autouse=True)
def _clean_webhooks():
    """Nettoie le store global avant chaque test."""
    wm._webhooks.clear()
    yield
    wm._webhooks.clear()


class TestRegistration:
    """Enregistrement et suppression de webhooks."""

    def test_add_returns_webhook_id(self):
        wh_id = wm.add_webhook("http://example.com/hook", ["analysis.completed"])
        assert wh_id.startswith("wh_")

    def test_get_webhooks_after_add(self):
        wm.add_webhook("http://example.com/hook", ["analysis.completed"])
        hooks = wm.get_webhooks()
        assert len(hooks) == 1
        assert hooks[0]["url"] == "http://example.com/hook"

    def test_remove_existing_webhook(self):
        wh_id = wm.add_webhook("http://example.com/hook", ["analysis.completed"])
        removed = wm.remove_webhook(wh_id)
        assert removed is True
        assert len(wm.get_webhooks()) == 0

    def test_remove_nonexistent_returns_false(self):
        assert wm.remove_webhook("wh_nonexistent") is False

    def test_multiple_webhooks(self):
        wm.add_webhook("http://a.com/hook", ["analysis.completed"])
        wm.add_webhook("http://b.com/hook", ["analysis.completed"])
        assert len(wm.get_webhooks()) == 2


class TestFireWebhooks:
    """Déclenchement des notifications."""

    def test_fire_sends_to_matching_subscriber(self):
        wm.add_webhook("http://example.com/hook", ["analysis.completed"])

        mock_resp = MagicMock(status_code=200)
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            asyncio.run(
                wm.fire_webhooks("analysis.completed", {"session_id": "s1", "quality_score": 90})
            )
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
            assert payload["event"] == "analysis.completed"
            assert payload["session_id"] == "s1"

    def test_fire_ignores_other_events(self):
        wm.add_webhook("http://example.com/hook", ["other.event"])

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            asyncio.run(wm.fire_webhooks("analysis.completed", {}))
            mock_post.assert_not_called()

    def test_fire_no_subscribers_is_noop(self):
        """Sans abonnés, fire_webhooks ne doit pas planter."""
        asyncio.run(wm.fire_webhooks("analysis.completed", {}))  # No exception

    def test_fire_continues_on_http_error(self):
        """Un webhook qui échoue ne doit pas lever d'exception."""
        wm.add_webhook("http://bad.host/hook", ["analysis.completed"])

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            # Ne doit pas lever
            asyncio.run(wm.fire_webhooks("analysis.completed", {"session_id": "s1"}))


class TestJsonPersistence:
    """Persistance JSON des webhooks (v0.4)."""

    def test_add_webhook_writes_json_file(self):
        """add_webhook doit appeler _save_to_disk et écrire un fichier JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "webhooks.json"
            with patch.object(wm, "_PERSIST_PATH", persist_path):
                wm.add_webhook("http://a.com/hook", ["analysis.completed"])
                assert persist_path.exists()
                data = json.loads(persist_path.read_text(encoding="utf-8"))
                assert len(data) == 1

    def test_remove_webhook_updates_json_file(self):
        """remove_webhook doit mettre à jour le fichier JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "webhooks.json"
            with patch.object(wm, "_PERSIST_PATH", persist_path):
                wh_id = wm.add_webhook("http://a.com/hook", ["analysis.completed"])
                wm.remove_webhook(wh_id)
                data = json.loads(persist_path.read_text(encoding="utf-8"))
                assert len(data) == 0

    def test_load_from_disk_restores_webhooks(self):
        """_load_from_disk doit charger les webhooks depuis le fichier JSON."""
        existing = {
            "wh_abc123": {
                "webhook_id": "wh_abc123",
                "url": "http://restored.com/hook",
                "events": ["analysis.completed"],
                "description": "restored",
                "created_at": "2026-01-01T00:00:00+00:00",
                "active": True,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "webhooks.json"
            persist_path.write_text(json.dumps(existing), encoding="utf-8")

            with patch.object(wm, "_PERSIST_PATH", persist_path):
                wm._load_from_disk()
                hooks = wm.get_webhooks()
                urls = [h["url"] for h in hooks]
                assert "http://restored.com/hook" in urls

    def test_load_from_disk_missing_file_is_noop(self):
        """_load_from_disk ne doit pas planter si le fichier n'existe pas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "nonexistent.json"
            with patch.object(wm, "_PERSIST_PATH", persist_path):
                wm._load_from_disk()  # No exception

    def test_load_from_disk_corrupt_file_is_noop(self):
        """_load_from_disk ne doit pas planter si le fichier est corrompu."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "webhooks.json"
            persist_path.write_text("THIS IS NOT JSON", encoding="utf-8")
            with patch.object(wm, "_PERSIST_PATH", persist_path):
                wm._load_from_disk()  # No exception
