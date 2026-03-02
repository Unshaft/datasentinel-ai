"""
Tests d'intégration pour les endpoints /webhooks.

Couvre :
- POST /webhooks → 201 avec webhook_id
- GET /webhooks → liste les webhooks enregistrés
- DELETE /webhooks/{id} → 204
- DELETE /webhooks/{id_inconnu} → 404
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.core.webhook_manager as wm


@pytest.fixture(autouse=True)
def _clean():
    """Nettoie le store global avant/après chaque test."""
    wm._webhooks.clear()
    yield
    wm._webhooks.clear()


@pytest.fixture
def app_client():
    with patch("src.memory.chroma_store.get_chroma_store", return_value=MagicMock()), \
         patch("src.core.config.settings") as mock_settings:

        mock_settings.auth_enabled = False
        mock_settings.cors_origins = ["*"]
        mock_settings.api_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_expire_minutes = 60

        from src.api.main import app
        yield TestClient(app)


class TestWebhookCRUD:
    """CRUD sur les webhooks."""

    def test_register_webhook_returns_201(self, app_client):
        response = app_client.post(
            "/webhooks",
            json={"url": "http://example.com/hook", "events": ["analysis.completed"]},
        )
        assert response.status_code == 201

    def test_register_returns_webhook_id(self, app_client):
        response = app_client.post(
            "/webhooks",
            json={"url": "http://example.com/hook", "events": ["analysis.completed"]},
        )
        body = response.json()
        assert "webhook_id" in body
        assert body["webhook_id"].startswith("wh_")

    def test_list_webhooks_empty(self, app_client):
        response = app_client.get("/webhooks")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_webhooks_after_register(self, app_client):
        app_client.post(
            "/webhooks",
            json={"url": "http://example.com/hook", "events": ["analysis.completed"]},
        )
        response = app_client.get("/webhooks")
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_delete_webhook(self, app_client):
        reg = app_client.post(
            "/webhooks",
            json={"url": "http://example.com/hook", "events": ["analysis.completed"]},
        )
        wh_id = reg.json()["webhook_id"]

        delete_resp = app_client.delete(f"/webhooks/{wh_id}")
        assert delete_resp.status_code == 204

        list_resp = app_client.get("/webhooks")
        assert len(list_resp.json()) == 0

    def test_delete_nonexistent_returns_404(self, app_client):
        response = app_client.delete("/webhooks/wh_nonexistent")
        assert response.status_code == 404
