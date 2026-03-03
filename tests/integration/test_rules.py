"""
Tests d'intégration pour GET/POST/DELETE /rules (F20 — v0.6).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.limiter import limiter
from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    limiter._storage.reset()
    yield


def _auth_patch():
    return patch("src.api.auth.settings", **{"auth_enabled": False})


def _mock_chroma():
    """ChromaStore mock qui simule les opérations rules."""
    store = MagicMock()
    store.get_all_rules.return_value = []
    store.add_rule.return_value = "rule_test_abc"
    store.deactivate_rule.return_value = None
    return store


class TestListRules:

    def test_list_rules_returns_200(self):
        """GET /rules → 200."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.get("/rules")

        assert resp.status_code == 200

    def test_list_rules_response_structure(self):
        """La réponse contient status, count et rules."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.get("/rules")

        data = resp.json()
        assert "status" in data
        assert "count" in data
        assert "rules" in data
        assert data["status"] == "success"
        assert isinstance(data["rules"], list)

    def test_list_rules_empty_when_no_rules(self):
        """Aucune règle → count=0, rules=[]."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.get("/rules")

        data = resp.json()
        assert data["count"] == 0
        assert data["rules"] == []

    def test_list_rules_with_type_filter(self):
        """GET /rules?rule_type=constraint appelle get_all_rules avec rule_type."""
        mock_store = _mock_chroma()
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=mock_store):
            mock_auth.auth_enabled = False
            resp = client.get("/rules?rule_type=constraint")

        assert resp.status_code == 200
        mock_store.get_all_rules.assert_called_once_with(rule_type="constraint")

    def test_list_rules_returns_rule_fields(self):
        """Chaque règle contient rule_id, text, rule_type, severity, category, active."""
        mock_store = _mock_chroma()
        mock_store.get_all_rules.return_value = [
            {
                "id": "rule_001",
                "text": "Age must be positive",
                "metadata": {"rule_type": "constraint", "severity": "high", "category": "validation"},
                "rule_type": "constraint",
                "severity": "high",
                "category": "validation",
                "active": True,
            }
        ]

        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=mock_store):
            mock_auth.auth_enabled = False
            resp = client.get("/rules")

        data = resp.json()
        assert data["count"] == 1
        rule = data["rules"][0]
        assert "rule_id" in rule
        assert "text" in rule
        assert "rule_type" in rule
        assert "severity" in rule
        assert "category" in rule
        assert "active" in rule


class TestCreateRule:

    def test_create_rule_returns_201(self):
        """POST /rules → 201."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.post("/rules", json={
                "rule_text": "Age must be between 0 and 150",
                "rule_type": "constraint",
                "severity": "high",
                "category": "validation",
            })

        assert resp.status_code == 201

    def test_create_rule_response_structure(self):
        """La réponse contient status et rule."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.post("/rules", json={
                "rule_text": "Email must be valid",
                "rule_type": "format",
                "severity": "medium",
                "category": "email",
            })

        data = resp.json()
        assert "status" in data
        assert "rule" in data
        assert data["status"] == "created"

    def test_create_rule_has_rule_id(self):
        """La règle créée a un rule_id non vide."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.post("/rules", json={
                "rule_text": "Salary must be positive",
                "rule_type": "constraint",
                "severity": "low",
                "category": "finance",
            })

        rule = resp.json()["rule"]
        assert "rule_id" in rule
        assert rule["rule_id"].startswith("rule_api_")

    def test_create_rule_persists_text(self):
        """Le texte de la règle est bien retourné dans la réponse."""
        rule_text = "Column 'id' must be unique"
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.post("/rules", json={
                "rule_text": rule_text,
                "rule_type": "constraint",
                "severity": "critical",
                "category": "integrity",
            })

        assert resp.json()["rule"]["text"] == rule_text

    def test_create_rule_calls_add_rule(self):
        """add_rule() est appelé avec les bons arguments."""
        mock_store = _mock_chroma()
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=mock_store):
            mock_auth.auth_enabled = False
            resp = client.post("/rules", json={
                "rule_text": "Test rule for validation purposes",
                "rule_type": "validation",
                "severity": "medium",
                "category": "test",
            })
            assert resp.status_code == 201
            mock_store.add_rule.assert_called_once()
            call_kwargs = mock_store.add_rule.call_args
            assert call_kwargs.kwargs["rule_text"] == "Test rule for validation purposes"
            assert call_kwargs.kwargs["rule_type"] == "validation"

    def test_create_rule_requires_rule_text(self):
        """POST sans rule_text → 422."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.post("/rules", json={
                "rule_type": "constraint",
                "severity": "high",
                "category": "test",
            })

        assert resp.status_code == 422


class TestDeleteRule:

    def test_delete_rule_returns_200(self):
        """DELETE /rules/{rule_id} → 200."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.delete("/rules/rule_test_001")

        assert resp.status_code == 200

    def test_delete_rule_response_structure(self):
        """La réponse contient status et rule_id."""
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=_mock_chroma()):
            mock_auth.auth_enabled = False
            resp = client.delete("/rules/rule_test_001")

        data = resp.json()
        assert "status" in data
        assert "rule_id" in data
        assert data["status"] == "deactivated"
        assert data["rule_id"] == "rule_test_001"

    def test_delete_rule_calls_deactivate(self):
        """deactivate_rule() est appelé avec le bon rule_id."""
        mock_store = _mock_chroma()
        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=mock_store):
            mock_auth.auth_enabled = False
            client.delete("/rules/rule_xyz_789")

        mock_store.deactivate_rule.assert_called_once_with("rule_xyz_789")

    def test_delete_rule_not_found_returns_404(self):
        """Règle inexistante → 404."""
        mock_store = _mock_chroma()
        mock_store.deactivate_rule.side_effect = Exception("Rule not found")

        with patch("src.api.auth.settings") as mock_auth, \
             patch("src.api.routes.rules.get_chroma_store", return_value=mock_store):
            mock_auth.auth_enabled = False
            resp = client.delete("/rules/rule_does_not_exist")

        assert resp.status_code == 404
