"""
Tests d'intégration pour GET /analyze/{session_id}/comparison (F19 — v0.6).
"""

import io
from unittest.mock import patch

import pandas as pd
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


def _csv_bytes(n_rows: int = 20) -> bytes:
    df = pd.DataFrame({
        "id": range(n_rows),
        "value": [float(i) for i in range(n_rows)],
        "name": [f"item_{i}" for i in range(n_rows)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _upload_file_and_get_session_id() -> str:
    """Uploade un fichier CSV et retourne le session_id."""
    with patch("src.api.auth.settings") as mock_auth:
        mock_auth.auth_enabled = False
        resp = client.post(
            "/upload",
            files=[("file", ("data.csv", _csv_bytes(), "text/csv"))],
        )
    assert resp.status_code == 200
    return resp.json()["session_id"]


class TestComparison:

    def test_comparison_returns_200_with_valid_session(self):
        """GET /analyze/{id}/comparison → 200 avec un session valide."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get(f"/analyze/{session_id}/comparison")

        assert resp.status_code == 200

    def test_comparison_response_has_required_fields(self):
        """La réponse contient tous les champs attendus du ComparisonResponse."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get(f"/analyze/{session_id}/comparison")

        assert resp.status_code == 200
        data = resp.json()

        assert "session_id" in data
        assert "score_before" in data
        assert "score_after" in data
        assert "delta" in data
        assert "issues_removed" in data
        assert "issues_remaining" in data
        assert "columns_improved" in data

    def test_comparison_session_id_matches(self):
        """Le session_id retourné correspond à celui envoyé."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get(f"/analyze/{session_id}/comparison")

        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

    def test_comparison_scores_are_numeric(self):
        """score_before, score_after et delta sont des nombres flottants."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get(f"/analyze/{session_id}/comparison")

        data = resp.json()
        assert isinstance(data["score_before"], (int, float))
        assert isinstance(data["score_after"], (int, float))
        assert isinstance(data["delta"], (int, float))

    def test_comparison_delta_coherent(self):
        """delta == score_after - score_before (à epsilon près)."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get(f"/analyze/{session_id}/comparison")

        data = resp.json()
        assert abs(data["delta"] - (data["score_after"] - data["score_before"])) < 0.01

    def test_comparison_issues_lists_are_lists(self):
        """issues_removed et issues_remaining sont des listes."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get(f"/analyze/{session_id}/comparison")

        data = resp.json()
        assert isinstance(data["issues_removed"], list)
        assert isinstance(data["issues_remaining"], list)
        assert isinstance(data["columns_improved"], list)

    def test_comparison_unknown_session_returns_404(self):
        """Un session_id inexistant → 404."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/analyze/session_nonexistent_abc123/comparison")

        assert resp.status_code == 404

    def test_comparison_is_stateless(self):
        """Appeler comparison deux fois retourne le même résultat (pas de side effects)."""
        session_id = _upload_file_and_get_session_id()

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp1 = client.get(f"/analyze/{session_id}/comparison")
            resp2 = client.get(f"/analyze/{session_id}/comparison")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["score_before"] == resp2.json()["score_before"]
        assert resp1.json()["score_after"] == resp2.json()["score_after"]
