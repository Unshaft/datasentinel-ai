"""
Tests d'intégration pour GET /stats et DELETE /stats (F22 — v0.6).
"""

import io
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.limiter import limiter
from src.api.main import app
from src.core.stats_manager import get_stats_manager

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    limiter._storage.reset()
    yield


@pytest.fixture(autouse=True)
def reset_stats():
    """Remet les stats à zéro avant chaque test."""
    get_stats_manager().reset()
    yield
    get_stats_manager().reset()


def _csv_bytes(n_rows: int = 20) -> bytes:
    df = pd.DataFrame({
        "id": range(n_rows),
        "value": [float(i) for i in range(n_rows)],
        "label": [f"item_{i}" for i in range(n_rows)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


class TestGetStats:

    def test_get_stats_returns_200(self):
        """GET /stats → 200."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        assert resp.status_code == 200

    def test_get_stats_response_structure(self):
        """La réponse contient tous les champs attendus du StatsResponse."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        data = resp.json()
        assert "total_sessions" in data
        assert "avg_quality_score" in data
        assert "top_issue_types" in data
        assert "sessions_by_day" in data
        assert "score_distribution" in data
        assert "updated_at" in data

    def test_initial_stats_total_sessions_zero(self):
        """Les stats fraîches ont total_sessions=0."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        assert resp.json()["total_sessions"] == 0

    def test_initial_stats_types_are_correct(self):
        """Les champs de stats ont les types attendus."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        data = resp.json()
        assert isinstance(data["total_sessions"], int)
        assert isinstance(data["avg_quality_score"], (int, float))
        assert isinstance(data["top_issue_types"], dict)
        assert isinstance(data["sessions_by_day"], dict)
        assert isinstance(data["score_distribution"], dict)

    def test_stats_increments_after_analysis(self):
        """Une analyse POST /upload incrémente total_sessions."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp_before = client.get("/stats")

        sessions_before = resp_before.json()["total_sessions"]

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            client.post(
                "/upload",
                files=[("file", ("data.csv", _csv_bytes(), "text/csv"))],
            )

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp_after = client.get("/stats")

        assert resp_after.json()["total_sessions"] >= sessions_before + 1

    def test_stats_avg_score_between_0_and_100(self):
        """avg_quality_score est entre 0 et 100 après une analyse."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            client.post(
                "/upload",
                files=[("file", ("data.csv", _csv_bytes(), "text/csv"))],
            )

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        avg_score = resp.json()["avg_quality_score"]
        assert 0.0 <= avg_score <= 100.0

    def test_stats_score_distribution_keys(self):
        """score_distribution a des clés de bucket valides."""
        valid_buckets = {"0-20", "20-40", "40-60", "60-80", "80-100"}

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            client.post(
                "/upload",
                files=[("file", ("data.csv", _csv_bytes(), "text/csv"))],
            )

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        buckets = set(resp.json()["score_distribution"].keys())
        # Au moins un bucket valide doit être présent
        assert len(buckets & valid_buckets) >= 1

    def test_stats_sessions_by_day_format(self):
        """sessions_by_day a des clés au format YYYY-MM-DD."""
        import re

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            client.post(
                "/upload",
                files=[("file", ("data.csv", _csv_bytes(), "text/csv"))],
            )

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        day_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for key in resp.json()["sessions_by_day"]:
            assert day_pattern.match(key), f"Clé invalide: {key}"


class TestResetStats:

    def test_reset_stats_returns_200(self):
        """DELETE /stats → 200."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.delete("/stats")

        assert resp.status_code == 200

    def test_reset_stats_response_structure(self):
        """La réponse de DELETE /stats contient status."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.delete("/stats")

        data = resp.json()
        assert "status" in data
        assert data["status"] == "reset"

    def test_reset_stats_clears_data(self):
        """Après reset, total_sessions revient à 0."""
        # Enregistrer une session
        get_stats_manager().record_session(85.0, ["missing_values"])

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            before = client.get("/stats").json()
        assert before["total_sessions"] >= 1

        # Reset
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            client.delete("/stats")

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            after = client.get("/stats").json()

        assert after["total_sessions"] == 0

    def test_multiple_sessions_increment_counter(self):
        """N sessions enregistrées → total_sessions == N."""
        for i in range(3):
            get_stats_manager().record_session(70.0 + i * 5, ["anomaly"])

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        assert resp.json()["total_sessions"] == 3

    def test_top_issue_types_reflects_recorded_issues(self):
        """top_issue_types reflète les types enregistrés."""
        get_stats_manager().record_session(75.0, ["missing_values", "anomaly"])
        get_stats_manager().record_session(80.0, ["missing_values"])

        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.get("/stats")

        top = resp.json()["top_issue_types"]
        assert "missing_values" in top
        assert top["missing_values"] >= 2
