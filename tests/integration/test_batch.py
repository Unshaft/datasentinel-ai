"""
Tests d'intégration pour POST /batch (analyse en lot — v0.5).
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.limiter import limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Vide le compteur slowapi avant chaque test pour éviter les 429."""
    limiter._storage.reset()
    yield


# =============================================================================
# Helpers
# =============================================================================

def _csv_bytes(n_rows: int = 20) -> bytes:
    df = pd.DataFrame({
        "id": range(n_rows),
        "value": [float(i) for i in range(n_rows)],
        "label": [f"item_{i}" for i in range(n_rows)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _auth_patch():
    """Contexte manager qui désactive l'auth."""
    return patch("src.api.auth.settings", **{"auth_enabled": False})


# =============================================================================
# Tests
# =============================================================================

class TestBatchAnalyze:

    def test_batch_single_file_200(self):
        """Un fichier CSV valide → HTTP 200, 1 résultat."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(
                "/batch",
                files=[("files", ("data.csv", _csv_bytes(), "text/csv"))],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert data["failed"] == 0
        assert len(data["results"]) == 1

    def test_batch_result_has_required_fields(self):
        """Chaque résultat contient filename, status, quality_score, issues_count, session_id."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(
                "/batch",
                files=[("files", ("test.csv", _csv_bytes(), "text/csv"))],
            )

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        assert "filename" in item
        assert "status" in item
        assert "quality_score" in item
        assert "issues_count" in item
        assert "session_id" in item
        assert item["filename"] == "test.csv"

    def test_batch_multiple_files(self):
        """Plusieurs fichiers → autant de résultats."""
        files = [
            ("files", (f"file_{i}.csv", _csv_bytes(15 + i), "text/csv"))
            for i in range(3)
        ]
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post("/batch", files=files)

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_batch_empty_files_list(self):
        """Aucun fichier → HTTP 400."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post("/batch", files=[])

        # FastAPI retourne 422 si le paramètre requis est vide / absent
        assert resp.status_code in (400, 422)

    def test_batch_invalid_extension(self):
        """Fichier avec extension invalide → résultat en erreur, pas d'exception globale."""
        files = [
            ("files", ("bad.txt", b"hello world", "text/plain")),
            ("files", ("good.csv", _csv_bytes(), "text/csv")),
        ]
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post("/batch", files=files)

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        # Le fichier .txt échoue, le .csv réussit
        statuses = {r["filename"]: r["status"] for r in data["results"]}
        assert statuses["bad.txt"] == "error"
        assert statuses["good.csv"] == "success"

    def test_batch_session_ids_are_unique(self):
        """Chaque fichier reçoit un session_id distinct."""
        files = [
            ("files", ("a.csv", _csv_bytes(), "text/csv")),
            ("files", ("b.csv", _csv_bytes(), "text/csv")),
        ]
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post("/batch", files=files)

        assert resp.status_code == 200
        results = resp.json()["results"]
        session_ids = [r["session_id"] for r in results if r.get("session_id")]
        assert len(session_ids) == len(set(session_ids)), "Les session_ids doivent être uniques"

    def test_batch_quality_score_in_range(self):
        """Le score de qualité est entre 0 et 100."""
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(
                "/batch",
                files=[("files", ("data.csv", _csv_bytes(), "text/csv"))],
            )

        assert resp.status_code == 200
        item = resp.json()["results"][0]
        if item["quality_score"] is not None:
            assert 0 <= item["quality_score"] <= 100

    def test_batch_succeeded_failed_counts_consistent(self):
        """succeeded + failed == total."""
        files = [
            ("files", ("ok.csv", _csv_bytes(), "text/csv")),
            ("files", ("ko.txt", b"garbage", "text/plain")),
        ]
        with patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post("/batch", files=files)

        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded"] + data["failed"] == data["total"]
