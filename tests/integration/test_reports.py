"""
Tests d'intégration pour les exports de rapport.

Couvre :
- GET /analyze/{session_id}/report.pdf → PDF valide
- GET /analyze/{session_id}/report.xlsx → Excel valide (v0.4)
- Session inexistante → HTTP 404 (PDF + xlsx)
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.models import AgentContext


def _minimal_context(session_id: str = "sess_pdf_test") -> AgentContext:
    ctx = AgentContext(session_id=session_id, dataset_id="ds_pdf")
    ctx.metadata["quality_score"] = 82.5
    ctx.metadata["final_status"] = "completed"
    ctx.metadata["processing_time_ms"] = 150
    ctx.metadata["summary"] = "Bonne qualité (82.5%)"
    ctx.metadata["column_scores"] = {"col_a": 95.0, "col_b": 70.0}
    return ctx


@pytest.fixture
def app_client_with_session():
    """TestClient avec une session pré-chargée dans le store mocké."""
    ctx = _minimal_context("sess_pdf_test")

    mock_store = MagicMock()
    mock_store.load.side_effect = lambda sid: ctx if sid == "sess_pdf_test" else None

    with patch("src.api.routes.analyze.get_session_store", return_value=mock_store), \
         patch("src.memory.chroma_store.get_chroma_store", return_value=MagicMock()), \
         patch("src.core.config.settings") as mock_settings, \
         patch("src.api.auth.settings") as mock_auth:

        mock_settings.auth_enabled = False
        mock_settings.cors_origins = ["*"]
        mock_settings.api_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_expire_minutes = 60
        mock_auth.auth_enabled = False

        from src.api.main import app
        yield TestClient(app)


class TestPdfReport:
    """Export PDF d'une session existante."""

    def test_pdf_returns_200(self, app_client_with_session):
        """GET /analyze/{session_id}/report.pdf → HTTP 200."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.pdf")
        assert response.status_code == 200

    def test_pdf_content_type(self, app_client_with_session):
        """La réponse doit avoir le Content-Type application/pdf."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.pdf")
        assert "application/pdf" in response.headers.get("content-type", "")

    def test_pdf_is_valid_pdf(self, app_client_with_session):
        """Le contenu doit commencer par la signature PDF %PDF."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.pdf")
        assert response.content[:4] == b"%PDF"

    def test_pdf_not_empty(self, app_client_with_session):
        """Le PDF doit contenir au moins 1 KB de données."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.pdf")
        assert len(response.content) > 1024

    def test_pdf_unknown_session_returns_404(self, app_client_with_session):
        """Une session inconnue doit retourner HTTP 404."""
        response = app_client_with_session.get("/analyze/nonexistent_session/report.pdf")
        assert response.status_code == 404


openpyxl = pytest.importorskip("openpyxl", reason="openpyxl non installé")


class TestXlsxReport:
    """Export Excel d'une session existante (v0.4)."""

    def test_xlsx_returns_200(self, app_client_with_session):
        """GET /analyze/{session_id}/report.xlsx → HTTP 200."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.xlsx")
        assert response.status_code == 200

    def test_xlsx_content_type(self, app_client_with_session):
        """La réponse doit avoir le Content-Type Excel."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.xlsx")
        ct = response.headers.get("content-type", "")
        assert "spreadsheetml" in ct or "officedocument" in ct or "xlsx" in ct or ct == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def test_xlsx_is_valid_zip(self, app_client_with_session):
        """Un fichier .xlsx est un ZIP — les 2 premiers bytes doivent être PK (0x50 0x4B)."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.xlsx")
        assert response.content[:2] == b"PK"

    def test_xlsx_not_empty(self, app_client_with_session):
        """Le fichier Excel doit peser au moins 1 KB."""
        response = app_client_with_session.get("/analyze/sess_pdf_test/report.xlsx")
        assert len(response.content) > 1024

    def test_xlsx_unknown_session_returns_404(self, app_client_with_session):
        """Une session inconnue doit retourner HTTP 404."""
        response = app_client_with_session.get("/analyze/nonexistent_session/report.xlsx")
        assert response.status_code == 404
