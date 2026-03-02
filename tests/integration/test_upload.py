"""
Tests d'intégration pour l'endpoint POST /upload.

Utilise httpx.AsyncClient + FastAPI TestClient (ASGI).
Le pipeline d'analyse est mocké pour isoler le routeur de l'orchestrateur.

Couvre :
- CSV valide → AnalyzeResponse
- Parquet valide → AnalyzeResponse
- Extension non supportée → HTTP 422
- Fichier vide → HTTP 400
- Fichier trop volumineux → HTTP 413
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.core.models import AgentContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _minimal_context(session_id: str = "sess_upload_test") -> AgentContext:
    """Contexte minimal retourné par l'orchestrateur mocké."""
    ctx = AgentContext(session_id=session_id, dataset_id="ds_upload")
    ctx.metadata["quality_score"] = 90.0
    ctx.metadata["final_status"] = "completed"
    ctx.metadata["processing_time_ms"] = 100
    ctx.metadata["summary"] = "Upload test OK"
    return ctx


@pytest.fixture
def df_sample() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 55000, 70000, 65000],
    })


@pytest.fixture
def app_client():
    """
    TestClient FastAPI avec orchestrateur et session store mockés.
    La dépendance JWT est court-circuitée (auth_enabled=False par défaut).
    """
    with patch("src.api.routes.upload.OrchestratorAgent") as MockOrch, \
         patch("src.api.routes.upload.get_session_store", return_value=MagicMock()), \
         patch("src.api.routes.upload.settings") as mock_settings, \
         patch("src.memory.chroma_store.get_chroma_store", return_value=MagicMock()), \
         patch("src.core.config.settings") as mock_config_settings, \
         patch("src.api.auth.settings") as mock_auth_settings:

        # Settings patchés sur upload.settings (référence locale au module)
        mock_settings.max_upload_size = 100 * 1024 * 1024  # 100 MB
        mock_settings.max_rows_analyze = 0  # illimité
        mock_settings.auth_enabled = False
        mock_settings.api_secret_key = "test-secret"
        mock_settings.jwt_algorithm = "HS256"
        mock_settings.jwt_expire_minutes = 60
        mock_settings.cors_origins = ["*"]

        # Patch auth.settings pour court-circuiter l'auth quel que soit l'ordre d'import
        mock_auth_settings.auth_enabled = False
        mock_config_settings.auth_enabled = False
        mock_config_settings.cors_origins = ["*"]

        # Orchestrateur mocké (pipeline async requis depuis v0.3)
        mock_orch = MagicMock()
        mock_orch.run_pipeline.return_value = _minimal_context()
        mock_orch.run_pipeline_async = AsyncMock(return_value=_minimal_context())
        MockOrch.return_value = mock_orch

        from src.api.main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUploadCSV:
    """Upload de fichiers CSV."""

    def test_csv_valid_returns_analyze_response(self, app_client, df_sample):
        """Un CSV valide doit déclencher l'analyse et retourner AnalyzeResponse."""
        csv_bytes = _make_csv_bytes(df_sample)
        response = app_client.post(
            "/upload",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "session_id" in body
        assert "quality_score" in body
        assert body["status"] == "completed"

    def test_csv_empty_raises_400(self, app_client):
        """Un CSV vide (aucune ligne) doit retourner HTTP 400."""
        empty_csv = b"col_a,col_b\n"  # Header only, no rows
        response = app_client.post(
            "/upload",
            files={"file": ("empty.csv", empty_csv, "text/csv")},
        )
        assert response.status_code == 400


class TestUploadParquet:
    """Upload de fichiers Parquet."""

    def test_parquet_valid_returns_analyze_response(self, app_client, df_sample):
        """Un Parquet valide doit déclencher l'analyse et retourner AnalyzeResponse."""
        parquet_bytes = _make_parquet_bytes(df_sample)
        response = app_client.post(
            "/upload",
            files={"file": ("data.parquet", parquet_bytes, "application/octet-stream")},
        )
        assert response.status_code == 200
        body = response.json()
        assert "session_id" in body
        assert "quality_score" in body


class TestUploadValidation:
    """Validation des fichiers uploadés."""

    def test_unsupported_extension_raises_422(self, app_client):
        """Une extension non supportée (.xlsx) doit retourner HTTP 422."""
        response = app_client.post(
            "/upload",
            files={"file": ("data.xlsx", b"fake content", "application/octet-stream")},
        )
        assert response.status_code == 422
        assert "non support" in response.json()["detail"].lower() or \
               "extension" in response.json()["detail"].lower()

    def test_file_too_large_raises_413(self, app_client, df_sample):
        """Un fichier dépassant la limite doit retourner HTTP 413."""
        csv_bytes = _make_csv_bytes(df_sample)
        # Patch max_upload_size à 10 bytes pour déclencher la limite
        with patch("src.api.routes.upload.settings") as mock_settings:
            mock_settings.max_upload_size = 10
            mock_settings.max_rows_analyze = 0
            response = app_client.post(
                "/upload",
                files={"file": ("big.csv", csv_bytes, "text/csv")},
            )
        assert response.status_code == 413

    def test_no_extension_raises_422(self, app_client):
        """Un fichier sans extension doit retourner HTTP 422."""
        response = app_client.post(
            "/upload",
            files={"file": ("datafile", b"some,data\n1,2\n", "text/plain")},
        )
        assert response.status_code == 422
