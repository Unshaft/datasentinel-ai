"""
Tests d'intégration pour POST /analyze/{session_id}/apply-corrections (v0.5).
"""

import io
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.models import AgentContext, AgentType, IssueType, QualityIssue, Severity
from src.memory.session_store import SessionStore

client = TestClient(app)


# =============================================================================
# Helpers
# =============================================================================

def _make_issue(issue_type: IssueType, severity: Severity, column: str | None = "col_a") -> QualityIssue:
    return QualityIssue(
        issue_id=f"issue_{issue_type.value}_{column or 'global'}",
        issue_type=issue_type,
        severity=severity,
        column=column,
        description="test issue",
        affected_count=5,
        affected_percentage=10.0,
        confidence=0.9,
        detected_by=AgentType.QUALITY,
    )


def _make_context(session_id: str = "sess_apply_test") -> AgentContext:
    ctx = AgentContext(session_id=session_id, dataset_id="ds_apply")
    ctx.metadata["quality_score"] = 70.0
    ctx.metadata["final_status"] = "completed"
    ctx.issues = [
        _make_issue(IssueType.MISSING_VALUES, Severity.MEDIUM, "col_a"),
        _make_issue(IssueType.DUPLICATE, Severity.HIGH, None),
    ]
    return ctx


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "col_a": [1.0, None, 3.0, 4.0, None, 6.0],
        "col_b": ["x", "y", "y", "z", "x", "y"],
    })


# =============================================================================
# Tests
# =============================================================================

class TestApplyCorrections:

    def setup_method(self):
        """Réinitialise le singleton SessionStore entre chaque test."""
        SessionStore._instance = None

    def test_apply_corrections_200(self):
        """Session valide + DataFrame disponible → HTTP 200, CSV retourné."""
        ctx = _make_context("sess_apply_ok")
        df = _make_df()

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_apply_corrections_returns_valid_csv(self):
        """Le corps de la réponse est un CSV parseable."""
        ctx = _make_context("sess_apply_csv")
        df = _make_df()

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 200
        result_df = pd.read_csv(io.StringIO(resp.text))
        assert list(result_df.columns) == ["col_a", "col_b"]
        assert len(result_df) > 0

    def test_apply_corrections_drops_duplicates(self):
        """Les doublons sont supprimés."""
        ctx = _make_context("sess_apply_dup")
        ctx.issues = [_make_issue(IssueType.DUPLICATE, Severity.HIGH, None)]
        df = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": ["x", "y", "y", "z"],
        })

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 200
        rows_before = int(resp.headers["X-Rows-Before"])
        rows_after = int(resp.headers["X-Rows-After"])
        assert rows_after == 3   # 1 doublon supprimé
        assert rows_before == 4

    def test_apply_corrections_imputes_missing_numeric(self):
        """Les valeurs manquantes numériques sont imputées par la médiane."""
        ctx = _make_context("sess_apply_miss")
        ctx.issues = [_make_issue(IssueType.MISSING_VALUES, Severity.MEDIUM, "score")]
        df = pd.DataFrame({"score": [10.0, None, 30.0, None, 50.0]})

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 200
        result_df = pd.read_csv(io.StringIO(resp.text))
        assert result_df["score"].isna().sum() == 0  # plus aucun null

    def test_apply_corrections_session_not_found(self):
        """Session inconnue → HTTP 404."""
        store_mock = SessionStore()

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post("/analyze/nonexistent_session/apply-corrections")

        assert resp.status_code == 404

    def test_apply_corrections_df_not_available(self):
        """Contexte présent mais DataFrame absent → HTTP 422."""
        ctx = _make_context("sess_apply_nodf")
        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        # Ne pas sauvegarder le DataFrame

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 422

    def test_apply_corrections_headers(self):
        """Les headers X-Rows-Before, X-Rows-After, X-Corrections-Count sont présents."""
        ctx = _make_context("sess_apply_hdr")
        df = _make_df()

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert "X-Rows-Before" in resp.headers
        assert "X-Rows-After" in resp.headers
        assert "X-Corrections-Count" in resp.headers

    def test_apply_corrections_pseudo_nulls_replaced(self):
        """Les pseudo-nulls (N/A, null, -, etc.) sont remplacés avant imputation."""
        ctx = _make_context("sess_apply_pn")
        ctx.issues = [_make_issue(IssueType.MISSING_VALUES, Severity.MEDIUM, "label")]
        df = pd.DataFrame({"label": ["OK", "N/A", "null", "-", "GOOD", "UNKNOWN"]})

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 200
        result_df = pd.read_csv(io.StringIO(resp.text))
        # Pseudo-nulls remplacés par mode (qui est "OK" ou "GOOD")
        assert result_df["label"].str.lower().isin(["n/a", "null", "-", "unknown"]).sum() == 0

    def test_apply_corrections_no_auto_issues(self):
        """Aucune correction auto → CSV retourné sans modification."""
        ctx = _make_context("sess_apply_noop")
        ctx.issues = [_make_issue(IssueType.ANOMALY, Severity.LOW, "col_a")]
        df = _make_df()

        store_mock = SessionStore()
        store_mock.save(ctx.session_id, ctx)
        store_mock.save_dataframe(ctx.session_id, df)

        with patch("src.api.routes.analyze.get_session_store", return_value=store_mock), \
             patch("src.api.auth.settings") as mock_auth:
            mock_auth.auth_enabled = False
            resp = client.post(f"/analyze/{ctx.session_id}/apply-corrections")

        assert resp.status_code == 200
        rows_after = int(resp.headers["X-Rows-After"])
        rows_before = int(resp.headers["X-Rows-Before"])
        assert rows_after == rows_before  # rien n'a changé
