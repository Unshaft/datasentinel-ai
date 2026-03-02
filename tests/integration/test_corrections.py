"""
Tests d'intégration pour l'endpoint GET /analyze/{session_id}/corrections.

Couvre :
- Session valide → HTTP 200, plan JSON structuré
- Session inexistante → HTTP 404
- Issues auto-corrigeables vs review manuelle
- `estimated_score_after_auto` supérieur ou égal au score initial
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.models import AgentContext, AgentType, IssueType, QualityIssue, Severity


def _make_issue(issue_type: IssueType, severity: Severity, column: str | None = "col") -> QualityIssue:
    return QualityIssue(
        issue_id=f"i_{issue_type.value}",
        issue_type=issue_type,
        severity=severity,
        column=column,
        row_indices=[],
        description=f"Test issue {issue_type.value}",
        details={},
        affected_count=5,
        affected_percentage=10.0,
        confidence=0.9,
        detected_by=AgentType.QUALITY,
    )


def _make_context(session_id: str = "sess_corr_test") -> AgentContext:
    ctx = AgentContext(session_id=session_id, dataset_id="ds_corr")
    ctx.metadata["quality_score"] = 70.0
    ctx.metadata["final_status"] = "completed"
    ctx.issues = [
        _make_issue(IssueType.MISSING_VALUES, Severity.MEDIUM),
        _make_issue(IssueType.DUPLICATE, Severity.HIGH),
        _make_issue(IssueType.ANOMALY, Severity.LOW),
        _make_issue(IssueType.CONSTRAINT_VIOLATION, Severity.HIGH, column="id"),
    ]
    return ctx


@pytest.fixture
def client_with_session():
    ctx = _make_context("sess_corr_test")

    mock_store = MagicMock()
    mock_store.load.side_effect = lambda sid: ctx if sid == "sess_corr_test" else None

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


class TestCorrectionsEndpoint:

    def test_returns_200_for_valid_session(self, client_with_session):
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        assert resp.status_code == 200

    def test_returns_404_for_unknown_session(self, client_with_session):
        resp = client_with_session.get("/analyze/nonexistent_session/corrections")
        assert resp.status_code == 404

    def test_response_has_required_keys(self, client_with_session):
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        assert "session_id" in data
        assert "quality_score" in data
        assert "total_issues" in data
        assert "auto_corrections" in data
        assert "manual_reviews" in data
        assert "estimated_score_after_auto" in data

    def test_missing_values_and_duplicate_are_auto(self, client_with_session):
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        auto_types = {e["issue_type"] for e in data["auto_corrections"]}
        assert "missing_values" in auto_types
        assert "duplicate" in auto_types

    def test_anomaly_and_constraint_are_manual(self, client_with_session):
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        manual_types = {e["issue_type"] for e in data["manual_reviews"]}
        assert "anomaly" in manual_types
        assert "constraint_violation" in manual_types

    def test_total_issues_matches(self, client_with_session):
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        total = len(data["auto_corrections"]) + len(data["manual_reviews"])
        assert total == data["total_issues"]

    def test_estimated_score_gte_current(self, client_with_session):
        """Le score estimé après corrections auto doit être ≥ score actuel."""
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        assert data["estimated_score_after_auto"] >= data["quality_score"]

    def test_estimated_score_capped_at_100(self, client_with_session):
        """Le score estimé ne doit jamais dépasser 100."""
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        assert data["estimated_score_after_auto"] <= 100.0

    def test_each_correction_has_required_fields(self, client_with_session):
        resp = client_with_session.get("/analyze/sess_corr_test/corrections")
        data = resp.json()
        for entry in data["auto_corrections"] + data["manual_reviews"]:
            assert "issue_id" in entry
            assert "issue_type" in entry
            assert "severity" in entry
            assert "recommended_action" in entry
            assert "auto_applicable" in entry
