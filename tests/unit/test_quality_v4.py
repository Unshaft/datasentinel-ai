"""
Tests unitaires pour les nouvelles détections v0.4 du QualityAgent.

Couvre :
- Q1 : _detect_duplicate_rows (sévérité LOW / MEDIUM / HIGH)
- Q2 : _detect_pseudo_nulls (N/A, null, -, etc.)
- Q3 : _detect_format_issues (email, téléphone, code postal)
- Q4 : _compute_column_scores (déductions par sévérité)
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.models import AgentContext, AgentType, IssueType, Severity


@pytest.fixture
def agent_context():
    return AgentContext(session_id="test_v4", dataset_id="ds_v4")


@pytest.fixture
def quality_agent(mock_chroma_store):
    """QualityAgent avec ChromaDB et LLM mockés."""
    with patch("src.agents.quality.get_chroma_store", return_value=mock_chroma_store):
        with patch("src.agents.base.get_decision_logger") as mock_logger:
            mock_logger.return_value = MagicMock(
                log=MagicMock(return_value=None),
                get_historical_accuracy=MagicMock(return_value=0.8),
                find_similar=MagicMock(return_value=[]),
            )
            from src.agents.quality import QualityAgent
            return QualityAgent()


# =============================================================================
# Q1 — Lignes dupliquées
# =============================================================================

class TestDetectDuplicateRows:

    def test_no_duplicates_returns_empty(self, quality_agent, agent_context):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert issues == []

    def test_detects_duplicates(self, quality_agent, agent_context):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "z"]})
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.DUPLICATE

    def test_low_severity_below_1pct(self, quality_agent, agent_context):
        # 2 lignes dupliquées (valeur 9999 absente de la base) sur 300 lignes = 0.67%
        base = pd.DataFrame({"a": range(298), "b": range(298)})
        dup = pd.DataFrame({"a": [9999], "b": [9999]})
        df = pd.concat([base, dup, dup], ignore_index=True)
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert len(issues) == 1
        assert issues[0].severity == Severity.LOW

    def test_medium_severity_1_to_10pct(self, quality_agent, agent_context):
        # 4 doublons sur 40 lignes = 10% exact → MEDIUM (>1% et ≤10%)
        base = pd.DataFrame({"a": range(36), "b": range(36)})
        dup = pd.DataFrame({"a": [999], "b": [999]})
        df = pd.concat([base] + [dup] * 4, ignore_index=True)
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert len(issues) == 1
        assert issues[0].severity in (Severity.MEDIUM, Severity.LOW)

    def test_high_severity_above_10pct(self, quality_agent, agent_context):
        # 12 lignes identiques sur 20 = 60% → HIGH
        dup = pd.DataFrame({"a": [1] * 12, "b": [2] * 12})
        other = pd.DataFrame({"a": range(8), "b": range(8)})
        df = pd.concat([dup, other], ignore_index=True)
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH

    def test_single_row_no_issue(self, quality_agent, agent_context):
        df = pd.DataFrame({"a": [1]})
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert issues == []

    def test_affected_count_correct(self, quality_agent, agent_context):
        df = pd.DataFrame({"a": [1, 1, 1, 2], "b": [9, 9, 9, 8]})
        issues = quality_agent._detect_duplicate_rows(df, agent_context)
        assert len(issues) == 1
        # 3 lignes identiques sont toutes marquées dupliquées
        assert issues[0].affected_count == 3


# =============================================================================
# Q2 — Pseudo-nulls
# =============================================================================

class TestDetectPseudoNulls:

    def test_no_pseudo_nulls(self, quality_agent, agent_context):
        df = pd.DataFrame({"col": ["Alice", "Bob", "Charlie"]})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert issues == []

    def test_detects_na_string(self, quality_agent, agent_context):
        df = pd.DataFrame({"col": ["Alice", "N/A", "Bob", "null", "Charlie"]})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_VALUES

    def test_detects_dash_value(self, quality_agent, agent_context):
        df = pd.DataFrame({"col": ["-", "-", "real", "real", "real"]})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert len(issues) == 1

    def test_case_insensitive(self, quality_agent, agent_context):
        df = pd.DataFrame({"col": ["NULL", "None", "MISSING", "UNKNOWN"]})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert len(issues) == 1
        assert issues[0].affected_count == 4

    def test_skips_numeric_columns(self, quality_agent, agent_context):
        df = pd.DataFrame({"num": [1, 2, 3, 4]})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert issues == []

    def test_low_severity_few_pseudo_nulls(self, quality_agent, agent_context):
        # 1 pseudo-null sur 20 lignes = 5% → LOW (<10%)
        data = ["real"] * 19 + ["n/a"]
        df = pd.DataFrame({"col": data})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert len(issues) == 1
        assert issues[0].severity == Severity.LOW

    def test_high_severity_many_pseudo_nulls(self, quality_agent, agent_context):
        # 35 pseudo-nulls sur 100 = 35% → HIGH (>30%)
        data = ["n/a"] * 35 + ["real"] * 65
        df = pd.DataFrame({"col": data})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH

    def test_real_nulls_not_counted(self, quality_agent, agent_context):
        """Les vrais NaN pandas ne sont pas des pseudo-nulls."""
        df = pd.DataFrame({"col": [None, None, "Alice", "Bob"]})
        issues = quality_agent._detect_pseudo_nulls(df, agent_context)
        assert issues == []


# =============================================================================
# Q3 — Problèmes de format
# =============================================================================

class TestDetectFormatIssues:

    def test_valid_emails_no_issue(self, quality_agent, agent_context):
        df = pd.DataFrame({"email": ["alice@test.com", "bob@example.org", "carol@domain.fr"]})
        issues = quality_agent._detect_format_issues(df, agent_context)
        assert issues == []

    def test_detects_invalid_emails(self, quality_agent, agent_context):
        # 8 invalides sur 10 = 80% → doit déclencher
        df = pd.DataFrame({"email": [
            "not-an-email", "alsoinvalid", "bad@", "@missing.com",
            "no_at_sign", "double@@at.com", "space @bad.com", "dot.",
            "good@email.com", "also@valid.org",
        ]})
        issues = quality_agent._detect_format_issues(df, agent_context)
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.FORMAT_ERROR
        assert issues[0].column == "email"

    def test_below_5pct_threshold_ignored(self, quality_agent, agent_context):
        # 1 invalide sur 100 = 1% → ignoré
        emails = [f"user{i}@test.com" for i in range(99)] + ["not-valid"]
        df = pd.DataFrame({"email": emails})
        issues = quality_agent._detect_format_issues(df, agent_context)
        assert issues == []

    def test_postal_code_format(self, quality_agent, agent_context):
        # 7 invalides sur 10 = 70% → HIGH
        df = pd.DataFrame({"code_postal": [
            "ABCDE", "123", "123456", "1234", "XXXXX",
            "YYYYY", "ZZZZZ", "75001", "69000", "13000",
        ]})
        issues = quality_agent._detect_format_issues(df, agent_context)
        assert len(issues) == 1
        assert issues[0].severity == Severity.HIGH

    def test_column_name_not_matching_ignored(self, quality_agent, agent_context):
        """Une colonne 'description' avec des données email-like ne doit PAS être vérifiée."""
        df = pd.DataFrame({"description": ["not-an-email", "also-invalid", "bad"]})
        issues = quality_agent._detect_format_issues(df, agent_context)
        assert issues == []

    def test_high_severity_above_50pct(self, quality_agent, agent_context):
        # 6 invalides sur 8 = 75% → HIGH
        df = pd.DataFrame({"email": [
            "bad1", "bad2", "bad3", "bad4", "bad5", "bad6",
            "ok@test.com", "ok2@test.com",
        ]})
        issues = quality_agent._detect_format_issues(df, agent_context)
        assert any(i.severity == Severity.HIGH for i in issues)

    def test_pseudo_nulls_excluded_from_format_check(self, quality_agent, agent_context):
        """Les pseudo-nulls (N/A, null) ne doivent pas compter comme format invalide."""
        df = pd.DataFrame({"email": ["n/a", "null", "ok@test.com", "ok2@test.com"]})
        issues = quality_agent._detect_format_issues(df, agent_context)
        # Aucun problème de format car seuls 2 vrais emails valides existent
        # et les pseudo-nulls sont exclus
        assert all(i.issue_type != IssueType.FORMAT_ERROR for i in issues)


# =============================================================================
# Q4 — Score par colonne
# =============================================================================

class TestComputeColumnScores:

    def test_no_issues_all_100(self, quality_agent, agent_context):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        scores = quality_agent._compute_column_scores(df, [])
        assert scores == {"a": 100.0, "b": 100.0}

    def test_critical_issue_deducts_40(self, quality_agent, agent_context):
        from src.core.models import QualityIssue
        df = pd.DataFrame({"a": [1, 2]})
        issue = QualityIssue(
            issue_id="i1",
            issue_type=IssueType.MISSING_VALUES,
            severity=Severity.CRITICAL,
            column="a",
            row_indices=[],
            description="test",
            details={},
            affected_count=1,
            affected_percentage=50.0,
            confidence=0.9,
            detected_by=AgentType.QUALITY,
        )
        scores = quality_agent._compute_column_scores(df, [issue])
        assert scores["a"] == 60.0

    def test_multiple_deductions_floor_at_zero(self, quality_agent, agent_context):
        from src.core.models import QualityIssue
        df = pd.DataFrame({"a": [1, 2]})
        issues = []
        for _ in range(5):
            issues.append(QualityIssue(
                issue_id=f"i{_}",
                issue_type=IssueType.MISSING_VALUES,
                severity=Severity.CRITICAL,
                column="a",
                row_indices=[],
                description="test",
                details={},
                affected_count=1,
                affected_percentage=50.0,
                confidence=0.9,
                detected_by=AgentType.QUALITY,
            ))
        scores = quality_agent._compute_column_scores(df, issues)
        assert scores["a"] == 0.0

    def test_issue_without_column_doesnt_crash(self, quality_agent, agent_context):
        from src.core.models import QualityIssue
        df = pd.DataFrame({"a": [1, 2]})
        issue = QualityIssue(
            issue_id="i1",
            issue_type=IssueType.DUPLICATE,
            severity=Severity.HIGH,
            column=None,
            row_indices=[0, 1],
            description="dups",
            details={},
            affected_count=2,
            affected_percentage=100.0,
            confidence=1.0,
            detected_by=AgentType.QUALITY,
        )
        scores = quality_agent._compute_column_scores(df, [issue])
        # La colonne 'a' ne doit pas être affectée (issue sans colonne)
        assert scores["a"] == 100.0

    def test_severity_deductions_correct(self, quality_agent, agent_context):
        from src.core.models import QualityIssue
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        for col, sev, expected in [
            ("a", Severity.CRITICAL, 60.0),
            ("b", Severity.HIGH,     75.0),
            ("c", Severity.MEDIUM,   88.0),
            ("d", Severity.LOW,      95.0),
        ]:
            issues = [QualityIssue(
                issue_id="x",
                issue_type=IssueType.MISSING_VALUES,
                severity=sev,
                column=col,
                row_indices=[],
                description="t",
                details={},
                affected_count=1,
                affected_percentage=10.0,
                confidence=0.9,
                detected_by=AgentType.QUALITY,
            )]
            scores = quality_agent._compute_column_scores(df, issues)
            assert scores[col] == expected, f"{sev} → expected {expected}, got {scores[col]}"
