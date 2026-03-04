"""
Schémas Pydantic pour les réponses API.

Ces schémas définissent la structure des réponses
pour garantir une API cohérente et documentée.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# DATASET MEMORY (v1.2 — F30)
# =============================================================================


class DatasetMemoryInfo(BaseModel):
    """Résumé de la mémoire inter-sessions pour AnalyzeResponse (F30 — v1.2)."""

    dataset_id: str
    is_known: bool = Field(description="True si ce dataset avait déjà été analysé")
    session_count: int = Field(description="Nombre total d'analyses enregistrées")
    avg_quality_score: float = Field(description="Score moyen sur toutes les sessions")
    trend: str = Field(description="'new' | 'improving' | 'degrading' | 'stable'")
    recurring_issues: list[str] = Field(
        default_factory=list,
        description="Top 3 types d'issues les plus fréquents"
    )
    suggested_rules: list[str] = Field(
        default_factory=list,
        description="Suggestions de règles pro-actives générées automatiquement"
    )


class DatasetSessionInfo(BaseModel):
    """Résumé d'une session dans l'historique dataset (F30 — v1.2)."""

    session_id: str
    timestamp: str
    quality_score: float
    issue_counts: dict[str, int] = Field(default_factory=dict)
    top_columns: list[str] = Field(default_factory=list)


class DatasetHistoryResponse(BaseModel):
    """Historique complet d'un dataset — GET /datasets/{id}/history (F30 — v1.2)."""

    dataset_id: str
    first_seen: str
    last_seen: str
    session_count: int
    avg_quality_score: float
    trend: str
    recurring_issues: dict[str, int] = Field(
        default_factory=dict,
        description="issue_type → nombre de sessions concernées"
    )
    problematic_columns: dict[str, int] = Field(
        default_factory=dict,
        description="col_name → nombre de sessions avec une issue sur cette colonne"
    )
    suggested_rules: list[str] = Field(default_factory=list)
    sessions: list[DatasetSessionInfo] = Field(
        default_factory=list,
        description="10 dernières sessions enregistrées"
    )


# =============================================================================
# SEMANTIC SCHEMA (v0.8 — F27/F29)
# =============================================================================


class SemanticColumnInfo(BaseModel):
    """Classification sémantique d'une colonne (F27/F29 — v0.8)."""

    name: str
    dtype: str
    inferred_type: str                  # Type technique du ProfilingAgent
    semantic_type: str | None = None    # Type métier du SemanticProfilerAgent
    confidence: float | None = None     # Confiance LLM (0.0-1.0)
    language: str | None = None         # Langue détectée ("fr", "en", ...)
    pattern: str | None = None          # Regex détectée si applicable
    notes: str | None = None            # Observations LLM
    null_percentage: float = 0.0
    unique_count: int = 0
    sample_values: list[Any] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    """Export du schéma sémantique d'un dataset analysé (F29 — v0.8)."""

    session_id: str
    dataset_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    columns: list[SemanticColumnInfo] = Field(default_factory=list)
    semantic_coverage: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="% de colonnes avec un type sémantique LLM (0-100)"
    )


# =============================================================================
# MODÈLES DE BASE
# =============================================================================


class ColumnProfileResponse(BaseModel):
    """Profil d'une colonne."""

    name: str
    dtype: str
    inferred_type: str
    count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    sample_values: list[Any] = Field(default_factory=list)


class DataProfileResponse(BaseModel):
    """Profil complet du dataset."""

    dataset_id: str
    row_count: int
    column_count: int
    memory_mb: float
    total_null_count: int
    columns: list[ColumnProfileResponse]


class IssueResponse(BaseModel):
    """Problème de qualité détecté."""

    issue_id: str
    issue_type: str
    severity: str
    column: str | None
    description: str
    affected_count: int
    affected_percentage: float
    confidence: float
    details: dict[str, Any] = Field(default_factory=dict)


class ProposalResponse(BaseModel):
    """Proposition de correction."""

    proposal_id: str
    issue_id: str
    correction_type: str
    description: str
    justification: str
    estimated_impact: str
    rows_affected: int
    confidence: float
    alternatives: list[str] = Field(default_factory=list)
    is_approved: bool | None = None


class ValidationResponse(BaseModel):
    """Résultat de validation."""

    validation_id: str
    proposal_id: str
    is_valid: bool
    status: str
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rules_checked: list[str] = Field(default_factory=list)


# =============================================================================
# RÉPONSES API
# =============================================================================


class AnalyzeResponse(BaseModel):
    """Réponse complète d'analyse."""

    session_id: str = Field(..., description="ID unique de la session")
    dataset_id: str = Field(..., description="ID du dataset analysé")
    status: str = Field(..., description="Statut: completed, escalated, failed")

    # Métriques
    quality_score: float = Field(..., ge=0, le=100, description="Score de qualité 0-100")
    processing_time_ms: int = Field(..., description="Temps de traitement en ms")

    # Résumé
    summary: str = Field(..., description="Résumé textuel de l'analyse")

    # Profil
    profile: DataProfileResponse | None = Field(
        default=None,
        description="Profil statistique du dataset"
    )

    # Problèmes détectés
    issues: list[IssueResponse] = Field(
        default_factory=list,
        description="Liste des problèmes de qualité"
    )
    issues_by_severity: dict[str, int] = Field(
        default_factory=dict,
        description="Nombre de problèmes par sévérité"
    )

    # Score par colonne (v0.4)
    column_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score de qualité individuel par colonne (0-100)"
    )

    # Types sémantiques LLM (v0.8 — F27, présent si ENABLE_LLM_CHECKS=true)
    semantic_types: dict[str, Any] | None = Field(
        default=None,
        description="Classification sémantique des colonnes (opt-in LLM)"
    )

    # Agent métier activé (v1.0 — F32, None si aucun profil ne correspond)
    domain_agent: str | None = Field(
        default=None,
        description="Nom de l'agent métier activé (F32), None si aucun profil ne correspond"
    )

    # Mémoire inter-sessions (v1.2 — F30, None si feature désactivée)
    dataset_memory: DatasetMemoryInfo | None = Field(
        default=None,
        description="Historique et tendance qualité pour ce dataset (F30)"
    )

    # ReAct Reflect (v1.3 — F31, vide si pipeline standard)
    reflect_flags: list[str] = Field(
        default_factory=list,
        description="Flags d'incohérence détectés par la phase Reflect (F31)"
    )

    # Étapes du raisonnement ReAct (v0.7 — F24, vide si include_reasoning=false)
    reasoning_steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Étapes du raisonnement ReAct (F24)"
    )

    # Escalade
    needs_human_review: bool = Field(
        default=False,
        description="Une revue humaine est-elle recommandée?"
    )
    escalation_reasons: list[str] = Field(
        default_factory=list,
        description="Raisons de l'escalade"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "session_abc123",
            "dataset_id": "dataset_def456",
            "status": "completed",
            "quality_score": 75.5,
            "processing_time_ms": 1234,
            "summary": "Bonne qualité (75.5%) | Problèmes: 3",
            "issues": [],
            "column_scores": {"age": 90.0, "email": 60.0},
            "needs_human_review": False
        }
    })


class RecommendResponse(BaseModel):
    """Réponse avec recommandations de corrections."""

    session_id: str
    status: str
    quality_score: float

    # Analyse (si incluse)
    issues_count: int = Field(..., description="Nombre de problèmes détectés")

    # Recommandations
    proposals: list[ProposalResponse] = Field(
        default_factory=list,
        description="Corrections proposées"
    )

    # Métriques
    estimated_improvement: float = Field(
        ..., ge=0, le=100,
        description="Amélioration estimée du score si corrections appliquées"
    )

    summary: str


class ExplainResponse(BaseModel):
    """Réponse avec explication détaillée."""

    session_id: str
    target_id: str
    target_type: str

    # Explication
    explanation: str = Field(..., description="Explication en langage naturel")

    # Facteurs
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Facteurs ayant contribué à la décision"
    )

    # Confiance
    confidence_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Décomposition du score de confiance"
    )

    # Contexte
    related_rules: list[str] = Field(
        default_factory=list,
        description="Règles métier pertinentes"
    )
    similar_past_decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Décisions passées similaires"
    )


class FeedbackResponse(BaseModel):
    """Confirmation de prise en compte du feedback."""

    feedback_id: str
    status: str = Field(default="recorded")
    message: str
    impact: str = Field(
        ...,
        description="Comment ce feedback sera utilisé"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RuleResponse(BaseModel):
    """Règle métier."""

    rule_id: str
    text: str
    rule_type: str
    severity: str
    category: str
    active: bool = True


class AddRuleResponse(BaseModel):
    """Confirmation d'ajout de règle."""

    status: str
    message: str
    rule: RuleResponse


class HealthResponse(BaseModel):
    """Réponse de health check."""

    status: str = Field(..., description="healthy, degraded, unhealthy")
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Composants
    components: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="État des composants"
    )


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée."""

    error_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str | None = None

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error_type": "ValidationError",
            "message": "Les données fournies sont invalides",
            "details": {"field": "data", "reason": "Aucune donnée fournie"},
            "request_id": "req_abc123"
        }
    })


class SessionListResponse(BaseModel):
    """Liste des sessions."""

    sessions: list[dict[str, Any]]
    total: int
    page: int = 1
    page_size: int = 20


# =============================================================================
# BATCH (v0.5)
# =============================================================================


class BatchResultItem(BaseModel):
    """Résultat d'analyse pour un fichier dans un batch."""

    filename: str = Field(..., description="Nom du fichier analysé")
    session_id: str | None = Field(None, description="ID de session (None si erreur)")
    status: str = Field(..., description="success | error")
    quality_score: float | None = Field(None, description="Score 0-100 (None si erreur)")
    issues_count: int = Field(default=0, description="Nombre de problèmes détectés")
    error: str | None = Field(None, description="Message d'erreur si status=error")


class BatchAnalyzeResponse(BaseModel):
    """Réponse d'analyse batch (plusieurs fichiers)."""

    total: int = Field(..., description="Nombre de fichiers soumis")
    succeeded: int = Field(..., description="Analyses réussies")
    failed: int = Field(..., description="Analyses en erreur")
    results: list[BatchResultItem] = Field(default_factory=list)


# =============================================================================
# COMPARISON (v0.6 — F19)
# =============================================================================


class ComparisonResponse(BaseModel):
    """Résultat de comparaison avant/après corrections."""

    session_id: str
    score_before: float = Field(..., ge=0, le=100)
    score_after: float = Field(..., ge=0, le=100)
    delta: float = Field(..., description="Amélioration du score (peut être 0)")
    issues_removed: list[str] = Field(
        default_factory=list,
        description="Types de problèmes disparus après corrections"
    )
    issues_remaining: list[str] = Field(
        default_factory=list,
        description="Types de problèmes persistants"
    )
    columns_improved: list[str] = Field(
        default_factory=list,
        description="Colonnes avec score amélioré"
    )


# =============================================================================
# RULES (v0.6 — F20)
# =============================================================================


class RuleListResponse(BaseModel):
    """Liste de règles métier."""

    status: str = "success"
    count: int
    rules: list[RuleResponse] = Field(default_factory=list)


class RuleCreateResponse(BaseModel):
    """Confirmation de création d'une règle."""

    status: str = "created"
    rule: RuleResponse


# =============================================================================
# ASYNC JOBS (v0.6 — F21)
# =============================================================================


class JobCreateResponse(BaseModel):
    """Réponse à la création d'un job asynchrone."""

    job_id: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobStatusResponse(BaseModel):
    """Statut d'un job asynchrone."""

    job_id: str
    status: str = Field(..., description="pending | running | completed | failed")
    progress: float = Field(default=0.0, ge=0, le=100)
    result: "AnalyzeResponse | None" = None
    error: str | None = None
    created_at: datetime | None = None


# =============================================================================
# STATS (v0.6 — F22)
# =============================================================================


class StatsResponse(BaseModel):
    """Dashboard analytique agrégé."""

    total_sessions: int = 0
    avg_quality_score: float = 0.0
    top_issue_types: dict[str, int] = Field(default_factory=dict)
    sessions_by_day: dict[str, int] = Field(default_factory=dict)
    score_distribution: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
