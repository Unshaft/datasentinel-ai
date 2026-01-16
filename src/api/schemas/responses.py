"""
Schémas Pydantic pour les réponses API.

Ces schémas définissent la structure des réponses
pour garantir une API cohérente et documentée.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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

    # Escalade
    needs_human_review: bool = Field(
        default=False,
        description="Une revue humaine est-elle recommandée?"
    )
    escalation_reasons: list[str] = Field(
        default_factory=list,
        description="Raisons de l'escalade"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "dataset_id": "dataset_def456",
                "status": "completed",
                "quality_score": 75.5,
                "processing_time_ms": 1234,
                "summary": "Bonne qualité (75.5%) | Problèmes: 3",
                "issues": [],
                "needs_human_review": False
            }
        }


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
    timestamp: datetime = Field(default_factory=datetime.utcnow)


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
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "error_type": "ValidationError",
                "message": "Les données fournies sont invalides",
                "details": {"field": "data", "reason": "Aucune donnée fournie"},
                "request_id": "req_abc123"
            }
        }


class SessionListResponse(BaseModel):
    """Liste des sessions."""

    sessions: list[dict[str, Any]]
    total: int
    page: int = 1
    page_size: int = 20
