"""
Modèles de domaine pour DataSentinel AI.

Ce module définit toutes les structures de données utilisées
à travers le système. Ces modèles sont la "langue commune"
entre tous les agents et composants.

Design Principles:
- Immutabilité où possible (frozen=True pour les données de sortie)
- Validation stricte des entrées
- Sérialisation JSON native pour l'API
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS - Types énumérés pour typage strict
# =============================================================================


class Severity(str, Enum):
    """Niveau de sévérité d'un problème détecté."""

    LOW = "low"           # Informatif, pas d'action requise
    MEDIUM = "medium"     # À surveiller, correction recommandée
    HIGH = "high"         # Problème significatif, correction nécessaire
    CRITICAL = "critical" # Bloquant, action immédiate requise


class IssueType(str, Enum):
    """Type de problème de qualité des données."""

    MISSING_VALUES = "missing_values"       # Valeurs nulles/manquantes
    ANOMALY = "anomaly"                     # Valeur statistiquement anormale
    TYPE_MISMATCH = "type_mismatch"         # Type de données incohérent
    CONSTRAINT_VIOLATION = "constraint_violation"  # Violation de règle métier
    DRIFT = "drift"                         # Dérive de distribution
    DUPLICATE = "duplicate"                 # Ligne/valeur dupliquée
    FORMAT_ERROR = "format_error"           # Format incorrect (date, email, etc.)
    OUTLIER = "outlier"                     # Valeur aberrante
    INCONSISTENCY = "inconsistency"         # Incohérence entre colonnes


class CorrectionType(str, Enum):
    """Type de correction proposée."""

    IMPUTE_MEAN = "impute_mean"         # Imputation par moyenne
    IMPUTE_MEDIAN = "impute_median"     # Imputation par médiane
    IMPUTE_MODE = "impute_mode"         # Imputation par mode
    IMPUTE_CUSTOM = "impute_custom"     # Imputation valeur personnalisée
    DELETE_ROW = "delete_row"           # Suppression de ligne
    DELETE_COLUMN = "delete_column"     # Suppression de colonne
    CAST_TYPE = "cast_type"             # Conversion de type
    CLIP_VALUES = "clip_values"         # Écrêtage des valeurs
    STANDARDIZE = "standardize"         # Standardisation
    FLAG_ONLY = "flag_only"             # Marquage sans correction
    MANUAL_REVIEW = "manual_review"     # Revue manuelle requise


class AgentType(str, Enum):
    """Type d'agent dans le système."""

    ORCHESTRATOR = "orchestrator"
    PROFILER = "profiler"
    QUALITY = "quality"
    CORRECTOR = "corrector"
    VALIDATOR = "validator"


class TaskStatus(str, Enum):
    """Statut d'exécution d'une tâche."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"  # Remonté pour intervention humaine


class SemanticType(str, Enum):
    """Type sémantique / métier d'une colonne (F27 — v0.8)."""

    EMAIL = "email"
    PHONE = "phone"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    FULL_NAME = "full_name"
    POSTAL_CODE = "postal_code"
    ADDRESS = "address"
    CITY = "city"
    COUNTRY = "country"
    IDENTIFIER = "identifier"
    MONETARY_AMOUNT = "monetary_amount"
    PERCENTAGE = "percentage"
    AGE = "age"
    DATE_STRING = "date_string"
    URL = "url"
    IP_ADDRESS = "ip_address"
    BOOLEAN_TEXT = "boolean_text"
    CATEGORY = "category"
    PRODUCT_CODE = "product_code"
    EMPLOYEE_ID = "employee_id"
    DESCRIPTION = "description"
    FREE_TEXT = "free_text"
    QUANTITY = "quantity"
    RATING = "rating"


class SemanticColumnType(TypedDict, total=False):
    """Résultat de classification sémantique d'une colonne (F27 — v0.8)."""

    semantic_type: str          # Valeur de SemanticType enum
    confidence: float           # 0.0 – 1.0
    language: str | None        # "fr", "en", etc.
    pattern: str | None         # Regex détectée si applicable
    notes: str | None           # Observations additionnelles


# =============================================================================
# DATA MODELS - Structures de données principales
# =============================================================================


class ColumnProfile(BaseModel):
    """Profil statistique d'une colonne."""

    name: str = Field(..., description="Nom de la colonne")
    dtype: str = Field(..., description="Type de données Pandas")
    inferred_type: str = Field(..., description="Type sémantique inféré")

    # Statistiques de base
    count: int = Field(..., ge=0, description="Nombre de valeurs non-nulles")
    null_count: int = Field(..., ge=0, description="Nombre de valeurs nulles")
    null_percentage: float = Field(..., ge=0, le=100, description="% de nulls")
    unique_count: int = Field(..., ge=0, description="Nombre de valeurs uniques")
    unique_percentage: float = Field(..., ge=0, le=100, description="% de valeurs uniques")

    # Statistiques numériques (optionnelles)
    mean: float | None = Field(default=None, description="Moyenne")
    std: float | None = Field(default=None, description="Écart-type")
    min: float | None = Field(default=None, description="Minimum")
    max: float | None = Field(default=None, description="Maximum")
    q25: float | None = Field(default=None, description="Premier quartile")
    q50: float | None = Field(default=None, description="Médiane")
    q75: float | None = Field(default=None, description="Troisième quartile")

    # Échantillons
    sample_values: list[Any] = Field(
        default_factory=list,
        max_length=10,
        description="Échantillon de valeurs"
    )

    @property
    def is_numeric(self) -> bool:
        """Vérifie si la colonne est numérique."""
        return self.inferred_type in ("integer", "float", "numeric")

    @property
    def has_nulls(self) -> bool:
        """Vérifie si la colonne contient des nulls."""
        return self.null_count > 0


class DataProfile(BaseModel):
    """Profil complet d'un dataset."""

    # Métadonnées
    dataset_id: str = Field(..., description="Identifiant unique du dataset")
    source: str = Field(..., description="Source du dataset (fichier, API, etc.)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de création du profil"
    )

    # Dimensions
    row_count: int = Field(..., ge=0, description="Nombre de lignes")
    column_count: int = Field(..., ge=0, description="Nombre de colonnes")
    memory_size_bytes: int = Field(..., ge=0, description="Taille en mémoire")

    # Profils des colonnes
    columns: list[ColumnProfile] = Field(
        default_factory=list,
        description="Profils de chaque colonne"
    )

    # Signature pour détection de drift
    data_hash: str = Field(..., description="Hash du dataset pour comparaison")

    @property
    def column_names(self) -> list[str]:
        """Liste des noms de colonnes."""
        return [col.name for col in self.columns]

    @property
    def total_null_count(self) -> int:
        """Nombre total de valeurs nulles."""
        return sum(col.null_count for col in self.columns)

    def get_column(self, name: str) -> ColumnProfile | None:
        """Récupère le profil d'une colonne par son nom."""
        for col in self.columns:
            if col.name == name:
                return col
        return None


class QualityIssue(BaseModel):
    """Problème de qualité détecté."""

    issue_id: str = Field(..., description="Identifiant unique du problème")
    issue_type: IssueType = Field(..., description="Type de problème")
    severity: Severity = Field(..., description="Niveau de sévérité")

    # Localisation
    column: str | None = Field(default=None, description="Colonne concernée")
    row_indices: list[int] = Field(
        default_factory=list,
        description="Indices des lignes concernées"
    )

    # Description
    description: str = Field(..., description="Description du problème")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Détails techniques"
    )

    # Métriques
    affected_count: int = Field(..., ge=0, description="Nombre d'éléments affectés")
    affected_percentage: float = Field(
        ..., ge=0, le=100,
        description="% d'éléments affectés"
    )
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Niveau de confiance dans la détection"
    )

    # Traçabilité
    detected_by: AgentType = Field(..., description="Agent ayant détecté le problème")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de détection"
    )

    @property
    def needs_escalation(self) -> bool:
        """Vérifie si le problème nécessite une escalade humaine."""
        return self.severity == Severity.CRITICAL or self.confidence < 0.7


class CorrectionProposal(BaseModel):
    """Proposition de correction pour un problème."""

    proposal_id: str = Field(..., description="Identifiant unique de la proposition")
    issue_id: str = Field(..., description="ID du problème corrigé")
    correction_type: CorrectionType = Field(..., description="Type de correction")

    # Description
    description: str = Field(..., description="Description de la correction")
    justification: str = Field(..., description="Justification de ce choix")

    # Paramètres
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Paramètres de la correction"
    )

    # Impact estimé
    estimated_impact: str = Field(..., description="Impact estimé sur les données")
    rows_affected: int = Field(..., ge=0, description="Nombre de lignes impactées")

    # Confiance
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Confiance dans cette proposition"
    )

    # Alternatives
    alternatives: list[str] = Field(
        default_factory=list,
        description="Autres approches possibles"
    )

    # Traçabilité
    proposed_by: AgentType = Field(
        default=AgentType.CORRECTOR,
        description="Agent ayant proposé la correction"
    )
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de proposition"
    )

    # Statut
    is_approved: bool | None = Field(
        default=None,
        description="Approbation (None = en attente)"
    )
    approved_by: str | None = Field(
        default=None,
        description="Validateur (agent ou utilisateur)"
    )


class ValidationResult(BaseModel):
    """Résultat de validation d'une correction."""

    validation_id: str = Field(..., description="Identifiant unique")
    proposal_id: str = Field(..., description="ID de la proposition validée")

    # Décision
    is_valid: bool = Field(..., description="La correction est-elle valide?")
    validation_status: str = Field(..., description="Statut détaillé")

    # Justification
    reasons: list[str] = Field(
        default_factory=list,
        description="Raisons de la décision"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Avertissements éventuels"
    )

    # Règles vérifiées
    rules_checked: list[str] = Field(
        default_factory=list,
        description="Règles métier vérifiées"
    )
    rules_passed: list[str] = Field(
        default_factory=list,
        description="Règles respectées"
    )
    rules_failed: list[str] = Field(
        default_factory=list,
        description="Règles violées"
    )

    # Traçabilité
    validated_by: AgentType = Field(
        default=AgentType.VALIDATOR,
        description="Agent validateur"
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de validation"
    )


# =============================================================================
# AGENT MODELS - Structures pour les agents
# =============================================================================


class AgentContext(BaseModel):
    """Contexte partagé entre agents."""

    session_id: str = Field(..., description="ID de la session d'analyse")
    dataset_id: str = Field(..., description="ID du dataset analysé")

    # État courant
    current_step: str = Field(default="init", description="Étape courante")
    iteration: int = Field(default=0, ge=0, description="Numéro d'itération")

    # Données accumulées
    profile: DataProfile | None = Field(
        default=None,
        description="Profil du dataset"
    )
    issues: list[QualityIssue] = Field(
        default_factory=list,
        description="Problèmes détectés"
    )
    proposals: list[CorrectionProposal] = Field(
        default_factory=list,
        description="Corrections proposées"
    )
    validations: list[ValidationResult] = Field(
        default_factory=list,
        description="Résultats de validation"
    )

    # Métadonnées
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées additionnelles"
    )

    def add_issue(self, issue: QualityIssue) -> None:
        """Ajoute un problème au contexte."""
        self.issues.append(issue)

    def add_proposal(self, proposal: CorrectionProposal) -> None:
        """Ajoute une proposition au contexte."""
        self.proposals.append(proposal)


class AgentDecision(BaseModel):
    """Décision prise par un agent (pour logging et apprentissage)."""

    decision_id: str = Field(..., description="Identifiant unique")
    agent_type: AgentType = Field(..., description="Agent ayant pris la décision")
    session_id: str = Field(..., description="ID de la session")

    # Décision
    action: str = Field(..., description="Action décidée")
    reasoning: str = Field(..., description="Raisonnement derrière la décision")

    # Inputs
    input_summary: str = Field(..., description="Résumé des données d'entrée")

    # Outputs
    output_summary: str = Field(..., description="Résumé du résultat")

    # Métriques
    confidence: float = Field(..., ge=0, le=1, description="Confiance")
    processing_time_ms: int = Field(..., ge=0, description="Temps de traitement")

    # Traçabilité
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date de la décision"
    )

    # Feedback (rempli ultérieurement)
    was_correct: bool | None = Field(
        default=None,
        description="La décision était-elle correcte?"
    )
    user_feedback: str | None = Field(
        default=None,
        description="Feedback utilisateur"
    )


# =============================================================================
# API MODELS - Requêtes et réponses
# =============================================================================


class AnalysisRequest(BaseModel):
    """Requête d'analyse de dataset."""

    # Source des données
    file_path: str | None = Field(
        default=None,
        description="Chemin vers le fichier"
    )
    data_json: dict[str, list] | None = Field(
        default=None,
        description="Données au format JSON"
    )

    # Options d'analyse
    include_profiling: bool = Field(
        default=True,
        description="Inclure le profiling"
    )
    detect_anomalies: bool = Field(
        default=True,
        description="Détecter les anomalies"
    )
    detect_drift: bool = Field(
        default=False,
        description="Détecter le drift"
    )
    reference_profile_id: str | None = Field(
        default=None,
        description="ID du profil de référence pour le drift"
    )

    # Règles personnalisées
    custom_rules: list[str] = Field(
        default_factory=list,
        description="Règles métier personnalisées"
    )

    @field_validator("file_path", "data_json")
    @classmethod
    def validate_source(cls, v: Any, info) -> Any:
        """Au moins une source de données doit être fournie."""
        # La validation croisée se fait au niveau du endpoint
        return v


class AnalysisResponse(BaseModel):
    """Réponse d'analyse complète."""

    # Identifiants
    session_id: str = Field(..., description="ID de la session")
    dataset_id: str = Field(..., description="ID du dataset")

    # Statut
    status: TaskStatus = Field(..., description="Statut de l'analyse")
    processing_time_ms: int = Field(..., ge=0, description="Temps total")

    # Résultats
    profile: DataProfile | None = Field(
        default=None,
        description="Profil du dataset"
    )
    issues: list[QualityIssue] = Field(
        default_factory=list,
        description="Problèmes détectés"
    )
    summary: str = Field(..., description="Résumé textuel de l'analyse")

    # Métriques globales
    quality_score: float = Field(
        ..., ge=0, le=100,
        description="Score de qualité global (0-100)"
    )
    issues_by_severity: dict[str, int] = Field(
        default_factory=dict,
        description="Nombre de problèmes par sévérité"
    )

    # Escalade
    needs_human_review: bool = Field(
        default=False,
        description="Nécessite une revue humaine"
    )
    escalation_reasons: list[str] = Field(
        default_factory=list,
        description="Raisons de l'escalade"
    )


class RecommendationResponse(BaseModel):
    """Réponse avec recommandations de corrections."""

    session_id: str = Field(..., description="ID de la session")
    proposals: list[CorrectionProposal] = Field(
        default_factory=list,
        description="Corrections proposées"
    )
    summary: str = Field(..., description="Résumé des recommandations")
    estimated_improvement: float = Field(
        ..., ge=0, le=100,
        description="Amélioration estimée du score de qualité"
    )


class ExplanationResponse(BaseModel):
    """Réponse avec explications détaillées."""

    session_id: str = Field(..., description="ID de la session")
    target_id: str = Field(..., description="ID de l'élément expliqué")
    target_type: str = Field(..., description="Type (issue, proposal, decision)")
    explanation: str = Field(..., description="Explication détaillée")
    contributing_factors: list[str] = Field(
        default_factory=list,
        description="Facteurs contributifs"
    )
    confidence_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Décomposition de la confiance"
    )
    related_rules: list[str] = Field(
        default_factory=list,
        description="Règles métier liées"
    )
    similar_past_decisions: list[str] = Field(
        default_factory=list,
        description="Décisions passées similaires"
    )


class FeedbackRequest(BaseModel):
    """Requête de feedback utilisateur."""

    session_id: str = Field(..., description="ID de la session")
    target_id: str = Field(..., description="ID de l'élément concerné")
    target_type: str = Field(
        ...,
        description="Type (issue, proposal, decision)"
    )
    is_correct: bool = Field(..., description="La décision était-elle correcte?")
    user_correction: str | None = Field(
        default=None,
        description="Correction suggérée par l'utilisateur"
    )
    comments: str | None = Field(
        default=None,
        description="Commentaires additionnels"
    )


class FeedbackResponse(BaseModel):
    """Confirmation de prise en compte du feedback."""

    feedback_id: str = Field(..., description="ID du feedback enregistré")
    status: str = Field(default="recorded", description="Statut")
    message: str = Field(..., description="Message de confirmation")
    impact: str = Field(
        ...,
        description="Comment ce feedback sera utilisé"
    )
