"""
Schémas Pydantic pour les requêtes API.

Ces schémas définissent et valident les données entrantes
pour chaque endpoint de l'API.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    """Requête d'analyse de dataset."""

    # Source des données (une des deux est requise)
    data: dict[str, list[Any]] | None = Field(
        default=None,
        description="Données au format JSON (colonnes: valeurs)"
    )
    file_content: str | None = Field(
        default=None,
        description="Contenu CSV encodé en string"
    )

    # Options d'analyse
    detect_anomalies: bool = Field(
        default=True,
        description="Activer la détection d'anomalies"
    )
    detect_drift: bool = Field(
        default=False,
        description="Activer la détection de drift"
    )
    reference_session_id: str | None = Field(
        default=None,
        description="ID de session de référence pour le drift"
    )

    # Règles personnalisées
    custom_rules: list[str] = Field(
        default_factory=list,
        description="Règles métier personnalisées à ajouter"
    )

    # Raisonnement adaptatif (v0.7)
    include_reasoning: bool = Field(
        default=False,
        description="Utiliser l'orchestrateur adaptatif ReAct et inclure les étapes de raisonnement"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "id": [1, 2, 3, 4, 5],
                    "name": ["Alice", "Bob", None, "Diana", "Eve"],
                    "age": [25, 30, 35, 200, 28],
                    "salary": [50000, 60000, 55000, 70000, None]
                },
                "detect_anomalies": True,
                "detect_drift": False
            }
        }

    @field_validator("data", "file_content")
    @classmethod
    def check_data_source(cls, v: Any, info) -> Any:
        """Validation de base des sources."""
        return v


class RecommendRequest(BaseModel):
    """Requête de recommandation de corrections."""

    # Soit session existante, soit nouvelles données
    session_id: str | None = Field(
        default=None,
        description="ID de session existante à compléter"
    )
    data: dict[str, list[Any]] | None = Field(
        default=None,
        description="Nouvelles données à analyser"
    )

    # Options
    include_analysis: bool = Field(
        default=True,
        description="Inclure l'analyse si nouvelles données"
    )
    max_proposals_per_issue: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre max de propositions par problème"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "product_id": [1, 1, 2, 3],
                    "price": [-10, 50, None, 100]
                },
                "max_proposals_per_issue": 2
            }
        }


class ExplainRequest(BaseModel):
    """Requête d'explication d'une décision."""

    session_id: str = Field(
        ...,
        description="ID de la session"
    )
    target_id: str = Field(
        ...,
        description="ID de l'élément à expliquer (issue, proposal, decision)"
    )
    target_type: str = Field(
        ...,
        pattern="^(issue|proposal|decision|validation)$",
        description="Type d'élément: issue, proposal, decision, validation"
    )
    detail_level: str = Field(
        default="normal",
        pattern="^(brief|normal|detailed)$",
        description="Niveau de détail: brief, normal, detailed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "target_id": "issue_def456",
                "target_type": "issue",
                "detail_level": "detailed"
            }
        }


class FeedbackRequest(BaseModel):
    """Requête de feedback utilisateur."""

    session_id: str = Field(
        ...,
        description="ID de la session"
    )
    target_id: str = Field(
        ...,
        description="ID de l'élément concerné"
    )
    target_type: str = Field(
        ...,
        pattern="^(issue|proposal|decision)$",
        description="Type: issue, proposal, decision"
    )
    is_correct: bool = Field(
        ...,
        description="La décision/détection était-elle correcte?"
    )
    user_correction: str | None = Field(
        default=None,
        max_length=1000,
        description="Correction suggérée par l'utilisateur"
    )
    comments: str | None = Field(
        default=None,
        max_length=2000,
        description="Commentaires additionnels"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "target_id": "proposal_xyz789",
                "target_type": "proposal",
                "is_correct": False,
                "user_correction": "La médiane serait plus appropriée que la moyenne",
                "comments": "Les données contiennent des outliers"
            }
        }


class AddRuleRequest(BaseModel):
    """Requête d'ajout de règle métier."""

    rule_text: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Texte de la règle en langage naturel"
    )
    rule_type: str = Field(
        default="constraint",
        pattern="^(constraint|validation|format|consistency)$",
        description="Type de règle"
    )
    severity: str = Field(
        default="medium",
        pattern="^(low|medium|high|critical)$",
        description="Sévérité des violations"
    )
    category: str = Field(
        default="general",
        description="Catégorie de la règle"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rule_text": "Le champ email doit contenir exactement un caractère @",
                "rule_type": "format",
                "severity": "medium",
                "category": "validity"
            }
        }


class HealthCheckRequest(BaseModel):
    """Requête de vérification de santé (optionnelle)."""

    include_details: bool = Field(
        default=False,
        description="Inclure les détails des composants"
    )
