"""
Route /feedback - Feedback utilisateur.

Endpoint pour enregistrer les retours utilisateur sur
les décisions du système, permettant l'amélioration continue.
"""

import uuid

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.requests import AddRuleRequest, FeedbackRequest
from src.api.schemas.responses import (
    AddRuleResponse,
    ErrorResponse,
    FeedbackResponse,
    RuleResponse,
)
from src.core.models import FeedbackRequest as FeedbackRequestModel
from src.memory.chroma_store import get_chroma_store
from src.memory.feedback_store import get_feedback_store

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Requête invalide"}
    },
    summary="Soumettre un feedback",
    description="""
    Enregistre un feedback utilisateur sur une décision du système.

    Le feedback permet:
    - De corriger les erreurs du système
    - D'améliorer les futures décisions similaires
    - D'ajuster les scores de confiance

    Types de cibles supportés:
    - `issue`: Feedback sur une détection de problème
    - `proposal`: Feedback sur une proposition de correction
    - `decision`: Feedback sur une décision d'agent
    """
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Enregistre un feedback utilisateur.

    Args:
        request: Détails du feedback

    Returns:
        Confirmation avec impact estimé
    """
    try:
        feedback_store = get_feedback_store()

        # Convertir en modèle interne
        feedback_model = FeedbackRequestModel(
            session_id=request.session_id,
            target_id=request.target_id,
            target_type=request.target_type,
            is_correct=request.is_correct,
            user_correction=request.user_correction,
            comments=request.comments
        )

        # Enregistrer le feedback
        response = feedback_store.record_feedback(feedback_model)

        return FeedbackResponse(
            feedback_id=response.feedback_id,
            status=response.status,
            message=response.message,
            impact=response.impact
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'enregistrement du feedback: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Statistiques des feedbacks",
    description="Retourne des statistiques sur les feedbacks reçus."
)
async def get_feedback_stats() -> dict:
    """
    Récupère les statistiques de feedback.

    Returns:
        Statistiques agrégées
    """
    try:
        feedback_store = get_feedback_store()
        stats = feedback_store.get_feedback_stats()

        return {
            "status": "success",
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur: {str(e)}"
        )


# =============================================================================
# ROUTES POUR LES RÈGLES MÉTIER
# =============================================================================


@router.post(
    "/rules",
    response_model=AddRuleResponse,
    summary="Ajouter une règle métier",
    description="""
    Ajoute une nouvelle règle métier personnalisée au système.

    La règle sera utilisée pour:
    - Valider les données lors des analyses
    - Vérifier les corrections proposées
    - Enrichir les explications

    Types de règles:
    - `constraint`: Contrainte sur les valeurs
    - `validation`: Règle de validation
    - `format`: Règle de format
    - `consistency`: Règle de cohérence
    """
)
async def add_business_rule(request: AddRuleRequest) -> AddRuleResponse:
    """
    Ajoute une règle métier.

    Args:
        request: Définition de la règle

    Returns:
        Confirmation avec détails de la règle
    """
    try:
        store = get_chroma_store()

        rule_id = f"rule_user_{uuid.uuid4().hex[:8]}"

        store.add_rule(
            rule_id=rule_id,
            rule_text=request.rule_text,
            rule_type=request.rule_type,
            metadata={
                "severity": request.severity,
                "category": request.category,
                "source": "user_api"
            }
        )

        return AddRuleResponse(
            status="success",
            message="Règle ajoutée avec succès",
            rule=RuleResponse(
                rule_id=rule_id,
                text=request.rule_text,
                rule_type=request.rule_type,
                severity=request.severity,
                category=request.category,
                active=True
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'ajout de la règle: {str(e)}"
        )


@router.get(
    "/rules",
    summary="Lister les règles métier",
    description="Retourne la liste des règles métier actives."
)
async def list_business_rules(rule_type: str | None = None) -> dict:
    """
    Liste les règles métier.

    Args:
        rule_type: Filtrer par type (optionnel)

    Returns:
        Liste des règles
    """
    try:
        store = get_chroma_store()
        rules = store.get_all_rules(rule_type=rule_type)

        return {
            "status": "success",
            "count": len(rules),
            "rules": [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "type": r["metadata"].get("rule_type"),
                    "severity": r["metadata"].get("severity"),
                    "category": r["metadata"].get("category")
                }
                for r in rules
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur: {str(e)}"
        )


@router.delete(
    "/rules/{rule_id}",
    summary="Désactiver une règle",
    description="Désactive une règle métier (soft delete)."
)
async def deactivate_rule(rule_id: str) -> dict:
    """
    Désactive une règle.

    Args:
        rule_id: ID de la règle

    Returns:
        Confirmation
    """
    try:
        store = get_chroma_store()
        store.deactivate_rule(rule_id)

        return {
            "status": "success",
            "message": f"Règle '{rule_id}' désactivée"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur: {str(e)}"
        )
