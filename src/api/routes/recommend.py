"""
Route /recommend - Recommandations de corrections.

Endpoint pour obtenir des propositions de corrections
pour les problèmes de qualité détectés.
"""

import io
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from src.agents.orchestrator import OrchestratorAgent, TaskType
from src.api.schemas.requests import RecommendRequest
from src.api.schemas.responses import ErrorResponse, ProposalResponse, RecommendResponse
from src.core.exceptions import DataSentinelError
from src.core.models import AgentContext

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


@router.post(
    "",
    response_model=RecommendResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Données invalides"},
        500: {"model": ErrorResponse, "description": "Erreur serveur"}
    },
    summary="Obtenir des recommandations de corrections",
    description="""
    Analyse un dataset et propose des corrections pour les problèmes détectés.

    Pour chaque problème, le système propose:
    - Une ou plusieurs solutions possibles
    - Une justification pour chaque proposition
    - Une estimation de l'impact
    - Des alternatives à considérer

    Chaque proposition inclut un score de confiance.
    """
)
async def get_recommendations(request: RecommendRequest) -> RecommendResponse:
    """
    Endpoint de recommandation de corrections.

    Args:
        request: Données et options

    Returns:
        Liste de propositions de correction
    """
    try:
        # Charger les données
        if request.data:
            df = pd.DataFrame(request.data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Données requises (data)"
            )

        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le dataset est vide"
            )

        # Créer l'orchestrateur
        orchestrator = OrchestratorAgent()

        context = AgentContext(
            session_id=request.session_id or f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}"
        )

        # Exécuter le pipeline avec recommandations
        context = orchestrator.run_pipeline(
            context, df,
            task_type=TaskType.RECOMMEND
        )

        # Calculer l'amélioration estimée
        current_score = context.metadata.get("quality_score", 100)
        # Estimation simple: chaque correction approuvée améliore de X points
        approved_count = sum(1 for p in context.proposals if p.is_approved is not False)
        estimated_improvement = min(
            100 - current_score,
            approved_count * 5  # ~5 points par correction
        )

        # Formater les propositions
        proposals = [
            ProposalResponse(
                proposal_id=p.proposal_id,
                issue_id=p.issue_id,
                correction_type=p.correction_type.value,
                description=p.description,
                justification=p.justification,
                estimated_impact=p.estimated_impact,
                rows_affected=p.rows_affected,
                confidence=p.confidence,
                alternatives=p.alternatives,
                is_approved=p.is_approved
            )
            for p in context.proposals
        ]

        return RecommendResponse(
            session_id=context.session_id,
            status=context.metadata.get("final_status", "completed"),
            quality_score=current_score,
            issues_count=len(context.issues),
            proposals=proposals,
            estimated_improvement=round(estimated_improvement, 1),
            summary=context.metadata.get("summary", "")
        )

    except HTTPException:
        raise
    except DataSentinelError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération des recommandations: {str(e)}"
        )


@router.post(
    "/validate",
    response_model=RecommendResponse,
    summary="Recommandations avec validation",
    description="""
    Comme /recommend mais inclut également la validation
    des corrections proposées contre les règles métier.
    """
)
async def get_validated_recommendations(request: RecommendRequest) -> RecommendResponse:
    """
    Recommandations avec étape de validation.
    """
    try:
        if not request.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Données requises"
            )

        df = pd.DataFrame(request.data)

        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le dataset est vide"
            )

        orchestrator = OrchestratorAgent()

        context = AgentContext(
            session_id=request.session_id or f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}"
        )

        # Pipeline complet avec validation
        context = orchestrator.run_pipeline(
            context, df,
            task_type=TaskType.FULL_PIPELINE
        )

        current_score = context.metadata.get("quality_score", 100)
        approved_count = sum(1 for p in context.proposals if p.is_approved)
        estimated_improvement = min(100 - current_score, approved_count * 5)

        proposals = [
            ProposalResponse(
                proposal_id=p.proposal_id,
                issue_id=p.issue_id,
                correction_type=p.correction_type.value,
                description=p.description,
                justification=p.justification,
                estimated_impact=p.estimated_impact,
                rows_affected=p.rows_affected,
                confidence=p.confidence,
                alternatives=p.alternatives,
                is_approved=p.is_approved
            )
            for p in context.proposals
        ]

        # Ajouter les informations de validation dans le summary
        validated_summary = context.metadata.get("summary", "")
        if context.validations:
            approved = sum(1 for v in context.validations if v.is_valid)
            validated_summary += f" | Validations: {approved}/{len(context.validations)} approuvées"

        return RecommendResponse(
            session_id=context.session_id,
            status=context.metadata.get("final_status", "completed"),
            quality_score=current_score,
            issues_count=len(context.issues),
            proposals=proposals,
            estimated_improvement=round(estimated_improvement, 1),
            summary=validated_summary
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur: {str(e)}"
        )
