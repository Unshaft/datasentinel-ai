"""
Route /analyze - Analyse de qualité des données.

Endpoint principal pour analyser un dataset et détecter
les problèmes de qualité.
"""

import io
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from src.agents.orchestrator import OrchestratorAgent, TaskType
from src.api.schemas.requests import AnalyzeRequest
from src.api.schemas.responses import (
    AnalyzeResponse,
    ColumnProfileResponse,
    DataProfileResponse,
    ErrorResponse,
    IssueResponse,
)
from src.core.exceptions import DataLoadError, DataSentinelError
from src.core.models import AgentContext

router = APIRouter(prefix="/analyze", tags=["Analysis"])


def _load_dataframe(request: AnalyzeRequest) -> pd.DataFrame:
    """
    Charge un DataFrame depuis la requête.

    Args:
        request: Requête d'analyse

    Returns:
        DataFrame chargé

    Raises:
        HTTPException: Si les données sont invalides
    """
    if request.data:
        try:
            return pd.DataFrame(request.data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Impossible de créer le DataFrame: {str(e)}"
            )

    elif request.file_content:
        try:
            return pd.read_csv(io.StringIO(request.file_content))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Impossible de parser le CSV: {str(e)}"
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucune source de données fournie (data ou file_content requis)"
        )


def _format_profile(profile) -> DataProfileResponse | None:
    """Formate le profil pour la réponse."""
    if profile is None:
        return None

    columns = [
        ColumnProfileResponse(
            name=col.name,
            dtype=col.dtype,
            inferred_type=col.inferred_type,
            count=col.count,
            null_count=col.null_count,
            null_percentage=col.null_percentage,
            unique_count=col.unique_count,
            unique_percentage=col.unique_percentage,
            mean=col.mean,
            std=col.std,
            min=col.min,
            max=col.max,
            sample_values=col.sample_values[:5]
        )
        for col in profile.columns
    ]

    return DataProfileResponse(
        dataset_id=profile.dataset_id,
        row_count=profile.row_count,
        column_count=profile.column_count,
        memory_mb=round(profile.memory_size_bytes / 1024 / 1024, 2),
        total_null_count=profile.total_null_count,
        columns=columns
    )


def _format_issues(issues: list) -> list[IssueResponse]:
    """Formate les issues pour la réponse."""
    return [
        IssueResponse(
            issue_id=issue.issue_id,
            issue_type=issue.issue_type.value,
            severity=issue.severity.value,
            column=issue.column,
            description=issue.description,
            affected_count=issue.affected_count,
            affected_percentage=issue.affected_percentage,
            confidence=issue.confidence,
            details=issue.details
        )
        for issue in issues
    ]


@router.post(
    "",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Données invalides"},
        500: {"model": ErrorResponse, "description": "Erreur serveur"}
    },
    summary="Analyser un dataset",
    description="""
    Analyse un dataset pour détecter les problèmes de qualité.

    Le système effectue:
    - Profilage statistique complet
    - Détection des valeurs manquantes
    - Détection des anomalies (optionnel)
    - Validation contre les règles métier

    Retourne un score de qualité global et la liste des problèmes détectés.
    """
)
async def analyze_dataset(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Endpoint principal d'analyse de données.

    Args:
        request: Données et options d'analyse

    Returns:
        Résultats d'analyse complets
    """
    try:
        # Charger les données
        df = _load_dataframe(request)

        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Le dataset est vide"
            )

        # Créer l'orchestrateur et exécuter
        orchestrator = OrchestratorAgent()

        context = AgentContext(
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}"
        )

        # Options
        options = {
            "detect_anomalies": request.detect_anomalies,
            "detect_drift": request.detect_drift,
        }

        # Ajouter les règles personnalisées si fournies
        if request.custom_rules:
            from src.memory.chroma_store import get_chroma_store
            store = get_chroma_store()
            for i, rule_text in enumerate(request.custom_rules):
                store.add_rule(
                    rule_id=f"custom_{context.session_id}_{i}",
                    rule_text=rule_text,
                    rule_type="custom"
                )

        # Exécuter l'analyse
        context = orchestrator.run_pipeline(
            context, df,
            task_type=TaskType.ANALYZE,
            **options
        )

        # Formater la réponse
        issues_by_severity = {}
        for issue in context.issues:
            sev = issue.severity.value
            issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1

        escalation_reasons = []
        if context.metadata.get("needs_human_review"):
            critical_count = issues_by_severity.get("critical", 0)
            if critical_count > 0:
                escalation_reasons.append(f"{critical_count} problème(s) critique(s)")

            low_confidence = [
                i for i in context.issues if i.confidence < 0.5
            ]
            if low_confidence:
                escalation_reasons.append(
                    f"{len(low_confidence)} détection(s) à faible confiance"
                )

        return AnalyzeResponse(
            session_id=context.session_id,
            dataset_id=context.dataset_id,
            status=context.metadata.get("final_status", "completed"),
            quality_score=context.metadata.get("quality_score", 100),
            processing_time_ms=context.metadata.get("processing_time_ms", 0),
            summary=context.metadata.get("summary", ""),
            profile=_format_profile(context.profile),
            issues=_format_issues(context.issues),
            issues_by_severity=issues_by_severity,
            needs_human_review=context.metadata.get("needs_human_review", False),
            escalation_reasons=escalation_reasons
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
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )


@router.get(
    "/{session_id}",
    response_model=AnalyzeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Session non trouvée"}
    },
    summary="Récupérer les résultats d'une analyse",
    description="Récupère les résultats d'une analyse précédente par son ID de session."
)
async def get_analysis_results(session_id: str) -> AnalyzeResponse:
    """
    Récupère les résultats d'une session existante.

    Note: Cette implémentation est simplifiée.
    En production, les sessions seraient persistées.
    """
    # TODO: Implémenter la persistance des sessions
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session '{session_id}' non trouvée. La persistance des sessions n'est pas encore implémentée."
    )
