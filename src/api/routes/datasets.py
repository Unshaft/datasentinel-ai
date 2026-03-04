"""
Route /datasets — Historique de qualité par dataset (v1.2 — F30).

Expose l'historique inter-sessions d'un dataset identifié par son fingerprint
(hash déterministe basé sur le schéma colonnes+types).
"""

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.responses import DatasetHistoryResponse, DatasetSessionInfo
from src.core.dataset_memory import get_dataset_memory_manager

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get(
    "/{dataset_id}/history",
    response_model=DatasetHistoryResponse,
    responses={
        404: {"description": "Dataset inconnu"},
    },
    summary="Historique de qualité d'un dataset",
    description="""
    Retourne l'historique des analyses pour un dataset identifié par son `dataset_id`.

    Le `dataset_id` est un fingerprint déterministe basé sur le schéma (colonnes + types).
    Il est retourné dans chaque `AnalyzeResponse` — deux fichiers avec le même schéma
    partagent le même `dataset_id`, même si leur contenu diffère.

    Inclut : score moyen, tendance, issues récurrentes, colonnes problématiques,
    suggestions de règles pro-actives, et les 10 dernières sessions.
    """,
)
def get_dataset_history(dataset_id: str) -> DatasetHistoryResponse:
    """Retourne l'historique complet d'un dataset."""
    history = get_dataset_memory_manager().get_history(dataset_id)

    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' inconnu — aucune session enregistrée",
        )

    sessions = [
        DatasetSessionInfo(
            session_id=s["session_id"],
            timestamp=s["timestamp"],
            quality_score=s["quality_score"],
            issue_counts=s.get("issue_counts", {}),
            top_columns=s.get("top_columns", []),
        )
        for s in history["sessions"]
    ]

    return DatasetHistoryResponse(
        dataset_id=history["dataset_id"],
        first_seen=history["first_seen"],
        last_seen=history["last_seen"],
        session_count=history["session_count"],
        avg_quality_score=history["avg_quality_score"],
        trend=history["trend"],
        recurring_issues=history["recurring_issues"],
        problematic_columns=history["problematic_columns"],
        suggested_rules=history["suggested_rules"],
        sessions=sessions,
    )
