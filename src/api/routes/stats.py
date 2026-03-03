"""
Route /stats — Dashboard analytique agrégé (v0.6 — F22).

GET  /stats        → métriques d'utilisation
DELETE /stats      → remise à zéro des compteurs
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from src.api.limiter import limiter
from src.api.schemas.responses import StatsResponse
from src.core.stats_manager import get_stats_manager

router = APIRouter(prefix="/stats", tags=["Stats"])


@router.get("", response_model=StatsResponse, summary="Dashboard analytique")
@limiter.limit("60/minute")
async def get_stats(request: Request) -> StatsResponse:
    """Retourne les statistiques agrégées d'utilisation."""
    mgr = get_stats_manager()
    data = mgr.get_stats()
    return StatsResponse(
        total_sessions=data["total_sessions"],
        avg_quality_score=data["avg_quality_score"],
        top_issue_types=data["top_issue_types"],
        sessions_by_day=data["sessions_by_day"],
        score_distribution=data["score_distribution"],
        updated_at=datetime.now(timezone.utc),
    )


@router.delete("", summary="Remise à zéro des statistiques")
@limiter.limit("5/minute")
async def reset_stats(request: Request) -> dict:
    """Remet les statistiques à zéro (opération destructive)."""
    get_stats_manager().reset()
    return {"status": "reset", "message": "Statistiques remises à zéro."}
