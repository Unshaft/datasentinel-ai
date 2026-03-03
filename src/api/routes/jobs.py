"""
Route /jobs — Analyse asynchrone pour fichiers volumineux (v0.6 — F21).

POST /jobs/analyze          → Soumet un fichier, retourne HTTP 202 + job_id
GET  /jobs/{job_id}         → Consulte l'état du job (pending/running/completed/failed)

Utile pour des fichiers > 50k lignes où le pipeline dépasse le timeout HTTP.
"""

import asyncio
import io
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from src.agents.orchestrator import OrchestratorAgent, TaskType
from src.api.limiter import limiter
from src.api.routes.upload import _format_issues, _format_profile, _read_uploaded_file
from src.api.schemas.responses import (
    AnalyzeResponse,
    JobCreateResponse,
    JobStatusResponse,
)
from src.core.config import settings
from src.core.job_manager import get_job_manager
from src.core.models import AgentContext
from src.memory.session_store import get_session_store

router = APIRouter(prefix="/jobs", tags=["Jobs"])

_ALLOWED_EXTENSIONS = {".csv", ".parquet"}


async def _run_job(job_id: str, filename: str, content: bytes) -> None:
    """Exécute le pipeline d'analyse pour un job en arrière-plan."""
    mgr = get_job_manager()
    mgr.update_job(job_id, status="running", progress=10.0)

    try:
        # Parsing du fichier
        class _FakeFile:
            def __init__(self, name: str):
                self.filename = name

        df = _read_uploaded_file(_FakeFile(filename), content)  # type: ignore[arg-type]

        if df.empty:
            mgr.update_job(job_id, status="failed", error="Le fichier est vide.")
            return

        if settings.max_rows_analyze > 0 and len(df) > settings.max_rows_analyze:
            df = df.head(settings.max_rows_analyze)

        mgr.update_job(job_id, status="running", progress=30.0)

        # Pipeline d'analyse
        orchestrator = OrchestratorAgent()
        context = AgentContext(
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}",
        )
        context = await orchestrator.run_pipeline_async(
            context, df, task_type=TaskType.ANALYZE
        )

        mgr.update_job(job_id, status="running", progress=80.0)

        # Persister la session + DataFrame
        try:
            store = get_session_store()
            store.save(context.session_id, context)
            store.save_dataframe(context.session_id, df)
        except Exception:
            pass

        # Formatage
        issues_by_severity: dict[str, int] = {}
        for issue in context.issues:
            sev = issue.severity.value
            issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1

        response = AnalyzeResponse(
            session_id=context.session_id,
            dataset_id=context.dataset_id,
            status=context.metadata.get("final_status", "completed"),
            quality_score=context.metadata.get("quality_score", 100),
            processing_time_ms=context.metadata.get("processing_time_ms", 0),
            summary=context.metadata.get("summary", ""),
            profile=_format_profile(context.profile),
            issues=_format_issues(context.issues),
            issues_by_severity=issues_by_severity,
            column_scores=context.metadata.get("column_scores", {}),
            needs_human_review=context.metadata.get("needs_human_review", False),
            escalation_reasons=[],
        )

        mgr.update_job(
            job_id,
            status="completed",
            progress=100.0,
            result=response.model_dump(mode="json"),
        )

    except Exception as exc:
        mgr.update_job(job_id, status="failed", error=str(exc))


@router.post(
    "/analyze",
    response_model=JobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Soumettre un fichier pour analyse asynchrone",
    description="""
    Accepte un fichier CSV ou Parquet et lance l'analyse en arrière-plan.
    Retourne immédiatement un `job_id` avec le statut `pending`.

    Interroger `GET /jobs/{job_id}` pour suivre la progression.

    **Rate limit** : 5 requêtes / minute.
    """,
)
@limiter.limit("5/minute")
async def submit_job(request: Request, file: UploadFile) -> JSONResponse:
    """Soumet un fichier pour analyse asynchrone."""
    content = await file.read()

    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux: {len(content) / 1024 / 1024:.1f} MB",
        )

    filename = file.filename or "upload"
    mgr = get_job_manager()
    job_id = mgr.create_job(filename)

    # Lance la tâche en arrière-plan (non bloquant)
    asyncio.create_task(_run_job(job_id, filename, content))

    resp = JobCreateResponse(job_id=job_id)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=resp.model_dump(mode="json"),
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Consulter l'état d'un job",
)
@limiter.limit("60/minute")
async def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
    """Retourne l'état courant d'un job asynchrone."""
    mgr = get_job_manager()
    job = mgr.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job introuvable ou expiré: {job_id}",
        )

    result = None
    if job.get("result"):
        result = AnalyzeResponse(**job["result"])

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress", 0.0),
        result=result,
        error=job.get("error"),
        created_at=job.get("created_at"),
    )
