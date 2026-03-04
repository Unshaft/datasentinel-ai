"""
Route /upload - Upload de fichiers CSV ou Parquet.

Permet d'envoyer un fichier directement au lieu de passer les données en JSON.
Lecture via pandas, puis passage au pipeline d'analyse standard.
"""

import io
import uuid

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, status

from src.agents.orchestrator import OrchestratorAgent, TaskType
from src.api.limiter import limiter
from src.api.schemas.responses import (
    AnalyzeResponse,
    ColumnProfileResponse,
    DataProfileResponse,
    ErrorResponse,
    IssueResponse,
)
from src.core.config import settings
from src.core.exceptions import DataSentinelError
from src.core.models import AgentContext
from src.core.dataset_memory import compute_dataset_id, get_dataset_memory_manager
from src.core.stats_manager import get_stats_manager
from src.core.webhook_manager import fire_webhooks
from src.memory.session_store import get_session_store

router = APIRouter(prefix="/upload", tags=["Upload"])

_ALLOWED_EXTENSIONS = {".csv", ".parquet"}


def _read_uploaded_file(file: UploadFile, content: bytes) -> pd.DataFrame:
    """
    Lit le fichier uploadé en DataFrame.

    Args:
        file: Metadata du fichier (nom, content-type)
        content: Contenu brut du fichier

    Returns:
        DataFrame pandas

    Raises:
        HTTPException 422 si extension non supportée
        HTTPException 400 si parsing échoue
    """
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        # Tentatives successives : séparateur auto + encodages courants (FR)
        attempts = [
            {},                                              # virgule + utf-8
            {"sep": None, "engine": "python"},              # séparateur auto + utf-8
            {"sep": None, "engine": "python", "encoding": "utf-8-sig"},   # BOM Excel
            {"sep": None, "engine": "python", "encoding": "latin-1"},     # Excel FR
            {"sep": None, "engine": "python", "encoding": "cp1252"},      # Windows FR
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                return pd.read_csv(io.BytesIO(content), **kwargs)
            except Exception as e:
                last_error = e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impossible de parser le CSV : {last_error}"
        )

    elif filename.endswith(".parquet"):
        try:
            return pd.read_parquet(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Impossible de parser le Parquet : {e}"
            )

    else:
        ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else "(aucune)"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Extension '{ext}' non supportée. "
                f"Formats acceptés : {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            )
        )


def _format_profile(profile) -> DataProfileResponse | None:
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
            sample_values=col.sample_values[:5],
        )
        for col in profile.columns
    ]
    return DataProfileResponse(
        dataset_id=profile.dataset_id,
        row_count=profile.row_count,
        column_count=profile.column_count,
        memory_mb=round(profile.memory_size_bytes / 1024 / 1024, 2),
        total_null_count=profile.total_null_count,
        columns=columns,
    )


def _format_issues(issues: list) -> list[IssueResponse]:
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
            details=issue.details,
        )
        for issue in issues
    ]


@router.post(
    "",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Fichier vide ou illisible"},
        413: {"model": ErrorResponse, "description": "Fichier trop volumineux"},
        422: {"model": ErrorResponse, "description": "Extension non supportée"},
        500: {"model": ErrorResponse, "description": "Erreur serveur"},
    },
    summary="Uploader et analyser un fichier",
    description="""
    Envoie un fichier CSV ou Parquet pour analyse qualité.

    Le fichier est lu, passé au pipeline complet (Profiler → Quality checks en parallèle),
    et retourne les mêmes résultats que `/analyze`.

    **Formats supportés** : `.csv`, `.parquet`

    **Taille max** : 100 MB (configurable via `MAX_UPLOAD_SIZE`)

    **Rate limit** : 10 requêtes / minute par IP.
    """,
)
@limiter.limit("10/minute")
async def upload_and_analyze(
    request: Request,
    file: UploadFile,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """
    Upload un fichier et lance l'analyse de qualité.

    Args:
        file: Fichier CSV ou Parquet (multipart/form-data)

    Returns:
        Résultats d'analyse (même format que POST /analyze)
    """
    # --- Lecture du contenu ---
    content = await file.read()

    # Vérification taille
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Fichier trop volumineux : {len(content) / 1024 / 1024:.1f} MB. "
                f"Maximum : {settings.max_upload_size / 1024 / 1024:.0f} MB"
            ),
        )

    # --- Parsing ---
    df = _read_uploaded_file(file, content)

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier est vide (aucune ligne de données)"
        )

    # Troncature si nécessaire
    if settings.max_rows_analyze > 0 and len(df) > settings.max_rows_analyze:
        df = df.head(settings.max_rows_analyze)

    # --- Pipeline async — quality checks en parallèle ---
    try:
        orchestrator = OrchestratorAgent()
        dataset_id = compute_dataset_id(df)
        was_known_before = get_dataset_memory_manager().get_entry(dataset_id) is not None
        context = AgentContext(
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=dataset_id,
        )
        context = await orchestrator.run_pipeline_async(
            context, df, task_type=TaskType.ANALYZE
        )

    except DataSentinelError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse : {e}",
        )

    # --- Formatage ---
    issues_by_severity: dict[str, int] = {}
    for issue in context.issues:
        sev = issue.severity.value
        issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1

    escalation_reasons = []
    if context.metadata.get("needs_human_review"):
        if issues_by_severity.get("critical", 0) > 0:
            escalation_reasons.append(
                f"{issues_by_severity['critical']} problème(s) critique(s)"
            )

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
        semantic_types=context.metadata.get("semantic_types") or None,
        domain_agent=context.metadata.get("domain_name"),
        reflect_flags=context.metadata.get("reflect_flags", []),
        reasoning_steps=context.metadata.get("reasoning_steps", []),
        needs_human_review=context.metadata.get("needs_human_review", False),
        escalation_reasons=escalation_reasons,
        dataset_memory=None,  # rempli après record_session ci-dessous
    )

    # Persister la session + DataFrame (best-effort)
    try:
        store = get_session_store()
        store.save(context.session_id, context)
        store.save_dataframe(context.session_id, df)
    except Exception:
        pass

    # Stats (best-effort)
    try:
        issue_types = [iss.issue_type.value for iss in context.issues]
        get_stats_manager().record_session(
            context.metadata.get("quality_score", 100),
            issue_types,
        )
    except Exception:
        pass

    # Dataset memory (best-effort)
    try:
        mem = get_dataset_memory_manager()
        mem.record_session(
            dataset_id=context.dataset_id,
            session_id=context.session_id,
            quality_score=context.metadata.get("quality_score", 100),
            issues=context.issues,
        )
        mem_info = mem.get_memory_info(context.dataset_id, was_known_before)
        from src.api.schemas.responses import DatasetMemoryInfo
        response.dataset_memory = DatasetMemoryInfo(**mem_info)
    except Exception:
        pass

    # Webhooks en background (non-bloquant)
    background_tasks.add_task(
        fire_webhooks,
        "analysis.completed",
        {
            "session_id": context.session_id,
            "dataset_id": context.dataset_id,
            "quality_score": context.metadata.get("quality_score", 100),
            "status": context.metadata.get("final_status", "completed"),
            "issues_count": len(context.issues),
            "needs_human_review": context.metadata.get("needs_human_review", False),
        },
    )

    return response
