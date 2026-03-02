"""
Route /batch - Analyse en lot de plusieurs fichiers CSV/Parquet.

Permet d'envoyer jusqu'à 10 fichiers en une seule requête.
Chaque fichier est analysé indépendamment et en parallèle via asyncio.gather.
"""

import asyncio
import io
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, Request, UploadFile, status
from fastapi.responses import JSONResponse

from src.agents.orchestrator import OrchestratorAgent, TaskType
from src.api.limiter import limiter
from src.api.schemas.responses import BatchAnalyzeResponse, BatchResultItem, ErrorResponse
from src.core.config import settings
from src.core.models import AgentContext
from src.memory.session_store import get_session_store

router = APIRouter(prefix="/batch", tags=["Batch"])

_ALLOWED_EXTENSIONS = {".csv", ".parquet"}
_MAX_FILES = 10


def _read_file(filename: str, content: bytes) -> pd.DataFrame:
    """Lit un fichier CSV ou Parquet en DataFrame."""
    name = (filename or "").lower()

    if name.endswith(".csv"):
        attempts = [
            {},
            {"sep": None, "engine": "python"},
            {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
            {"sep": None, "engine": "python", "encoding": "latin-1"},
        ]
        for kwargs in attempts:
            try:
                return pd.read_csv(io.BytesIO(content), **kwargs)
            except Exception:
                pass
        raise ValueError(f"Impossible de parser le CSV '{filename}'")

    elif name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))

    else:
        ext = "." + name.rsplit(".", 1)[-1] if "." in name else "(aucune)"
        raise ValueError(
            f"Extension '{ext}' non supportée. Formats acceptés : .csv, .parquet"
        )


async def _analyze_one(filename: str, content: bytes) -> BatchResultItem:
    """Analyse un fichier unique et retourne un BatchResultItem."""
    try:
        df = _read_file(filename, content)
    except Exception as e:
        return BatchResultItem(filename=filename, status="error", error=str(e))

    if df.empty:
        return BatchResultItem(
            filename=filename,
            status="error",
            error="Fichier vide (aucune ligne de données)",
        )

    # Troncature si nécessaire
    if settings.max_rows_analyze > 0 and len(df) > settings.max_rows_analyze:
        df = df.head(settings.max_rows_analyze)

    try:
        orchestrator = OrchestratorAgent()
        context = AgentContext(
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}",
        )
        context = await orchestrator.run_pipeline_async(
            context, df, task_type=TaskType.ANALYZE
        )
    except Exception as e:
        return BatchResultItem(filename=filename, status="error", error=str(e))

    # Persistance best-effort
    try:
        store = get_session_store()
        store.save(context.session_id, context)
        store.save_dataframe(context.session_id, df)
    except Exception:
        pass

    return BatchResultItem(
        filename=filename,
        session_id=context.session_id,
        status="success",
        quality_score=context.metadata.get("quality_score", 100),
        issues_count=len(context.issues),
    )


@router.post(
    "",
    response_model=BatchAnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Aucun fichier ou trop de fichiers"},
        429: {"description": "Trop de requêtes"},
    },
    summary="Analyser plusieurs fichiers en lot",
    description="""
    Envoie jusqu'à **10 fichiers** CSV ou Parquet pour analyse simultanée.

    Tous les fichiers sont analysés **en parallèle** (asyncio.gather).
    Chaque résultat contient le score de qualité, le nombre de problèmes
    et le `session_id` pour accéder aux détails via `GET /analyze/{session_id}`.

    En cas d'erreur sur un fichier, les autres continuent leur analyse
    (pas d'arrêt global).

    **Rate limit** : 5 requêtes / minute par IP.
    """,
)
@limiter.limit("5/minute")
async def batch_analyze(
    request: Request,
    files: list[UploadFile],
) -> BatchAnalyzeResponse:
    """Analyse plusieurs fichiers en parallèle."""
    if not files:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun fichier fourni.",
        )

    if len(files) > _MAX_FILES:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {_MAX_FILES} fichiers par batch (reçu : {len(files)}).",
        )

    # Lecture de tous les contenus (await nécessaire)
    file_data: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        if len(content) > settings.max_upload_size:
            file_data.append((f.filename or "unknown", b""))  # sera géré comme erreur
        else:
            file_data.append((f.filename or "unknown", content))

    # Analyse parallèle
    tasks = [_analyze_one(fname, content) for fname, content in file_data]
    results: list[BatchResultItem] = await asyncio.gather(*tasks)

    succeeded = sum(1 for r in results if r.status == "success")
    failed = len(results) - succeeded

    return BatchAnalyzeResponse(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )
