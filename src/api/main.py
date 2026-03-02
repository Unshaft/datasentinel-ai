"""
Point d'entrée de l'API FastAPI pour DataSentinel AI.

Ce module configure et lance l'application FastAPI avec:
- Les routes principales (/analyze, /recommend, /explain, /feedback)
- La gestion des erreurs
- La documentation OpenAPI
- Le CORS
- Le health check
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.auth import get_current_user
from src.api.limiter import limiter
from src.api.routes import analyze, batch, explain, feedback, recommend, upload
from src.api.routes import auth as auth_router
from src.api.routes import webhooks as webhooks_router
from src.api.schemas.responses import ErrorResponse, HealthResponse
from src.core.config import settings
from src.core.exceptions import DataSentinelError
from src.memory.chroma_store import get_chroma_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie de l'application.

    Startup:
    - Initialise ChromaDB
    - Charge les règles par défaut

    Shutdown:
    - Nettoyage des ressources
    """
    # === STARTUP ===
    print("🚀 Démarrage de DataSentinel AI...")

    # Initialiser ChromaDB
    try:
        store = get_chroma_store()
        print(f"✅ ChromaDB initialisé: {store.get_stats()}")

        # Charger les règles par défaut si collection vide
        if store.rules_collection.count() == 0:
            await _load_default_rules(store)

    except Exception as e:
        print(f"⚠️ Erreur initialisation ChromaDB: {e}")

    print("✅ DataSentinel AI prêt!")

    yield

    # === SHUTDOWN ===
    print("👋 Arrêt de DataSentinel AI...")


async def _load_default_rules(store):
    """Charge les règles par défaut depuis le fichier JSON."""
    rules_file = Path("data/rules/default_rules.json")

    if rules_file.exists():
        with open(rules_file) as f:
            data = json.load(f)

        for rule in data.get("rules", []):
            store.add_rule(
                rule_id=rule["id"],
                rule_text=rule["text"],
                rule_type=rule["type"],
                metadata={
                    "severity": rule.get("severity", "medium"),
                    "category": rule.get("category", "general"),
                    "source": "default"
                }
            )

        print(f"✅ {len(data.get('rules', []))} règles par défaut chargées")


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="DataSentinel AI",
    description="""
## Système Multi-Agents pour la Qualité des Données (v0.5)

DataSentinel AI est un système d'IA agentique capable de:
- **Analyser** les datasets pour détecter les problèmes de qualité
- **Recommander** des corrections avec justifications
- **Expliquer** chaque décision de manière transparente
- **Apprendre** des feedbacks utilisateur

### Architecture

Le système utilise 4 agents spécialisés coordonnés par un orchestrateur:
- **Profiler Agent**: Analyse statistique du dataset
- **Quality Agent**: Détection des problèmes (nulls, anomalies, drift)
- **Corrector Agent**: Proposition de corrections
- **Validator Agent**: Validation contre les règles métier

### Technologies
- LangChain pour l'orchestration des agents
- Claude (Anthropic) comme LLM
- ChromaDB pour le RAG et la mémoire
- Scikit-learn pour la détection d'anomalies
    """,
    version="0.5.0",
    contact={
        "name": "DataSentinel Team",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

# =============================================================================
# MIDDLEWARE
# =============================================================================

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(DataSentinelError)
async def datasentinel_exception_handler(
    request: Request,
    exc: DataSentinelError
) -> JSONResponse:
    """Handler pour les exceptions DataSentinel."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_type=exc.__class__.__name__,
            message=exc.message,
            details=exc.details
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handler pour les exceptions non gérées."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_type="InternalError",
            message="Une erreur interne est survenue",
            details={"error": str(exc)} if settings.is_development else {}
        ).model_dump()
    )


# =============================================================================
# ROUTES
# =============================================================================

# Dépendance d'authentification (no-op si auth_enabled=False)
_auth_dep = [Depends(get_current_user)]

# Inclure les routers (protégés par JWT quand auth_enabled=True)
app.include_router(analyze.router,   dependencies=_auth_dep)
app.include_router(recommend.router, dependencies=_auth_dep)
app.include_router(explain.router,   dependencies=_auth_dep)
app.include_router(feedback.router,  dependencies=_auth_dep)
app.include_router(upload.router,    dependencies=_auth_dep)
app.include_router(batch.router,     dependencies=_auth_dep)
# Auth router : pas de protection (c'est le endpoint de login)
app.include_router(auth_router.router)
# Webhooks : pas de protection (enregistrement public)
app.include_router(webhooks_router.router)

# Prometheus — expose /metrics automatiquement
Instrumentator().instrument(app).expose(app)


# Health check
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Vérification de santé",
    description="Vérifie l'état de santé du système et de ses composants."
)
async def health_check() -> HealthResponse:
    """Endpoint de health check."""
    components = {}

    # Vérifier ChromaDB
    try:
        store = get_chroma_store()
        stats = store.get_stats()
        components["chromadb"] = {
            "status": "healthy",
            "rules_count": stats["rules_count"],
            "decisions_count": stats["decisions_count"],
            "feedback_count": stats["feedback_count"]
        }
    except Exception as e:
        components["chromadb"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # Vérifier la config
    components["config"] = {
        "status": "healthy",
        "environment": settings.environment,
        "model": settings.claude_model
    }

    # Déterminer le statut global
    all_healthy = all(
        c.get("status") == "healthy"
        for c in components.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        timestamp=datetime.utcnow(),
        components=components
    )


# Root endpoint
@app.get(
    "/",
    tags=["System"],
    summary="Bienvenue",
    description="Point d'entrée de l'API"
)
async def root():
    """Message de bienvenue."""
    return {
        "name": "DataSentinel AI",
        "version": "0.1.0",
        "description": "Système multi-agents pour la qualité des données",
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

def run():
    """Lance le serveur uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        workers=1 if settings.is_development else settings.api_workers
    )


if __name__ == "__main__":
    run()
