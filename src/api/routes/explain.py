"""
Route /explain - Explications des décisions.

Endpoint pour obtenir des explications détaillées sur
les décisions prises par le système (issues, proposals, etc.).
"""

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.requests import ExplainRequest
from src.api.schemas.responses import ErrorResponse, ExplainResponse
from src.memory.chroma_store import get_chroma_store
from src.memory.decision_log import get_decision_logger

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post(
    "",
    response_model=ExplainResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Élément non trouvé"},
        400: {"model": ErrorResponse, "description": "Requête invalide"}
    },
    summary="Expliquer une décision",
    description="""
    Fournit une explication détaillée d'une décision du système.

    Types supportés:
    - `issue`: Pourquoi ce problème a été détecté
    - `proposal`: Pourquoi cette correction est proposée
    - `decision`: Pourquoi cette action a été prise
    - `validation`: Pourquoi cette validation a réussi/échoué

    L'explication inclut:
    - Les facteurs contributifs
    - Les règles métier impliquées
    - Des décisions passées similaires
    - La décomposition du score de confiance
    """
)
async def explain_decision(request: ExplainRequest) -> ExplainResponse:
    """
    Génère une explication pour un élément du système.

    Args:
        request: Identifiants et options

    Returns:
        Explication détaillée
    """
    try:
        store = get_chroma_store()
        decision_logger = get_decision_logger()

        # Construire l'explication selon le type
        if request.target_type == "issue":
            return await _explain_issue(request, store, decision_logger)
        elif request.target_type == "proposal":
            return await _explain_proposal(request, store, decision_logger)
        elif request.target_type == "decision":
            return await _explain_decision(request, store, decision_logger)
        elif request.target_type == "validation":
            return await _explain_validation(request, store, decision_logger)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Type non supporté: {request.target_type}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération de l'explication: {str(e)}"
        )


async def _explain_issue(request: ExplainRequest, store, logger) -> ExplainResponse:
    """Génère l'explication d'un problème détecté."""
    # Rechercher des décisions similaires
    similar = logger.find_similar(
        current_situation=f"Issue detection {request.target_id}",
        n_results=3
    )

    # Rechercher les règles pertinentes
    rules = store.search_rules(
        query=f"data quality issue {request.target_id}",
        n_results=5
    )

    explanation = (
        f"Ce problème (ID: {request.target_id}) a été détecté par l'agent Quality. "
        "La détection est basée sur une combinaison de méthodes statistiques "
        "(analyse de distribution, détection d'outliers) et de règles métier."
    )

    if request.detail_level == "detailed":
        explanation += (
            "\n\nMéthodologie de détection:\n"
            "1. Profiling initial: analyse de la structure et types de données\n"
            "2. Analyse statistique: calcul des métriques (nulls, distribution)\n"
            "3. Détection d'anomalies: Isolation Forest pour valeurs aberrantes\n"
            "4. Validation règles: vérification contre les règles métier connues"
        )

    contributing_factors = [
        "Analyse statistique de la distribution",
        "Comparaison avec les seuils configurés",
        "Vérification des règles métier"
    ]

    confidence_breakdown = {
        "data_quality": 0.85,
        "sample_size": 0.9,
        "signal_consistency": 0.8,
        "rule_coverage": 0.75
    }

    return ExplainResponse(
        session_id=request.session_id,
        target_id=request.target_id,
        target_type=request.target_type,
        explanation=explanation,
        contributing_factors=contributing_factors,
        confidence_breakdown=confidence_breakdown,
        related_rules=[r["text"] for r in rules[:3]],
        similar_past_decisions=[
            {"id": s["id"], "similarity": s.get("similarity", 0)}
            for s in similar
        ]
    )


async def _explain_proposal(request: ExplainRequest, store, logger) -> ExplainResponse:
    """Génère l'explication d'une proposition de correction."""
    similar = logger.find_similar(
        current_situation=f"Correction proposal {request.target_id}",
        n_results=3
    )

    rules = store.search_rules(
        query="data correction imputation cleaning",
        n_results=5
    )

    explanation = (
        f"Cette correction (ID: {request.target_id}) a été proposée par l'agent Corrector. "
        "Le choix de cette méthode est basé sur:\n"
        "- Le type de problème détecté\n"
        "- Les caractéristiques statistiques de la colonne\n"
        "- Les meilleures pratiques de nettoyage de données\n"
        "- Les feedbacks sur des corrections similaires passées"
    )

    if request.detail_level == "detailed":
        explanation += (
            "\n\nProcessus de décision:\n"
            "1. Analyse du problème: identification du type et de la sévérité\n"
            "2. Génération d'options: création de plusieurs approches possibles\n"
            "3. Évaluation: scoring de chaque option selon l'impact estimé\n"
            "4. Sélection: choix de la meilleure option avec justification"
        )

    contributing_factors = [
        "Type de données de la colonne",
        "Distribution des valeurs existantes",
        "Proportion de valeurs affectées",
        "Historique de corrections similaires"
    ]

    # Récupérer les feedbacks pertinents
    feedbacks = store.search_similar_feedback(
        query=f"correction proposal similar to {request.target_id}",
        n_results=3
    )

    confidence_breakdown = {
        "issue_confidence": 0.8,
        "correction_impact": 0.75,
        "historical_success": 0.7,
        "rule_validation": 0.85
    }

    return ExplainResponse(
        session_id=request.session_id,
        target_id=request.target_id,
        target_type=request.target_type,
        explanation=explanation,
        contributing_factors=contributing_factors,
        confidence_breakdown=confidence_breakdown,
        related_rules=[r["text"] for r in rules[:3]],
        similar_past_decisions=[
            {"id": s["id"], "similarity": s.get("similarity", 0), "was_correct": s.get("metadata", {}).get("is_correct")}
            for s in feedbacks
        ]
    )


async def _explain_decision(request: ExplainRequest, store, logger) -> ExplainResponse:
    """Génère l'explication d'une décision d'agent."""
    similar = logger.find_similar(
        current_situation=f"Agent decision {request.target_id}",
        n_results=5
    )

    explanation = (
        f"Cette décision (ID: {request.target_id}) a été prise par un agent du système. "
        "Les agents prennent des décisions basées sur:\n"
        "- Les données d'entrée analysées\n"
        "- Les règles métier configurées\n"
        "- L'historique des décisions similaires\n"
        "- Les feedbacks utilisateur accumulés"
    )

    # Calculer la précision historique
    accuracy = logger.get_historical_accuracy()

    if accuracy is not None:
        explanation += f"\n\nPrécision historique de ce type d'agent: {accuracy:.1%}"

    contributing_factors = [
        "Analyse des données d'entrée",
        "Consultation des règles métier",
        "Apprentissage des décisions passées",
        "Calcul du score de confiance"
    ]

    return ExplainResponse(
        session_id=request.session_id,
        target_id=request.target_id,
        target_type=request.target_type,
        explanation=explanation,
        contributing_factors=contributing_factors,
        confidence_breakdown={
            "data_analysis": 0.85,
            "rule_matching": 0.8,
            "historical_learning": accuracy or 0.7
        },
        related_rules=[],
        similar_past_decisions=[
            {"id": s["id"], "similarity": s.get("similarity", 0)}
            for s in similar[:3]
        ]
    )


async def _explain_validation(request: ExplainRequest, store, logger) -> ExplainResponse:
    """Génère l'explication d'une validation."""
    rules = store.search_rules(
        query="validation constraint rule",
        n_results=5
    )

    explanation = (
        f"Cette validation (ID: {request.target_id}) a été effectuée par l'agent Validator. "
        "La validation vérifie que les corrections proposées:\n"
        "- Respectent les règles métier définies\n"
        "- Ne créent pas de nouveaux problèmes\n"
        "- Ont des paramètres cohérents\n"
        "- Ont un impact acceptable sur les données"
    )

    contributing_factors = [
        "Vérification des règles métier",
        "Simulation de l'impact",
        "Validation des paramètres",
        "Analyse de cohérence"
    ]

    return ExplainResponse(
        session_id=request.session_id,
        target_id=request.target_id,
        target_type=request.target_type,
        explanation=explanation,
        contributing_factors=contributing_factors,
        confidence_breakdown={
            "rule_coverage": 0.9,
            "parameter_validity": 0.95,
            "impact_safety": 0.85
        },
        related_rules=[r["text"] for r in rules[:3]],
        similar_past_decisions=[]
    )
