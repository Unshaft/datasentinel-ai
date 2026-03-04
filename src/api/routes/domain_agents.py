"""
Route /domain-agents — CRUD des agents métier personnalisés (v1.0 — F32).

GET    /domain-agents                → liste les profils actifs
POST   /domain-agents                → crée un profil → HTTP 201
GET    /domain-agents/{domain_id}    → récupère un profil par ID
DELETE /domain-agents/{domain_id}    → supprime un profil
"""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.limiter import limiter
from src.core.domain_manager import DomainManager, DomainProfile, DomainRule

router = APIRouter(prefix="/domain-agents", tags=["Domain Agents"])


# ---------------------------------------------------------------------------
# Schémas de requête / réponse
# ---------------------------------------------------------------------------


class DomainRuleIn(BaseModel):
    text: str = Field(..., min_length=5, description="Texte de la règle")
    applies_to_types: list[str] = Field(default_factory=list)


class CreateDomainAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="Nom du domaine (ex: RH)")
    description: str = Field(default="", max_length=200)
    trigger_types: list[str] = Field(
        default_factory=list,
        description="Types sémantiques déclencheurs (ex: ['employee_id', 'monetary_amount'])",
    )
    min_match_ratio: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Fraction minimale des trigger_types présents pour activer le profil",
    )
    required_types: list[str] = Field(
        default_factory=list,
        description="Types sémantiques obligatoires (absence → CONSTRAINT_VIOLATION CRITICAL)",
    )
    rules: list[DomainRuleIn] = Field(default_factory=list)
    severity_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Override de sévérité par type sémantique ({type: 'critical'|'high'|...})",
    )


class DomainRuleOut(BaseModel):
    rule_id: str
    text: str
    applies_to_types: list[str]


class DomainAgentResponse(BaseModel):
    domain_id: str
    name: str
    description: str
    trigger_types: list[str]
    min_match_ratio: float
    required_types: list[str]
    rules: list[DomainRuleOut]
    severity_overrides: dict[str, str]
    active: bool
    created_at: str


class DomainAgentListResponse(BaseModel):
    status: str
    count: int
    profiles: list[DomainAgentResponse]


class DomainAgentCreateResponse(BaseModel):
    status: str
    profile: DomainAgentResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile_to_response(p: DomainProfile) -> DomainAgentResponse:
    return DomainAgentResponse(
        domain_id=p.domain_id,
        name=p.name,
        description=p.description,
        trigger_types=p.trigger_types,
        min_match_ratio=p.min_match_ratio,
        required_types=p.required_types,
        rules=[
            DomainRuleOut(
                rule_id=r.rule_id, text=r.text, applies_to_types=r.applies_to_types
            )
            for r in p.rules
        ],
        severity_overrides=p.severity_overrides,
        active=p.active,
        created_at=p.created_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=DomainAgentListResponse,
    summary="Lister les agents métier",
)
@limiter.limit("60/minute")
async def list_domain_agents(request: Request) -> DomainAgentListResponse:
    """Retourne tous les profils de domaine actifs."""
    manager = DomainManager()
    profiles = manager.list_profiles(active_only=True)
    return DomainAgentListResponse(
        status="success",
        count=len(profiles),
        profiles=[_profile_to_response(p) for p in profiles],
    )


@router.post(
    "",
    response_model=DomainAgentCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Créer un agent métier",
)
@limiter.limit("20/minute")
async def create_domain_agent(
    request: Request, body: CreateDomainAgentRequest
) -> DomainAgentCreateResponse:
    """Crée un nouveau profil de domaine et le persiste."""
    profile = DomainProfile(
        name=body.name,
        description=body.description,
        trigger_types=body.trigger_types,
        min_match_ratio=body.min_match_ratio,
        required_types=body.required_types,
        rules=[
            DomainRule(text=r.text, applies_to_types=r.applies_to_types)
            for r in body.rules
        ],
        severity_overrides=body.severity_overrides,
    )
    created = DomainManager().create(profile)
    return DomainAgentCreateResponse(status="created", profile=_profile_to_response(created))


@router.get(
    "/{domain_id}",
    response_model=DomainAgentResponse,
    summary="Récupérer un agent métier",
)
@limiter.limit("60/minute")
async def get_domain_agent(request: Request, domain_id: str) -> DomainAgentResponse:
    """Récupère un profil de domaine par son ID."""
    profile = DomainManager().get(domain_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent métier introuvable : {domain_id}",
        )
    return _profile_to_response(profile)


@router.delete(
    "/{domain_id}",
    summary="Supprimer un agent métier",
)
@limiter.limit("20/minute")
async def delete_domain_agent(request: Request, domain_id: str) -> dict:
    """Supprime définitivement un profil de domaine."""
    deleted = DomainManager().delete(domain_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent métier introuvable : {domain_id}",
        )
    return {"status": "deleted", "domain_id": domain_id}
