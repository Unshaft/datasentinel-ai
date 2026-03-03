"""
Route /rules — CRUD de règles métier (v0.6 — F20).

GET    /rules              → liste toutes les règles actives
POST   /rules              → crée une nouvelle règle
DELETE /rules/{rule_id}    → désactive une règle (soft delete)
"""

import uuid

from fastapi import APIRouter, HTTPException, Request, status

from src.api.limiter import limiter
from src.api.schemas.requests import AddRuleRequest
from src.api.schemas.responses import RuleCreateResponse, RuleListResponse, RuleResponse
from src.memory.chroma_store import get_chroma_store

router = APIRouter(prefix="/rules", tags=["Rules"])


@router.get("", response_model=RuleListResponse, summary="Lister les règles métier")
@limiter.limit("60/minute")
async def list_rules(request: Request, rule_type: str | None = None) -> RuleListResponse:
    """Retourne toutes les règles métier actives."""
    store = get_chroma_store()
    raw_rules = store.get_all_rules(rule_type=rule_type)

    rules = [
        RuleResponse(
            rule_id=r["id"],
            text=r["text"],
            rule_type=r.get("rule_type", "constraint"),
            severity=r.get("severity", "medium"),
            category=r.get("category", "general"),
            active=r.get("active", True),
        )
        for r in raw_rules
    ]

    return RuleListResponse(status="success", count=len(rules), rules=rules)


@router.post(
    "",
    response_model=RuleCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Créer une règle métier",
)
@limiter.limit("20/minute")
async def create_rule(request: Request, body: AddRuleRequest) -> RuleCreateResponse:
    """Ajoute une nouvelle règle métier dans ChromaDB."""
    store = get_chroma_store()
    rule_id = f"rule_api_{uuid.uuid4().hex[:12]}"

    store.add_rule(
        rule_id=rule_id,
        rule_text=body.rule_text,
        rule_type=body.rule_type,
        metadata={
            "severity": body.severity,
            "category": body.category,
            "source": "rules_api",
        },
    )

    rule = RuleResponse(
        rule_id=rule_id,
        text=body.rule_text,
        rule_type=body.rule_type,
        severity=body.severity,
        category=body.category,
        active=True,
    )
    return RuleCreateResponse(status="created", rule=rule)


@router.delete("/{rule_id}", summary="Désactiver une règle métier")
@limiter.limit("20/minute")
async def delete_rule(request: Request, rule_id: str) -> dict:
    """Désactive une règle métier (soft delete — ne supprime pas de ChromaDB)."""
    store = get_chroma_store()
    try:
        store.deactivate_rule(rule_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Règle introuvable ou déjà désactivée: {rule_id}",
        ) from exc

    return {"status": "deactivated", "rule_id": rule_id}
