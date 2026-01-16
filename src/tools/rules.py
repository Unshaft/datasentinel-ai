"""
Tool de consultation des règles métier pour les agents LangChain.

Permet aux agents de rechercher et valider contre les règles
métier stockées dans ChromaDB via RAG.
"""

import json
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.memory.chroma_store import ChromaStore, get_chroma_store


class SearchRulesInput(BaseModel):
    """Input pour la recherche de règles."""

    query: str = Field(
        ...,
        description="Description du problème ou contexte pour trouver les règles pertinentes"
    )
    rule_type: str | None = Field(
        default=None,
        description="Type de règle à chercher: constraint, validation, format, consistency"
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximum de règles à retourner"
    )


class ValidateAgainstRulesInput(BaseModel):
    """Input pour validation contre les règles."""

    context: str = Field(
        ...,
        description="Description du contexte ou de la donnée à valider"
    )
    column_name: str | None = Field(
        default=None,
        description="Nom de la colonne concernée"
    )
    value: Any = Field(
        default=None,
        description="Valeur à valider"
    )


class AddRuleInput(BaseModel):
    """Input pour ajouter une règle."""

    rule_text: str = Field(
        ...,
        description="Texte de la règle en langage naturel"
    )
    rule_type: str = Field(
        default="constraint",
        description="Type: constraint, validation, format, consistency"
    )
    severity: str = Field(
        default="medium",
        description="Sévérité: low, medium, high, critical"
    )
    category: str = Field(
        default="general",
        description="Catégorie: completeness, uniqueness, validity, consistency, accuracy"
    )


class SearchRulesTool(BaseTool):
    """
    Tool pour rechercher des règles métier pertinentes via RAG.

    Utilise la recherche sémantique dans ChromaDB pour trouver
    les règles applicables au contexte actuel.
    """

    name: str = "search_business_rules"
    description: str = """
    Recherche des règles métier pertinentes pour un problème ou contexte donné.
    Utilise la recherche sémantique pour trouver les règles les plus applicables.
    Utilisez cet outil pour savoir quelles règles s'appliquent à une situation.
    """
    args_schema: Type[BaseModel] = SearchRulesInput

    store: ChromaStore | None = None

    def _run(
        self,
        query: str,
        rule_type: str | None = None,
        n_results: int = 5
    ) -> str:
        """Recherche les règles pertinentes."""
        if self.store is None:
            self.store = get_chroma_store()

        try:
            rules = self.store.search_rules(
                query=query,
                n_results=n_results,
                rule_type=rule_type
            )

            if not rules:
                return json.dumps({
                    "status": "success",
                    "message": "Aucune règle pertinente trouvée",
                    "rules": [],
                    "query": query
                })

            result = {
                "status": "success",
                "query": query,
                "rule_type_filter": rule_type,
                "rules_found": len(rules),
                "rules": [
                    {
                        "id": rule["id"],
                        "text": rule["text"],
                        "type": rule["metadata"].get("rule_type"),
                        "severity": rule["metadata"].get("severity"),
                        "category": rule["metadata"].get("category"),
                        "similarity": round(rule.get("similarity", 0), 4)
                    }
                    for rule in rules
                ]
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })


class ValidateAgainstRulesTool(BaseTool):
    """
    Tool pour valider une donnée ou contexte contre les règles métier.

    Cherche les règles pertinentes et évalue si le contexte
    les respecte ou les viole.
    """

    name: str = "validate_against_rules"
    description: str = """
    Valide un contexte ou une valeur contre les règles métier applicables.
    Retourne les règles pertinentes et une évaluation de conformité.
    Utilisez cet outil pour vérifier si une donnée respecte les règles.
    """
    args_schema: Type[BaseModel] = ValidateAgainstRulesInput

    store: ChromaStore | None = None

    def _run(
        self,
        context: str,
        column_name: str | None = None,
        value: Any = None
    ) -> str:
        """Valide contre les règles."""
        if self.store is None:
            self.store = get_chroma_store()

        # Construire la requête de recherche
        search_query = context
        if column_name:
            search_query += f" column {column_name}"
        if value is not None:
            search_query += f" value {value}"

        try:
            rules = self.store.search_rules(
                query=search_query,
                n_results=10
            )

            if not rules:
                return json.dumps({
                    "status": "success",
                    "message": "Aucune règle applicable trouvée",
                    "validation_result": "no_rules_apply",
                    "context": context
                })

            # Classifier les règles par pertinence
            highly_relevant = [r for r in rules if r.get("similarity", 0) > 0.7]
            moderately_relevant = [r for r in rules if 0.5 < r.get("similarity", 0) <= 0.7]

            # Évaluation (simplifiée - en production, ce serait plus sophistiqué)
            potential_violations = []
            for rule in highly_relevant:
                # Heuristique simple basée sur les mots-clés
                rule_text = rule["text"].lower()
                severity = rule["metadata"].get("severity", "medium")

                # Si le contexte mentionne un problème et la règle est pertinente
                problem_keywords = ["null", "missing", "invalid", "incorrect", "error", "anomaly"]
                if any(kw in context.lower() for kw in problem_keywords):
                    potential_violations.append({
                        "rule_id": rule["id"],
                        "rule_text": rule["text"],
                        "severity": severity,
                        "relevance": round(rule.get("similarity", 0), 4),
                        "assessment": "potential_violation"
                    })

            result = {
                "status": "success",
                "context": context,
                "column": column_name,
                "value": str(value) if value is not None else None,
                "rules_checked": len(rules),
                "highly_relevant_rules": len(highly_relevant),
                "potential_violations": potential_violations,
                "validation_summary": self._generate_summary(potential_violations, rules),
                "applicable_rules": [
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "severity": r["metadata"].get("severity"),
                        "relevance": round(r.get("similarity", 0), 4)
                    }
                    for r in highly_relevant[:5]
                ]
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    @staticmethod
    def _generate_summary(violations: list, all_rules: list) -> str:
        """Génère un résumé de la validation."""
        if not violations:
            if all_rules:
                return "Aucune violation détectée parmi les règles vérifiées"
            return "Aucune règle applicable à ce contexte"

        critical = sum(1 for v in violations if v.get("severity") == "critical")
        high = sum(1 for v in violations if v.get("severity") == "high")

        if critical > 0:
            return f"ATTENTION: {critical} violation(s) critique(s) potentielle(s) détectée(s)"
        elif high > 0:
            return f"Alerte: {high} violation(s) de sévérité haute détectée(s)"
        else:
            return f"{len(violations)} règle(s) potentiellement concernée(s)"


class ListAllRulesTool(BaseTool):
    """
    Tool pour lister toutes les règles métier actives.

    Utile pour avoir une vue d'ensemble des règles disponibles.
    """

    name: str = "list_all_rules"
    description: str = """
    Liste toutes les règles métier actives dans le système.
    Peut filtrer par type de règle.
    Utilisez cet outil pour voir l'ensemble des règles disponibles.
    """

    store: ChromaStore | None = None

    def _run(self, rule_type: str | None = None) -> str:
        """Liste toutes les règles."""
        if self.store is None:
            self.store = get_chroma_store()

        try:
            rules = self.store.get_all_rules(rule_type=rule_type)

            # Grouper par catégorie
            by_category: dict[str, list] = {}
            for rule in rules:
                category = rule["metadata"].get("category", "general")
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append({
                    "id": rule["id"],
                    "text": rule["text"],
                    "type": rule["metadata"].get("rule_type"),
                    "severity": rule["metadata"].get("severity")
                })

            result = {
                "status": "success",
                "total_rules": len(rules),
                "filter_type": rule_type,
                "by_category": by_category,
                "severity_distribution": {
                    "critical": sum(1 for r in rules if r["metadata"].get("severity") == "critical"),
                    "high": sum(1 for r in rules if r["metadata"].get("severity") == "high"),
                    "medium": sum(1 for r in rules if r["metadata"].get("severity") == "medium"),
                    "low": sum(1 for r in rules if r["metadata"].get("severity") == "low"),
                }
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })


class AddRuleTool(BaseTool):
    """
    Tool pour ajouter une nouvelle règle métier.

    Permet d'enrichir la base de règles avec des règles
    personnalisées pour le contexte métier spécifique.
    """

    name: str = "add_business_rule"
    description: str = """
    Ajoute une nouvelle règle métier au système.
    La règle sera utilisée pour les validations futures.
    Utilisez cet outil pour enrichir les règles avec des contraintes métier spécifiques.
    """
    args_schema: Type[BaseModel] = AddRuleInput

    store: ChromaStore | None = None

    def _run(
        self,
        rule_text: str,
        rule_type: str = "constraint",
        severity: str = "medium",
        category: str = "general"
    ) -> str:
        """Ajoute une règle."""
        if self.store is None:
            self.store = get_chroma_store()

        import uuid
        rule_id = f"rule_custom_{uuid.uuid4().hex[:8]}"

        try:
            self.store.add_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                rule_type=rule_type,
                metadata={
                    "severity": severity,
                    "category": category,
                    "source": "user_defined"
                }
            )

            return json.dumps({
                "status": "success",
                "message": "Règle ajoutée avec succès",
                "rule_id": rule_id,
                "rule": {
                    "text": rule_text,
                    "type": rule_type,
                    "severity": severity,
                    "category": category
                }
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })


# Factory pour créer les tools
def create_rules_tools(store: ChromaStore | None = None) -> list[BaseTool]:
    """
    Crée les outils de gestion des règles métier.

    Args:
        store: Instance ChromaStore (optionnel)

    Returns:
        Liste d'outils configurés
    """
    search_tool = SearchRulesTool()
    search_tool.store = store

    validate_tool = ValidateAgainstRulesTool()
    validate_tool.store = store

    list_tool = ListAllRulesTool()
    list_tool.store = store

    add_tool = AddRuleTool()
    add_tool.store = store

    return [search_tool, validate_tool, list_tool, add_tool]
