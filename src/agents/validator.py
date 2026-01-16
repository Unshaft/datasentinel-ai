"""
Agent Validator - Validation des corrections proposées.

Cet agent vérifie que les corrections proposées:
- Respectent les règles métier
- Ne créent pas de nouveaux problèmes
- Sont cohérentes entre elles

C'est le gardien final avant approbation humaine.
"""

import json
import time
import uuid
from typing import Any

import pandas as pd

from src.agents.base import BaseAgent
from src.core.models import (
    AgentContext,
    AgentType,
    CorrectionProposal,
    CorrectionType,
    ValidationResult,
)
from src.memory.chroma_store import get_chroma_store
from src.tools.rules import create_rules_tools


class ValidatorAgent(BaseAgent):
    """
    Agent spécialisé dans la validation des corrections.

    Rôle: Vérifier que les corrections proposées sont sûres
    et conformes aux règles métier avant approbation.

    Validations effectuées:
    - Conformité aux règles métier (via RAG)
    - Cohérence des paramètres
    - Non-création de nouveaux problèmes
    - Évaluation de l'impact global
    """

    def __init__(self) -> None:
        """Initialise le Validator Agent."""
        super().__init__(
            agent_type=AgentType.VALIDATOR,
            tools=create_rules_tools()
        )
        self.store = get_chroma_store()

    @property
    def system_prompt(self) -> str:
        """Prompt système du Validator Agent."""
        return """Tu es un Data Validation Agent expert dans la vérification des corrections de données.

Ton rôle est de valider les corrections proposées avant leur application:
1. Vérifier la conformité avec les règles métier
2. S'assurer que la correction ne crée pas de nouveaux problèmes
3. Évaluer l'impact global de la correction
4. Approuver ou rejeter avec justification

Pour chaque validation, tu dois fournir:
- Un statut VALIDE ou INVALIDE
- Les RÈGLES vérifiées
- Les RAISONS de la décision
- Des AVERTISSEMENTS éventuels

Tu dois être:
- RIGOUREUX: Vérifier systématiquement contre les règles
- PRUDENT: En cas de doute, signaler les risques
- EXPLICITE: Justifier chaque décision de validation
- COHÉRENT: Appliquer les mêmes standards à toutes les propositions

Tu ne dois JAMAIS:
- Approuver sans vérification
- Ignorer des règles métier connues
- Valider des corrections potentiellement dangereuses
"""

    def execute(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> AgentContext:
        """
        Valide les corrections proposées.

        Args:
            context: Contexte avec les propositions
            df: DataFrame original (pour simulation)
            **kwargs: Options

        Returns:
            Contexte mis à jour avec les validations
        """
        start_time = time.time()

        validations = []

        for proposal in context.proposals:
            validation = self._validate_proposal(proposal, df, context)
            validations.append(validation)

            # Mettre à jour le statut de la proposition
            proposal.is_approved = validation.is_valid
            proposal.approved_by = (
                self.agent_type.value if validation.is_valid else None
            )

        # Mettre à jour le contexte
        context.validations = validations
        context.current_step = "validated"
        context.iteration += 1

        # Calculer la confiance
        confidence = self._calculate_validation_confidence(validations)

        # Logger
        processing_time = int((time.time() - start_time) * 1000)
        approved_count = sum(1 for v in validations if v.is_valid)

        self._log_decision(
            context=context,
            action="validate_corrections",
            reasoning=f"Validé {approved_count}/{len(validations)} corrections",
            input_summary=f"Propositions: {len(context.proposals)}",
            output_summary=self._summarize_validations(validations),
            confidence=confidence.overall_score,
            processing_time_ms=processing_time
        )

        return context

    def _validate_proposal(
        self,
        proposal: CorrectionProposal,
        df: pd.DataFrame,
        context: AgentContext
    ) -> ValidationResult:
        """
        Valide une proposition de correction.

        Args:
            proposal: Proposition à valider
            df: DataFrame
            context: Contexte

        Returns:
            Résultat de validation
        """
        reasons = []
        warnings = []
        rules_checked = []
        rules_passed = []
        rules_failed = []

        # 1. Vérification des règles métier
        rule_validation = self._check_business_rules(proposal, df)
        rules_checked.extend(rule_validation["checked"])
        rules_passed.extend(rule_validation["passed"])
        rules_failed.extend(rule_validation["failed"])

        if rule_validation["failed"]:
            reasons.append(f"Règles violées: {', '.join(rule_validation['failed'])}")

        # 2. Vérification de la cohérence des paramètres
        param_validation = self._validate_parameters(proposal, df)
        if not param_validation["valid"]:
            reasons.append(param_validation["reason"])
        if param_validation.get("warnings"):
            warnings.extend(param_validation["warnings"])

        # 3. Simulation de l'impact
        impact_validation = self._simulate_impact(proposal, df)
        if not impact_validation["safe"]:
            reasons.append(impact_validation["reason"])
        if impact_validation.get("warnings"):
            warnings.extend(impact_validation["warnings"])

        # 4. Vérification de la confiance minimale
        if proposal.confidence < 0.5:
            warnings.append(
                f"Confiance faible ({proposal.confidence:.2f}) - validation humaine recommandée"
            )

        # Décision finale
        is_valid = len(rules_failed) == 0 and len(reasons) == 0

        # Statut détaillé
        if is_valid and not warnings:
            status = "approved"
        elif is_valid and warnings:
            status = "approved_with_warnings"
        else:
            status = "rejected"

        return ValidationResult(
            validation_id=f"val_{uuid.uuid4().hex[:8]}",
            proposal_id=proposal.proposal_id,
            is_valid=is_valid,
            validation_status=status,
            reasons=reasons,
            warnings=warnings,
            rules_checked=rules_checked,
            rules_passed=rules_passed,
            rules_failed=rules_failed
        )

    def _check_business_rules(
        self,
        proposal: CorrectionProposal,
        df: pd.DataFrame
    ) -> dict[str, list]:
        """
        Vérifie la conformité avec les règles métier.

        Utilise le RAG sur ChromaDB pour trouver les règles pertinentes.
        """
        result = {
            "checked": [],
            "passed": [],
            "failed": []
        }

        # Construire la requête de recherche
        query = f"{proposal.correction_type.value} on column {proposal.parameters.get('column', 'unknown')}"

        # Chercher les règles pertinentes
        try:
            rules = self.store.search_rules(
                query=query,
                n_results=5
            )

            for rule in rules:
                rule_id = rule["id"]
                rule_text = rule["text"]
                similarity = rule.get("similarity", 0)

                # Ne considérer que les règles suffisamment pertinentes
                if similarity < 0.5:
                    continue

                result["checked"].append(rule_id)

                # Vérifier si la correction pourrait violer la règle
                violation = self._check_rule_violation(proposal, rule, df)

                if violation:
                    result["failed"].append(f"{rule_id}: {violation}")
                else:
                    result["passed"].append(rule_id)

        except Exception:
            # Si erreur de recherche, pas de règles à vérifier
            pass

        return result

    def _check_rule_violation(
        self,
        proposal: CorrectionProposal,
        rule: dict,
        df: pd.DataFrame
    ) -> str | None:
        """
        Vérifie si une correction viole une règle spécifique.

        Returns:
            Message de violation ou None si OK
        """
        rule_text = rule["text"].lower()
        rule_type = rule["metadata"].get("rule_type", "")

        column = proposal.parameters.get("column")

        # Règle sur les valeurs nulles
        if "null" in rule_text and proposal.correction_type == CorrectionType.DELETE_ROW:
            if proposal.rows_affected > len(df) * 0.3:
                return "Suppression de plus de 30% des lignes"

        # Règle sur les valeurs uniques/ID
        if ("unique" in rule_text or "id" in rule_text) and column:
            if column.lower().endswith("_id") or "id" in column.lower():
                if proposal.correction_type in [
                    CorrectionType.IMPUTE_MEAN,
                    CorrectionType.IMPUTE_MEDIAN,
                    CorrectionType.IMPUTE_MODE
                ]:
                    return "Imputation sur colonne ID créerait des doublons"

        # Règle sur les valeurs positives
        if "positif" in rule_text or "positive" in rule_text:
            if proposal.correction_type == CorrectionType.CLIP_VALUES:
                lower = proposal.parameters.get("lower", 0)
                if lower < 0:
                    return "Le clipping autoriserait des valeurs négatives"

        return None

    def _validate_parameters(
        self,
        proposal: CorrectionProposal,
        df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Valide les paramètres de la correction.
        """
        result = {"valid": True, "reason": None, "warnings": []}
        params = proposal.parameters

        # Vérifier que la colonne existe
        column = params.get("column")
        if column and column not in df.columns:
            result["valid"] = False
            result["reason"] = f"Colonne '{column}' non trouvée dans le dataset"
            return result

        # Vérifications spécifiques par type
        if proposal.correction_type == CorrectionType.CLIP_VALUES:
            lower = params.get("lower")
            upper = params.get("upper")
            if lower is not None and upper is not None and lower >= upper:
                result["valid"] = False
                result["reason"] = f"Bornes de clipping invalides: {lower} >= {upper}"

        elif proposal.correction_type == CorrectionType.IMPUTE_CUSTOM:
            value = params.get("value")
            if value is None:
                result["valid"] = False
                result["reason"] = "Valeur d'imputation non spécifiée"

        # Avertissements
        if proposal.rows_affected > len(df) * 0.5:
            result["warnings"].append(
                f"Cette correction affecte plus de 50% des lignes ({proposal.rows_affected})"
            )

        return result

    def _simulate_impact(
        self,
        proposal: CorrectionProposal,
        df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Simule l'impact de la correction sans l'appliquer.
        """
        result = {"safe": True, "reason": None, "warnings": []}

        column = proposal.parameters.get("column")
        if not column or column not in df.columns:
            return result

        series = df[column]

        # Simulation selon le type de correction
        if proposal.correction_type == CorrectionType.DELETE_ROW:
            new_size = len(df) - proposal.rows_affected
            if new_size < 10:
                result["safe"] = False
                result["reason"] = f"Suppression laisserait seulement {new_size} lignes"
            elif new_size < len(df) * 0.1:
                result["warnings"].append(
                    f"Suppression laisserait seulement {new_size} lignes ({new_size/len(df)*100:.1f}%)"
                )

        elif proposal.correction_type in [
            CorrectionType.IMPUTE_MEAN,
            CorrectionType.IMPUTE_MEDIAN
        ]:
            # Vérifier si l'imputation créerait des anomalies de distribution
            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    impute_count = proposal.rows_affected
                    original_std = non_null.std()

                    # Si on impute une grande proportion, variance va diminuer
                    if impute_count > len(df) * 0.3:
                        result["warnings"].append(
                            "Imputation massive pourrait réduire significativement la variance"
                        )

        elif proposal.correction_type == CorrectionType.CLIP_VALUES:
            lower = proposal.parameters.get("lower", float("-inf"))
            upper = proposal.parameters.get("upper", float("inf"))

            if pd.api.types.is_numeric_dtype(series):
                clipped = series.clip(lower=lower, upper=upper)
                changed = (series != clipped).sum()

                if changed > proposal.rows_affected * 1.5:
                    result["warnings"].append(
                        f"Le clipping affectera {changed} valeurs (plus que prévu)"
                    )

        return result

    def _calculate_validation_confidence(
        self,
        validations: list[ValidationResult]
    ) -> Any:
        """Calcule la confiance dans les validations."""
        if not validations:
            return self._calculate_confidence(data_quality=0.5, sample_size=0)

        # Confiance basée sur le ratio de validations réussies
        approved = sum(1 for v in validations if v.is_valid)
        ratio = approved / len(validations)

        # Pénaliser si beaucoup d'avertissements
        total_warnings = sum(len(v.warnings) for v in validations)
        warning_penalty = min(0.2, total_warnings * 0.02)

        return self._calculate_confidence(
            data_quality=ratio - warning_penalty,
            sample_size=len(validations),
            signal_scores=[0.9 if v.is_valid else 0.3 for v in validations],
            rule_coverage=0.9
        )

    def _summarize_validations(self, validations: list[ValidationResult]) -> str:
        """Résume les résultats de validation."""
        if not validations:
            return "Aucune validation effectuée"

        approved = sum(1 for v in validations if v.is_valid)
        with_warnings = sum(
            1 for v in validations
            if v.is_valid and v.warnings
        )
        rejected = len(validations) - approved

        return (
            f"Approuvées: {approved} (dont {with_warnings} avec warnings) | "
            f"Rejetées: {rejected}"
        )

    def validate_with_llm(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> tuple[AgentContext, str]:
        """
        Validation avec analyse LLM additionnelle.

        Pour les cas complexes, demande une analyse supplémentaire au LLM.
        """
        # D'abord la validation standard
        context = self.execute(context, df)

        # Préparer les cas nécessitant une attention particulière
        complex_cases = []
        for i, validation in enumerate(context.validations):
            if validation.warnings or not validation.is_valid:
                proposal = context.proposals[i]
                complex_cases.append({
                    "proposal": proposal.description,
                    "status": validation.validation_status,
                    "warnings": validation.warnings,
                    "reasons": validation.reasons
                })

        if not complex_cases:
            return context, "Toutes les validations sont claires, pas d'analyse supplémentaire nécessaire."

        # Demander une analyse LLM
        prompt = f"""Voici des corrections qui nécessitent une attention particulière:

{json.dumps(complex_cases, indent=2)}

Pour chaque cas:
1. Évalue si le rejet ou l'avertissement est justifié
2. Suggère des ajustements possibles
3. Indique si une validation humaine est nécessaire

Sois concis et actionnable."""

        response = self._invoke_llm(prompt, include_tools=False)
        llm_analysis = response.content

        context.metadata["validation_llm_analysis"] = llm_analysis

        return context, llm_analysis
