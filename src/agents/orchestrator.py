"""
Agent Orchestrator - Chef d'orchestre du système multi-agents.

L'Orchestrator est le point d'entrée principal du système.
Il coordonne les agents spécialisés selon la tâche demandée:
- Détermine quels agents activer
- Gère le flux d'exécution
- Agrège les résultats
- Décide des escalades

C'est ici que se manifeste la logique "agentique" du système.
"""

import time
import uuid
from enum import Enum
from typing import Any

import pandas as pd

from src.agents.base import AgentResult, BaseAgent
from src.agents.corrector import CorrectorAgent
from src.agents.profiler import ProfilerAgent
from src.agents.quality import QualityAgent
from src.agents.validator import ValidatorAgent
from src.core.config import settings
from src.core.models import AgentContext, AgentType, Severity, TaskStatus
from src.ml.confidence_scorer import ConfidenceScorer


class TaskType(str, Enum):
    """Types de tâches supportées par l'orchestrateur."""

    ANALYZE = "analyze"          # Analyse complète (profile + quality)
    RECOMMEND = "recommend"      # Analyse + propositions de correction
    FULL_PIPELINE = "full"      # Pipeline complet avec validation
    PROFILE_ONLY = "profile"     # Uniquement le profilage
    QUALITY_ONLY = "quality"     # Uniquement la détection de qualité


class OrchestratorAgent(BaseAgent):
    """
    Orchestrateur central du système multi-agents.

    Responsabilités:
    - Routing des tâches vers les agents appropriés
    - Coordination du flux d'exécution
    - Agrégation des résultats
    - Gestion des escalades humaines
    - Calcul des métriques globales

    Design:
    - Un seul LLM (Claude) avec prompts spécialisés par agent
    - Orchestration logique, pas de multi-LLM
    - Décision dynamique basée sur les résultats intermédiaires
    """

    def __init__(self) -> None:
        """Initialise l'Orchestrator."""
        super().__init__(
            agent_type=AgentType.ORCHESTRATOR,
            tools=[]
        )

        # Initialiser les agents spécialisés
        self.profiler = ProfilerAgent()
        self.quality_agent = QualityAgent()
        self.corrector = CorrectorAgent()
        self.validator = ValidatorAgent()

        self.confidence_scorer = ConfidenceScorer(
            escalation_threshold=settings.confidence_threshold
        )

    @property
    def system_prompt(self) -> str:
        """Prompt système de l'Orchestrator."""
        return """Tu es l'Orchestrator Agent, le coordinateur central d'un système de qualité de données.

Ton rôle est de:
1. Analyser les requêtes entrantes
2. Déterminer quels agents spécialisés activer
3. Coordonner le flux d'exécution
4. Agréger et synthétiser les résultats
5. Décider si une escalade humaine est nécessaire

Agents disponibles:
- PROFILER: Analyse la structure et les statistiques du dataset
- QUALITY: Détecte les problèmes de qualité (nulls, anomalies, drift)
- CORRECTOR: Propose des corrections avec justifications
- VALIDATOR: Valide les corrections contre les règles métier

Principes de décision:
- Toujours commencer par le PROFILER pour comprendre les données
- Activer QUALITY si analyse de problèmes demandée
- Activer CORRECTOR uniquement si des problèmes sont détectés
- VALIDATOR vérifie avant toute proposition finale

Tu dois être:
- EFFICACE: Minimiser les appels d'agents inutiles
- PRUDENT: Escalader si confiance insuffisante
- TRANSPARENT: Expliquer le flux d'exécution choisi
"""

    def execute(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> AgentContext:
        """
        Point d'entrée principal - exécute le pipeline complet.

        Args:
            context: Contexte initial
            df: DataFrame à analyser
            **kwargs: Options (task_type, detect_drift, etc.)

        Returns:
            Contexte final avec tous les résultats
        """
        task_type = kwargs.get("task_type", TaskType.ANALYZE)
        return self.run_pipeline(context, df, task_type, **kwargs)

    def run_pipeline(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        task_type: TaskType,
        **options: Any
    ) -> AgentContext:
        """
        Exécute le pipeline selon le type de tâche.

        Args:
            context: Contexte de session
            df: DataFrame
            task_type: Type de tâche
            **options: Options additionnelles

        Returns:
            Contexte mis à jour
        """
        start_time = time.time()

        # Sélectionner et exécuter les agents
        if task_type == TaskType.PROFILE_ONLY:
            context = self._run_profiling(context, df)

        elif task_type == TaskType.QUALITY_ONLY:
            context = self._run_profiling(context, df)
            context = self._run_quality_check(context, df, **options)

        elif task_type == TaskType.ANALYZE:
            context = self._run_profiling(context, df)
            context = self._run_quality_check(context, df, **options)

        elif task_type == TaskType.RECOMMEND:
            context = self._run_profiling(context, df)
            context = self._run_quality_check(context, df, **options)
            if context.issues:
                context = self._run_correction(context, df)

        elif task_type == TaskType.FULL_PIPELINE:
            context = self._run_profiling(context, df)
            context = self._run_quality_check(context, df, **options)
            if context.issues:
                context = self._run_correction(context, df)
                if context.proposals:
                    context = self._run_validation(context, df)

        # Finaliser
        context = self._finalize(context, df, start_time)

        return context

    def _run_profiling(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> AgentContext:
        """Exécute l'agent Profiler."""
        context.current_step = "profiling"
        return self.profiler.execute(context, df)

    def _run_quality_check(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **options: Any
    ) -> AgentContext:
        """Exécute l'agent Quality."""
        context.current_step = "quality_checking"

        # Passer les options pertinentes
        quality_options = {
            "detect_anomalies": options.get("detect_anomalies", True),
            "detect_drift": options.get("detect_drift", False),
            "reference_df": options.get("reference_df"),
        }

        return self.quality_agent.execute(context, df, **quality_options)

    def _run_correction(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> AgentContext:
        """Exécute l'agent Corrector."""
        context.current_step = "proposing_corrections"
        return self.corrector.execute(context, df)

    def _run_validation(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> AgentContext:
        """Exécute l'agent Validator."""
        context.current_step = "validating"
        return self.validator.execute(context, df)

    def _finalize(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        start_time: float
    ) -> AgentContext:
        """
        Finalise le contexte avec métriques et statut.

        Args:
            context: Contexte en cours
            df: DataFrame
            start_time: Timestamp de début

        Returns:
            Contexte finalisé
        """
        processing_time = int((time.time() - start_time) * 1000)

        # Calculer les métriques globales
        quality_score = self._calculate_quality_score(context, df)
        needs_escalation = self._check_escalation_needed(context)

        # Déterminer le statut final
        if needs_escalation:
            status = TaskStatus.ESCALATED
        elif context.issues and not context.proposals:
            status = TaskStatus.COMPLETED
        elif context.proposals and not any(p.is_approved for p in context.proposals):
            status = TaskStatus.COMPLETED
        else:
            status = TaskStatus.COMPLETED

        # Enrichir les métadonnées
        context.metadata.update({
            "processing_time_ms": processing_time,
            "quality_score": quality_score,
            "needs_human_review": needs_escalation,
            "final_status": status.value,
            "summary": self._generate_summary(context, quality_score, needs_escalation)
        })

        context.current_step = "completed"

        # Logger la décision d'orchestration
        self._log_decision(
            context=context,
            action="orchestrate_pipeline",
            reasoning=f"Pipeline complété avec score qualité {quality_score:.1f}%",
            input_summary=f"Dataset: {len(df)} rows x {len(df.columns)} cols",
            output_summary=context.metadata["summary"],
            confidence=0.9 if not needs_escalation else 0.5,
            processing_time_ms=processing_time
        )

        return context

    def _calculate_quality_score(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> float:
        """
        Calcule un score de qualité global (0-100).

        Le score est basé sur:
        - Complétude (absence de nulls)
        - Validité (absence d'anomalies)
        - Consistance (absence de violations)
        """
        if context.profile is None:
            return 100.0  # Pas de profil = pas de problèmes détectés

        total_cells = context.profile.row_count * context.profile.column_count
        if total_cells == 0:
            return 100.0

        # Pénalités
        null_penalty = 0
        if context.profile.total_null_count > 0:
            null_ratio = context.profile.total_null_count / total_cells
            null_penalty = min(30, null_ratio * 100)  # Max 30 points

        issue_penalty = 0
        if context.issues:
            # Pondérer par sévérité
            severity_weights = {
                Severity.LOW: 1,
                Severity.MEDIUM: 3,
                Severity.HIGH: 7,
                Severity.CRITICAL: 15
            }
            weighted_issues = sum(
                severity_weights.get(i.severity, 1)
                for i in context.issues
            )
            issue_penalty = min(50, weighted_issues)  # Max 50 points

        score = max(0, 100 - null_penalty - issue_penalty)
        return round(score, 1)

    def _check_escalation_needed(self, context: AgentContext) -> bool:
        """
        Détermine si une escalade humaine est nécessaire.

        Critères:
        - Problèmes critiques détectés
        - Confiance globale faible
        - Corrections non validées
        """
        # Problèmes critiques
        critical_issues = [
            i for i in context.issues
            if i.severity == Severity.CRITICAL
        ]
        if critical_issues:
            return True

        # Problèmes nécessitant escalade
        escalation_issues = [
            i for i in context.issues
            if i.needs_escalation
        ]
        if len(escalation_issues) > len(context.issues) * 0.3:
            return True

        # Corrections rejetées ou avec faible confiance
        if context.proposals:
            low_confidence = [
                p for p in context.proposals
                if p.confidence < settings.confidence_threshold
            ]
            if len(low_confidence) > len(context.proposals) * 0.5:
                return True

        return False

    def _generate_summary(
        self,
        context: AgentContext,
        quality_score: float,
        needs_escalation: bool
    ) -> str:
        """Génère un résumé textuel de l'analyse."""
        parts = []

        # Score global
        if quality_score >= 90:
            parts.append(f"Excellente qualité ({quality_score:.0f}%)")
        elif quality_score >= 70:
            parts.append(f"Bonne qualité ({quality_score:.0f}%)")
        elif quality_score >= 50:
            parts.append(f"Qualité moyenne ({quality_score:.0f}%)")
        else:
            parts.append(f"Qualité insuffisante ({quality_score:.0f}%)")

        # Problèmes
        if context.issues:
            by_severity = {}
            for issue in context.issues:
                s = issue.severity.value
                by_severity[s] = by_severity.get(s, 0) + 1
            parts.append(f"Problèmes: {len(context.issues)} ({by_severity})")
        else:
            parts.append("Aucun problème détecté")

        # Corrections
        if context.proposals:
            approved = sum(1 for p in context.proposals if p.is_approved)
            parts.append(f"Corrections: {approved}/{len(context.proposals)} approuvées")

        # Escalade
        if needs_escalation:
            parts.append("⚠️ REVUE HUMAINE RECOMMANDÉE")

        return " | ".join(parts)

    # =========================================================================
    # API PUBLIQUE - Méthodes de haut niveau
    # =========================================================================

    def analyze(
        self,
        df: pd.DataFrame,
        session_id: str | None = None,
        **options: Any
    ) -> dict[str, Any]:
        """
        API de haut niveau pour l'analyse de données.

        Args:
            df: DataFrame à analyser
            session_id: ID de session (optionnel)
            **options: Options (detect_anomalies, detect_drift, etc.)

        Returns:
            Résultats d'analyse structurés
        """
        # Créer le contexte
        context = AgentContext(
            session_id=session_id or f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}"
        )

        # Exécuter le pipeline
        context = self.run_pipeline(
            context, df,
            task_type=TaskType.ANALYZE,
            **options
        )

        # Formater la réponse
        return self._format_analysis_response(context, df)

    def recommend(
        self,
        df: pd.DataFrame,
        session_id: str | None = None,
        **options: Any
    ) -> dict[str, Any]:
        """
        API de haut niveau pour obtenir des recommandations.

        Args:
            df: DataFrame
            session_id: ID de session
            **options: Options

        Returns:
            Recommandations structurées
        """
        context = AgentContext(
            session_id=session_id or f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}"
        )

        context = self.run_pipeline(
            context, df,
            task_type=TaskType.RECOMMEND,
            **options
        )

        return self._format_recommendation_response(context, df)

    def full_analysis(
        self,
        df: pd.DataFrame,
        session_id: str | None = None,
        **options: Any
    ) -> dict[str, Any]:
        """
        API pour le pipeline complet avec validation.

        Args:
            df: DataFrame
            session_id: ID de session
            **options: Options

        Returns:
            Résultats complets
        """
        context = AgentContext(
            session_id=session_id or f"session_{uuid.uuid4().hex[:12]}",
            dataset_id=f"dataset_{uuid.uuid4().hex[:8]}"
        )

        context = self.run_pipeline(
            context, df,
            task_type=TaskType.FULL_PIPELINE,
            **options
        )

        return self._format_full_response(context, df)

    # =========================================================================
    # FORMATAGE DES RÉPONSES
    # =========================================================================

    def _format_analysis_response(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> dict[str, Any]:
        """Formate la réponse d'analyse."""
        return {
            "session_id": context.session_id,
            "dataset_id": context.dataset_id,
            "status": context.metadata.get("final_status", "completed"),
            "quality_score": context.metadata.get("quality_score", 100),
            "summary": context.metadata.get("summary", ""),
            "profile": {
                "rows": context.profile.row_count if context.profile else len(df),
                "columns": context.profile.column_count if context.profile else len(df.columns),
                "null_count": context.profile.total_null_count if context.profile else 0,
            } if context.profile else None,
            "issues": [
                {
                    "id": i.issue_id,
                    "type": i.issue_type.value,
                    "severity": i.severity.value,
                    "column": i.column,
                    "description": i.description,
                    "affected_count": i.affected_count,
                    "confidence": i.confidence
                }
                for i in context.issues
            ],
            "needs_human_review": context.metadata.get("needs_human_review", False),
            "processing_time_ms": context.metadata.get("processing_time_ms", 0)
        }

    def _format_recommendation_response(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> dict[str, Any]:
        """Formate la réponse avec recommandations."""
        base = self._format_analysis_response(context, df)

        base["proposals"] = [
            {
                "id": p.proposal_id,
                "issue_id": p.issue_id,
                "type": p.correction_type.value,
                "description": p.description,
                "justification": p.justification,
                "impact": p.estimated_impact,
                "confidence": p.confidence,
                "alternatives": p.alternatives
            }
            for p in context.proposals
        ]

        return base

    def _format_full_response(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> dict[str, Any]:
        """Formate la réponse complète."""
        base = self._format_recommendation_response(context, df)

        base["validations"] = [
            {
                "id": v.validation_id,
                "proposal_id": v.proposal_id,
                "is_valid": v.is_valid,
                "status": v.validation_status,
                "reasons": v.reasons,
                "warnings": v.warnings,
                "rules_checked": v.rules_checked
            }
            for v in context.validations
        ]

        # Résumé des corrections approuvées
        approved_proposals = [
            p for p in context.proposals if p.is_approved
        ]
        base["approved_corrections"] = [
            {
                "id": p.proposal_id,
                "description": p.description,
                "type": p.correction_type.value
            }
            for p in approved_proposals
        ]

        return base


# Fonction utilitaire pour créer une instance
def create_orchestrator() -> OrchestratorAgent:
    """Crée et retourne une instance de l'Orchestrator."""
    return OrchestratorAgent()
