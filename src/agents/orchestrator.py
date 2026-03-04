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

import logging
import time
import uuid
from enum import Enum
from typing import Any

import pandas as pd

from src.agents.base import AgentResult, BaseAgent
from src.agents.corrector import CorrectorAgent
from src.agents.profiler import ProfilerAgent
from src.agents.quality import QualityAgent
from src.agents.semantic_profiler import SemanticProfilerAgent
from src.agents.validator import ValidatorAgent
from src.core.config import settings
from src.core.domain_manager import DomainManager
from src.core.models import AgentContext, AgentType, Severity, TaskStatus
from src.ml.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


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
        self.semantic_profiler = SemanticProfilerAgent()
        self.quality_agent = QualityAgent()
        self.corrector = CorrectorAgent()
        self.validator = ValidatorAgent()

        self.confidence_scorer = ConfidenceScorer(
            escalation_threshold=settings.confidence_threshold
        )
        self._domain_manager = DomainManager()

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
        logger.info("Pipeline START — %s %d×%d", task_type.value, len(df), len(df.columns))

        # Sélectionner et exécuter les agents
        if task_type == TaskType.PROFILE_ONLY:
            context = self._run_profiling(context, df)

        elif task_type == TaskType.QUALITY_ONLY:
            context = self._run_profiling(context, df)
            context = self._run_quality_check(context, df, **options)

        elif task_type == TaskType.ANALYZE:
            context = self._run_profiling(context, df)
            context = self._run_semantic_enrichment_sync(context, df)
            context = self._detect_domain(context)
            context = self._run_quality_check(context, df, **options)

        elif task_type == TaskType.RECOMMEND:
            context = self._run_profiling(context, df)
            context = self._run_semantic_enrichment_sync(context, df)
            context = self._detect_domain(context)
            context = self._run_quality_check(context, df, **options)
            if context.issues:
                context = self._run_correction(context, df)

        elif task_type == TaskType.FULL_PIPELINE:
            context = self._run_profiling(context, df)
            context = self._run_semantic_enrichment_sync(context, df)
            context = self._detect_domain(context)
            context = self._run_quality_check(context, df, **options)
            if context.issues:
                context = self._run_correction(context, df)
                if context.proposals:
                    context = self._run_validation(context, df)

        # Finaliser
        context = self._finalize(context, df, start_time)
        logger.info(
            "Pipeline END — score=%.1f | %d issues | %dms",
            context.metadata.get("quality_score", 0),
            len(context.issues),
            context.metadata.get("processing_time_ms", 0),
        )
        return context

    async def run_pipeline_async(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        task_type: TaskType,
        **options: Any,
    ) -> AgentContext:
        """
        Version async du pipeline — profiler séquentiel, quality en parallèle.

        Le Profiler doit terminer en premier (le QualityAgent utilise le profil).
        Les 4 checks du QualityAgent s'exécutent ensuite en parallèle via
        execute_async(), ce qui réduit la latence de ~40%.

        Args:
            context: Contexte de session
            df: DataFrame à analyser
            task_type: Type de tâche (ANALYZE, RECOMMEND, etc.)
            **options: Options additionnelles

        Returns:
            Contexte finalisé
        """
        import asyncio

        start_time = time.time()
        logger.info("Pipeline START — %s %d×%d", task_type.value, len(df), len(df.columns))

        if task_type == TaskType.PROFILE_ONLY:
            context = await asyncio.to_thread(self._run_profiling, context, df)

        elif task_type == TaskType.QUALITY_ONLY:
            context = await asyncio.to_thread(self._run_profiling, context, df)
            context = await self._run_quality_check_async(context, df, **options)

        elif task_type == TaskType.ANALYZE:
            context = await asyncio.to_thread(self._run_profiling, context, df)
            context = await self.semantic_profiler.enrich_async(context, df)
            context = self._detect_domain(context)
            context = await self._run_quality_check_async(context, df, **options)

        elif task_type == TaskType.RECOMMEND:
            context = await asyncio.to_thread(self._run_profiling, context, df)
            context = await self.semantic_profiler.enrich_async(context, df)
            context = self._detect_domain(context)
            context = await self._run_quality_check_async(context, df, **options)
            if context.issues:
                context = await asyncio.to_thread(self._run_correction, context, df)

        elif task_type == TaskType.FULL_PIPELINE:
            context = await asyncio.to_thread(self._run_profiling, context, df)
            context = await self.semantic_profiler.enrich_async(context, df)
            context = self._detect_domain(context)
            context = await self._run_quality_check_async(context, df, **options)
            if context.issues:
                context = await asyncio.to_thread(self._run_correction, context, df)
                if context.proposals:
                    context = await asyncio.to_thread(self._run_validation, context, df)

        context = await asyncio.to_thread(self._finalize, context, df, start_time)
        logger.info(
            "Pipeline END — score=%.1f | %d issues | %dms",
            context.metadata.get("quality_score", 0),
            len(context.issues),
            context.metadata.get("processing_time_ms", 0),
        )
        return context

    async def run_pipeline_adaptive(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        task_type: TaskType = TaskType.ANALYZE,
        **options: Any,
    ) -> AgentContext:
        """
        Pipeline adaptatif ReAct (F24 — v0.7, F31 — v1.2).

        Phases :
        1. Observe  — Profiler exécuté en premier
        2. Reason   — _build_execution_plan() décide les checks à lancer
        3. Act      — Quality checks sélectifs
        4. Reflect  — Cohérence scores/issues, flags d'incohérence (F31)
        5. Observe  — Si CRITICAL → activation Corrector

        Le raisonnement est tracé dans context.metadata["reasoning_steps"].
        L'API expose ces étapes si include_reasoning=True dans la requête.
        """
        import asyncio

        start_time = time.time()
        logger.info("Pipeline START — %s(adaptive) %d×%d", task_type.value, len(df), len(df.columns))
        reasoning_steps: list[dict] = []

        # ── Phase 1 : Observe (Profiler + Semantic enrichment + Domain) ─────
        context = await asyncio.to_thread(self._run_profiling, context, df)
        context = await self.semantic_profiler.enrich_async(context, df)
        context = self._detect_domain(context)
        sem_count = len(context.metadata.get("semantic_types", {}))
        reasoning_steps.append({
            "step": 1,
            "phase": "observe",
            "thought": "Profil du dataset disponible.",
            "action": "profiler.execute() + semantic_profiler.enrich_async()",
            "observation": (
                f"rows={context.profile.row_count if context.profile else '?'}, "
                f"cols={context.profile.column_count if context.profile else '?'}, "
                f"semantic_types={sem_count}"
            ),
        })

        # ── Phase 2 : Reason (plan adaptatif) ────────────────────────────────
        plan = self._build_execution_plan(context, df, **options)
        context.metadata["execution_plan"] = plan  # exposé pour Reflect (F31)
        logger.info("[ReAct] Plan: %s", " + ".join(plan))
        reasoning_steps.append({
            "step": 2,
            "phase": "reason",
            "thought": "Construction du plan d'exécution basé sur le profil.",
            "action": "_build_execution_plan()",
            "observation": f"checks retenus: {plan}",
        })

        # ── Phase 3 : Act (Quality adaptatif) ────────────────────────────────
        quality_options = {
            "detect_anomalies": "detect_anomalies" in plan,
            "detect_drift": options.get("detect_drift", False) and "drift" in plan,
            "reference_df": options.get("reference_df"),
            "_skip_checks": [c for c in ["missing_values", "format", "anomaly"] if c not in plan],
        }
        context = await self._run_quality_check_async(context, df, **quality_options)
        reasoning_steps.append({
            "step": 3,
            "phase": "act",
            "thought": "Exécution des checks de qualité sélectifs.",
            "action": "quality_agent.execute_async()",
            "observation": f"{len(context.issues)} problèmes détectés.",
        })

        # ── Phase 4 : Reflect (cohérence scores/issues — F31) ────────────────
        reflect_flags = self._reflect_coherence(context, df)
        context.metadata["reflect_flags"] = reflect_flags
        if reflect_flags:
            logger.info("[ReAct/Reflect] Flags d'incohérence: %s", reflect_flags)
        col_scores = context.metadata.get("column_scores", {})
        avg_col = round(sum(col_scores.values()) / len(col_scores), 1) if col_scores else 100.0
        n_crit_reflect = sum(1 for i in context.issues if i.severity.value == "critical")
        reasoning_steps.append({
            "step": 4,
            "phase": "reflect",
            "thought": (
                f"Cohérence : avg_col={avg_col:.0f}%, {n_crit_reflect} critique(s), "
                f"anomaly_in_plan={'detect_anomalies' in plan}."
            ),
            "action": "_reflect_coherence()",
            "observation": (
                f"Flags: {reflect_flags}." if reflect_flags else "Aucune incohérence détectée."
            ),
        })

        # ── Phase 5 : Observe (re-plan si critique) ──────────────────────────
        critical = [i for i in context.issues if i.severity.value == "critical"]
        if critical and task_type in (TaskType.RECOMMEND, TaskType.FULL_PIPELINE):
            reasoning_steps.append({
                "step": 5,
                "phase": "observe",
                "thought": f"{len(critical)} problème(s) critique(s) → activation Corrector.",
                "action": "corrector.execute()",
                "observation": "Corrections proposées pour les problèmes critiques.",
            })
            context = await asyncio.to_thread(self._run_correction, context, df)

        context.metadata["reasoning_steps"] = reasoning_steps
        context = await asyncio.to_thread(self._finalize, context, df, start_time)
        logger.info(
            "Pipeline END — score=%.1f | %d issues | %dms",
            context.metadata.get("quality_score", 0),
            len(context.issues),
            context.metadata.get("processing_time_ms", 0),
        )
        return context

    def _reflect_coherence(
        self,
        context: AgentContext,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Vérifie la cohérence entre scores colonnes et issues détectées (F31 — v1.2).

        Deux règles :
        - score_vs_critical : scores colonnes ≥ 80 en moyenne mais ≥ 2 issues CRITICAL
          (signale des violations domaine ajoutées après le scoring par colonne)
        - plan_blind_spot : detect_anomalies absent du plan mais ≥ 2 issues HIGH/CRITICAL
          (signale que le plan adaptatif a peut-être raté des anomalies)

        Returns:
            Liste de flags d'incohérence (vide si cohérent).
        """
        flags: list[str] = []

        # Règle 1 : scores colonnes bons mais issues CRITICAL présentes
        col_scores = context.metadata.get("column_scores", {})
        n_critical = sum(1 for i in context.issues if i.severity.value == "critical")
        if col_scores and n_critical >= 2:
            avg = sum(col_scores.values()) / len(col_scores)
            if avg >= 80.0:
                flags.append("score_vs_critical")

        # Règle 2 : plan a sauté detect_anomalies mais issues HIGH/CRITICAL détectées
        plan = context.metadata.get("execution_plan", [])
        if plan and "detect_anomalies" not in plan:
            n_high_plus = sum(
                1 for i in context.issues if i.severity.value in ("high", "critical")
            )
            if n_high_plus >= 2:
                flags.append("plan_blind_spot")

        return flags

    def _build_execution_plan(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **options: Any,
    ) -> list[str]:
        """
        Construit un plan d'exécution adaptatif basé sur le profil (F24).

        Returns:
            Liste des checks à exécuter (parmi : missing_values, detect_anomalies,
            format, drift, type, duplicates)
        """
        plan = ["missing_values", "type", "duplicates", "pseudo_nulls", "format"]

        if context.profile is None:
            return plan + ["detect_anomalies"]

        profile = context.profile
        row_count = profile.row_count
        col_count = profile.column_count
        total_nulls = profile.total_null_count

        # Règle 1 : trop peu de lignes → skip IsolationForest (instable < 30 lignes)
        if row_count >= 30:
            plan.append("detect_anomalies")

        # Règle 2 : aucun null → skip missing_values check
        if total_nulls == 0:
            plan = [c for c in plan if c != "missing_values"]

        # Règle 3 : presque tout null (>50%) → skip format checks
        total_cells = row_count * col_count if col_count > 0 else 1
        if total_nulls / max(total_cells, 1) > 0.5:
            plan = [c for c in plan if c != "format"]

        # Règle 4 : trop de colonnes → mode sampling (note dans metadata)
        if col_count > 100:
            context.metadata["sampling_mode"] = True

        # Règle 5 : pas de colonnes numériques → skip anomaly + drift
        numeric_cols = [c for c in profile.columns if c.is_numeric]
        if not numeric_cols:
            plan = [c for c in plan if c not in ("detect_anomalies", "drift")]

        # Règle 6 : detect_drift demandé et référence fournie → activer
        if options.get("detect_drift") and options.get("reference_df") is not None:
            plan.append("drift")

        return plan

    async def _run_quality_check_async(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **options: Any,
    ) -> AgentContext:
        """Délègue au QualityAgent en mode parallèle."""
        context.current_step = "quality_checking"
        quality_options = {
            "detect_anomalies": options.get("detect_anomalies", True),
            "detect_drift": options.get("detect_drift", False),
            "reference_df": options.get("reference_df"),
        }
        return await self.quality_agent.execute_async(context, df, **quality_options)

    def _run_semantic_enrichment_sync(
        self,
        context: AgentContext,
        df: pd.DataFrame,
    ) -> AgentContext:
        """
        Enrichissement sémantique synchrone — heuristiques (F27v2).

        Popule context.metadata["semantic_types"] via le classificateur heuristique
        du SemanticProfilerAgent, sans appel LLM. Permet à F32 (detect_domain)
        et F28 (validate_semantic_types) de fonctionner dans le pipeline synchrone.
        """
        return self.semantic_profiler.enrich_sync(context, df)

    def _detect_domain(self, context: AgentContext) -> AgentContext:
        """
        Détecte le domaine métier actif après F27 (F32 — v1.0).

        Si des semantic_types sont présents dans le contexte, cherche le profil
        DomainManager dont le ratio trigger_types est le meilleur. Le résultat
        est stocké dans context.metadata["domain_id"] / ["domain_name"].
        """
        semantic_types = context.metadata.get("semantic_types")
        if not semantic_types:
            return context
        matched = self._domain_manager.detect_domain(semantic_types)
        if matched:
            context.metadata["domain_id"] = matched.domain_id
            context.metadata["domain_name"] = matched.name
            logger.info("[Domain] Agent '%s' activé", matched.name)
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
        df: pd.DataFrame | None = None,
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

        # Estimation de l'amélioration : chaque correction en attente vaut ~5 points,
        # plafonné par l'écart restant jusqu'à 100%.
        quality_score = context.metadata.get("quality_score", 100)
        pending_count = sum(1 for p in context.proposals if p.is_approved is not False)
        base["estimated_improvement"] = round(
            min(100 - quality_score, pending_count * 5), 1
        )

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
