"""
Agent Corrector - Proposition de corrections.

Cet agent analyse les problèmes détectés et propose des corrections:
- Imputation de valeurs manquantes
- Traitement des anomalies
- Correction de types
- Suggestions de nettoyage

IMPORTANT: Cet agent ne modifie JAMAIS les données.
Il propose uniquement des corrections avec justifications.
"""

import json
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.core.models import (
    AgentContext,
    AgentType,
    CorrectionProposal,
    CorrectionType,
    IssueType,
    QualityIssue,
    Severity,
)
from src.memory.feedback_store import FeedbackStore, get_feedback_store


class CorrectorAgent(BaseAgent):
    """
    Agent spécialisé dans la proposition de corrections.

    Rôle: Pour chaque problème détecté, proposer une ou plusieurs
    corrections avec justification et estimation d'impact.

    Principes:
    - Ne jamais appliquer de correction automatiquement
    - Toujours justifier le choix
    - Proposer des alternatives quand pertinent
    - Estimer l'impact de chaque correction
    """

    def __init__(self) -> None:
        """Initialise le Corrector Agent."""
        super().__init__(
            agent_type=AgentType.CORRECTOR,
            tools=[]
        )
        self.feedback_store = get_feedback_store()

    @property
    def system_prompt(self) -> str:
        """Prompt système du Corrector Agent."""
        return """Tu es un Data Correction Agent expert dans la proposition de corrections de données.

Ton rôle est de proposer des corrections pour les problèmes de qualité détectés:
1. Analyser chaque problème en détail
2. Identifier les corrections possibles
3. Justifier tes recommandations
4. Estimer l'impact de chaque correction

Pour chaque correction proposée, tu dois fournir:
- Le TYPE de correction (imputation, suppression, conversion, etc.)
- Une JUSTIFICATION claire (pourquoi cette approche)
- L'IMPACT estimé sur les données
- Des ALTERNATIVES possibles
- Un niveau de CONFIANCE

Tu dois être:
- PRUDENT: Proposer, jamais appliquer automatiquement
- EXPLICITE: Chaque correction doit être justifiée
- COMPLET: Proposer des alternatives quand possible
- RÉALISTE: Estimer correctement l'impact

Tu ne dois JAMAIS:
- Appliquer une correction sans validation
- Ignorer des problèmes signalés
- Proposer des corrections qui pourraient aggraver la situation
"""

    def execute(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> AgentContext:
        """
        Propose des corrections pour les problèmes détectés.

        Args:
            context: Contexte avec les issues
            df: DataFrame original
            **kwargs: Options

        Returns:
            Contexte mis à jour avec les propositions
        """
        start_time = time.time()

        proposals = []

        # Pour chaque problème, proposer une correction
        for issue in context.issues:
            issue_proposals = self._propose_correction(issue, df, context)
            proposals.extend(issue_proposals)

        # Apprendre des feedbacks passés pour ajuster les propositions
        proposals = self._adjust_from_feedback(proposals, context)

        # Mettre à jour le contexte
        context.proposals = proposals
        context.current_step = "corrections_proposed"
        context.iteration += 1

        # Calculer la confiance
        confidence = self._calculate_correction_confidence(proposals)

        # Logger
        processing_time = int((time.time() - start_time) * 1000)
        self._log_decision(
            context=context,
            action="propose_corrections",
            reasoning=f"Proposé {len(proposals)} corrections pour {len(context.issues)} problèmes",
            input_summary=f"Issues: {len(context.issues)}",
            output_summary=self._summarize_proposals(proposals),
            confidence=confidence.overall_score,
            processing_time_ms=processing_time
        )

        return context

    def _propose_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame,
        context: AgentContext
    ) -> list[CorrectionProposal]:
        """
        Propose des corrections pour un problème spécifique.

        Args:
            issue: Problème à corriger
            df: DataFrame
            context: Contexte

        Returns:
            Liste de propositions de correction
        """
        proposals = []

        if issue.issue_type == IssueType.MISSING_VALUES:
            proposals = self._propose_missing_value_correction(issue, df)

        elif issue.issue_type == IssueType.ANOMALY:
            proposals = self._propose_anomaly_correction(issue, df)

        elif issue.issue_type == IssueType.TYPE_MISMATCH:
            proposals = self._propose_type_correction(issue, df)

        elif issue.issue_type == IssueType.CONSTRAINT_VIOLATION:
            proposals = self._propose_constraint_correction(issue, df)

        elif issue.issue_type == IssueType.DRIFT:
            proposals = self._propose_drift_correction(issue, df)

        else:
            # Proposition générique pour les autres types
            proposals = [self._propose_generic_correction(issue, df)]

        return proposals

    def _propose_missing_value_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame
    ) -> list[CorrectionProposal]:
        """Propose des corrections pour les valeurs manquantes."""
        proposals = []
        column = issue.column

        if column not in df.columns:
            return []

        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            # Option 1: Imputation par médiane (recommandée pour robustesse)
            median_val = series.median()
            proposals.append(CorrectionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                issue_id=issue.issue_id,
                correction_type=CorrectionType.IMPUTE_MEDIAN,
                description=f"Imputer les valeurs manquantes par la médiane ({median_val:.2f})",
                justification=(
                    "La médiane est robuste aux outliers et préserve la distribution centrale. "
                    "Recommandée quand des anomalies sont présentes dans les données."
                ),
                parameters={"value": float(median_val), "column": column},
                estimated_impact=f"Remplira {issue.affected_count} valeurs avec {median_val:.2f}",
                rows_affected=issue.affected_count,
                confidence=0.8,
                alternatives=[
                    "Imputation par moyenne (si distribution normale)",
                    "Suppression des lignes (si peu nombreuses)",
                    "Imputation par modèle (si relations fortes avec autres colonnes)"
                ]
            ))

            # Option 2: Imputation par moyenne
            mean_val = series.mean()
            proposals.append(CorrectionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                issue_id=issue.issue_id,
                correction_type=CorrectionType.IMPUTE_MEAN,
                description=f"Imputer les valeurs manquantes par la moyenne ({mean_val:.2f})",
                justification=(
                    "La moyenne est appropriée si la distribution est symétrique "
                    "et sans outliers significatifs."
                ),
                parameters={"value": float(mean_val), "column": column},
                estimated_impact=f"Remplira {issue.affected_count} valeurs avec {mean_val:.2f}",
                rows_affected=issue.affected_count,
                confidence=0.7,
                alternatives=["Imputation par médiane", "Suppression des lignes"]
            ))

        else:
            # Colonne catégorielle: imputation par mode
            mode_val = series.mode()
            if len(mode_val) > 0:
                mode_str = str(mode_val.iloc[0])
                proposals.append(CorrectionProposal(
                    proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                    issue_id=issue.issue_id,
                    correction_type=CorrectionType.IMPUTE_MODE,
                    description=f"Imputer les valeurs manquantes par le mode ('{mode_str}')",
                    justification=(
                        "Le mode est la valeur la plus fréquente. Approprié pour les "
                        "colonnes catégorielles à faible cardinalité."
                    ),
                    parameters={"value": mode_str, "column": column},
                    estimated_impact=f"Remplira {issue.affected_count} valeurs avec '{mode_str}'",
                    rows_affected=issue.affected_count,
                    confidence=0.7,
                    alternatives=[
                        "Créer une catégorie 'Unknown'",
                        "Supprimer les lignes concernées"
                    ]
                ))

        # Option: suppression si peu de lignes affectées
        if issue.affected_percentage < 5:
            proposals.append(CorrectionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                issue_id=issue.issue_id,
                correction_type=CorrectionType.DELETE_ROW,
                description=f"Supprimer les {issue.affected_count} lignes avec valeurs manquantes",
                justification=(
                    f"Avec seulement {issue.affected_percentage:.1f}% de lignes affectées, "
                    "la suppression a un impact minimal sur le dataset."
                ),
                parameters={"column": column, "indices": issue.row_indices},
                estimated_impact=f"Réduction de {issue.affected_count} lignes ({issue.affected_percentage:.1f}%)",
                rows_affected=issue.affected_count,
                confidence=0.85,
                alternatives=["Imputation par médiane/moyenne/mode"]
            ))

        return proposals

    def _propose_anomaly_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame
    ) -> list[CorrectionProposal]:
        """Propose des corrections pour les anomalies."""
        proposals = []
        column = issue.column

        if column not in df.columns:
            return []

        series = df[column]

        # Option 1: Écrêtage (clipping) aux percentiles
        q01 = series.quantile(0.01)
        q99 = series.quantile(0.99)

        proposals.append(CorrectionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            issue_id=issue.issue_id,
            correction_type=CorrectionType.CLIP_VALUES,
            description=f"Écrêter les valeurs extrêmes entre {q01:.2f} et {q99:.2f}",
            justification=(
                "L'écrêtage aux percentiles 1-99 préserve la distribution tout en "
                "limitant l'impact des valeurs extrêmes."
            ),
            parameters={
                "column": column,
                "lower": float(q01),
                "upper": float(q99)
            },
            estimated_impact=f"Les {issue.affected_count} anomalies seront ramenées dans la plage [{q01:.2f}, {q99:.2f}]",
            rows_affected=issue.affected_count,
            confidence=0.75,
            alternatives=["Suppression des lignes", "Marquage pour revue manuelle"]
        ))

        # Option 2: Marquage uniquement
        proposals.append(CorrectionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            issue_id=issue.issue_id,
            correction_type=CorrectionType.FLAG_ONLY,
            description="Marquer les anomalies dans une colonne dédiée sans les modifier",
            justification=(
                "Les anomalies peuvent être des valeurs légitimes (événements rares). "
                "Le marquage permet une revue manuelle avant correction."
            ),
            parameters={
                "column": column,
                "flag_column": f"{column}_is_anomaly",
                "indices": issue.row_indices
            },
            estimated_impact=f"Crée une colonne de flag pour {issue.affected_count} lignes",
            rows_affected=issue.affected_count,
            confidence=0.9,
            alternatives=["Écrêtage", "Suppression"]
        ))

        return proposals

    def _propose_type_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame
    ) -> list[CorrectionProposal]:
        """Propose des corrections pour les problèmes de type."""
        proposals = []
        column = issue.column

        proposals.append(CorrectionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            issue_id=issue.issue_id,
            correction_type=CorrectionType.CAST_TYPE,
            description=f"Convertir '{column}' en numérique avec gestion des erreurs",
            justification=(
                "La colonne contient majoritairement des valeurs numériques. "
                "Les valeurs non-convertibles seront marquées comme nulles."
            ),
            parameters={
                "column": column,
                "target_type": "numeric",
                "errors": "coerce"
            },
            estimated_impact=f"{issue.affected_count} valeurs non-numériques seront converties en NaN",
            rows_affected=issue.affected_count,
            confidence=0.8,
            alternatives=["Conserver en string", "Nettoyer manuellement les valeurs"]
        ))

        return proposals

    def _propose_constraint_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame
    ) -> list[CorrectionProposal]:
        """Propose des corrections pour les violations de contraintes."""
        proposals = []

        if "duplicate" in issue.description.lower():
            proposals.append(CorrectionProposal(
                proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
                issue_id=issue.issue_id,
                correction_type=CorrectionType.DELETE_ROW,
                description="Supprimer les lignes dupliquées (conserver la première occurrence)",
                justification="Les identifiants doivent être uniques. Les doublons peuvent causer des erreurs d'intégrité.",
                parameters={
                    "column": issue.column,
                    "keep": "first"
                },
                estimated_impact=f"Suppression de {issue.affected_count} lignes dupliquées",
                rows_affected=issue.affected_count,
                confidence=0.85,
                alternatives=["Conserver la dernière occurrence", "Revue manuelle"]
            ))

        proposals.append(CorrectionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            issue_id=issue.issue_id,
            correction_type=CorrectionType.MANUAL_REVIEW,
            description="Escalader pour revue manuelle des violations de contraintes",
            justification="Les violations de contraintes peuvent avoir des causes métier spécifiques nécessitant une expertise humaine.",
            parameters={"issue_id": issue.issue_id, "reason": "constraint_violation"},
            estimated_impact="Les lignes seront marquées pour revue humaine",
            rows_affected=issue.affected_count,
            confidence=0.95,
            alternatives=["Suppression automatique"]
        ))

        return proposals

    def _propose_drift_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame
    ) -> list[CorrectionProposal]:
        """Propose des corrections pour le drift."""
        # Le drift n'a pas de correction automatique simple
        return [CorrectionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            issue_id=issue.issue_id,
            correction_type=CorrectionType.MANUAL_REVIEW,
            description="Investiguer la cause du drift et décider de l'action appropriée",
            justification=(
                "Le drift de données peut avoir de nombreuses causes (changement de source, "
                "évolution métier, erreur de collecte). Une investigation est nécessaire."
            ),
            parameters={
                "column": issue.column,
                "drift_details": issue.details
            },
            estimated_impact="Nécessite une analyse approfondie avant action",
            rows_affected=issue.affected_count,
            confidence=0.6,
            alternatives=[
                "Réentraîner les modèles sur les nouvelles données",
                "Normaliser/standardiser les données",
                "Mettre à jour la référence"
            ]
        )]

    def _propose_generic_correction(
        self,
        issue: QualityIssue,
        df: pd.DataFrame
    ) -> CorrectionProposal:
        """Proposition générique pour les cas non couverts."""
        return CorrectionProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            issue_id=issue.issue_id,
            correction_type=CorrectionType.MANUAL_REVIEW,
            description=f"Revue manuelle requise pour: {issue.description}",
            justification="Ce type de problème nécessite une évaluation humaine.",
            parameters={"issue": issue.to_dict() if hasattr(issue, "to_dict") else str(issue)},
            estimated_impact="À déterminer après analyse",
            rows_affected=issue.affected_count,
            confidence=0.5,
            alternatives=[]
        )

    def _adjust_from_feedback(
        self,
        proposals: list[CorrectionProposal],
        context: AgentContext
    ) -> list[CorrectionProposal]:
        """
        Ajuste les propositions basé sur les feedbacks passés.

        Si des corrections similaires ont été rejetées dans le passé,
        réduire leur confiance.
        """
        for proposal in proposals:
            # Chercher des feedbacks sur des corrections similaires
            feedback_insights = self.feedback_store.learn_from_feedback(
                context=f"{proposal.correction_type.value} on {proposal.description}",
                target_type="proposal"
            )

            if feedback_insights.get("has_relevant_feedback"):
                adjustment = feedback_insights.get("confidence_adjustment", 0)
                proposal.confidence = max(0.1, min(1.0, proposal.confidence + adjustment))

                if adjustment < -0.1:
                    proposal.alternatives.insert(
                        0,
                        f"Note: Des corrections similaires ont été rejetées précédemment"
                    )

        return proposals

    def _calculate_correction_confidence(
        self,
        proposals: list[CorrectionProposal]
    ) -> Any:
        """Calcule la confiance globale dans les corrections."""
        if not proposals:
            return self._calculate_confidence(
                data_quality=0.5,
                sample_size=0,
                signal_scores=[0.5]
            )

        avg_confidence = sum(p.confidence for p in proposals) / len(proposals)

        return self._calculate_confidence(
            data_quality=avg_confidence,
            sample_size=len(proposals),
            signal_scores=[p.confidence for p in proposals],
            rule_coverage=0.7
        )

    def _summarize_proposals(self, proposals: list[CorrectionProposal]) -> str:
        """Résume les propositions."""
        if not proposals:
            return "Aucune correction proposée"

        by_type = {}
        for p in proposals:
            t = p.correction_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return f"Total: {len(proposals)} propositions | Par type: {by_type}"
