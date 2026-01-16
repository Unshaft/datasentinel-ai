"""
Agent Quality - Détection des problèmes de qualité.

Cet agent est responsable de l'identification des problèmes:
- Valeurs manquantes excessives
- Anomalies statistiques
- Drift de distribution
- Violations de contraintes de type
- Incohérences entre colonnes
"""

import json
import time
import uuid
from typing import Any

import pandas as pd

from src.agents.base import AgentResult, BaseAgent
from src.core.models import (
    AgentContext,
    AgentType,
    IssueType,
    QualityIssue,
    Severity,
)
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.drift_detector import DriftDetector, DriftSeverity
from src.tools.anomaly import create_anomaly_tools
from src.tools.rules import create_rules_tools


class QualityAgent(BaseAgent):
    """
    Agent spécialisé dans la détection des problèmes de qualité.

    Rôle: Identifier tous les problèmes de qualité dans le dataset
    en utilisant des méthodes statistiques et des règles métier.

    Problèmes détectés:
    - Valeurs nulles (seuils configurables)
    - Anomalies (Isolation Forest)
    - Drift (vs référence si disponible)
    - Violations de contraintes
    """

    # Seuils par défaut
    NULL_THRESHOLD_HIGH = 0.3      # 30% nulls = HIGH severity
    NULL_THRESHOLD_MEDIUM = 0.1   # 10% nulls = MEDIUM severity
    NULL_THRESHOLD_LOW = 0.01     # 1% nulls = LOW severity

    def __init__(self) -> None:
        """Initialise le Quality Agent."""
        super().__init__(
            agent_type=AgentType.QUALITY,
            tools=create_rules_tools()  # Tools pour règles métier
        )

        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DriftDetector()

    @property
    def system_prompt(self) -> str:
        """Prompt système du Quality Agent."""
        return """Tu es un Data Quality Agent expert dans la détection des problèmes de données.

Ton rôle est d'identifier et de qualifier les problèmes de qualité:
1. Valeurs manquantes (nulls, empty strings)
2. Anomalies statistiques (outliers, valeurs aberrantes)
3. Violations de types (formats incorrects)
4. Incohérences (valeurs contradictoires entre colonnes)
5. Drift de distribution (changement par rapport à une référence)

Pour chaque problème détecté, tu dois fournir:
- Le TYPE de problème (missing_values, anomaly, type_mismatch, etc.)
- La SÉVÉRITÉ (low, medium, high, critical)
- La LOCALISATION (colonnes et lignes concernées)
- Une DESCRIPTION claire du problème
- Un niveau de CONFIANCE dans la détection

Tu dois être:
- EXHAUSTIF: Ne manquer aucun problème significatif
- PRÉCIS: Localiser exactement les problèmes
- PROPORTIONNÉ: Sévérité adaptée à l'impact réel
- EXPLICITE: Décrire clairement pourquoi c'est un problème

Tu ne dois PAS:
- Proposer de corrections (c'est le rôle du Corrector)
- Ignorer les problèmes mineurs (les signaler avec LOW severity)
"""

    def execute(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> AgentContext:
        """
        Détecte tous les problèmes de qualité.

        Args:
            context: Contexte avec le profil
            df: DataFrame à analyser
            **kwargs: Options (detect_anomalies, detect_drift, etc.)

        Returns:
            Contexte mis à jour avec les issues
        """
        start_time = time.time()

        issues = []

        # 1. Détection des valeurs manquantes
        null_issues = self._detect_missing_values(df, context)
        issues.extend(null_issues)

        # 2. Détection des anomalies (si activé)
        if kwargs.get("detect_anomalies", True):
            anomaly_issues = self._detect_anomalies(df, context)
            issues.extend(anomaly_issues)

        # 3. Détection des problèmes de type
        type_issues = self._detect_type_issues(df, context)
        issues.extend(type_issues)

        # 4. Détection du drift (si référence fournie)
        if kwargs.get("detect_drift", False):
            reference_df = kwargs.get("reference_df")
            if reference_df is not None:
                drift_issues = self._detect_drift(df, reference_df, context)
                issues.extend(drift_issues)

        # 5. Validation contre les règles métier
        rule_issues = self._validate_against_rules(df, context)
        issues.extend(rule_issues)

        # Mettre à jour le contexte
        context.issues = issues
        context.current_step = "quality_checked"
        context.iteration += 1

        # Calculer la confiance globale
        confidence = self._calculate_quality_confidence(df, issues)

        # Logger
        processing_time = int((time.time() - start_time) * 1000)
        self._log_decision(
            context=context,
            action="detect_quality_issues",
            reasoning=f"Détecté {len(issues)} problèmes de qualité",
            input_summary=f"Dataset: {len(df)} lignes",
            output_summary=self._summarize_issues(issues),
            confidence=confidence.overall_score,
            processing_time_ms=processing_time
        )

        return context

    def _detect_missing_values(
        self,
        df: pd.DataFrame,
        context: AgentContext
    ) -> list[QualityIssue]:
        """Détecte les problèmes de valeurs manquantes."""
        issues = []

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_pct = null_count / len(df) if len(df) > 0 else 0

            if null_pct > 0:
                # Déterminer la sévérité
                if null_pct >= self.NULL_THRESHOLD_HIGH:
                    severity = Severity.HIGH
                elif null_pct >= self.NULL_THRESHOLD_MEDIUM:
                    severity = Severity.MEDIUM
                elif null_pct >= self.NULL_THRESHOLD_LOW:
                    severity = Severity.LOW
                else:
                    continue  # Trop peu de nulls pour signaler

                # Identifier les indices des lignes avec nulls
                null_indices = df[df[col].isna()].index.tolist()

                issue = QualityIssue(
                    issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                    issue_type=IssueType.MISSING_VALUES,
                    severity=severity,
                    column=col,
                    row_indices=null_indices[:100],  # Limiter pour performance
                    description=f"Colonne '{col}' contient {null_count} valeurs manquantes ({null_pct:.1%})",
                    details={
                        "null_count": int(null_count),
                        "null_percentage": round(null_pct * 100, 2),
                        "total_rows": len(df)
                    },
                    affected_count=int(null_count),
                    affected_percentage=round(null_pct * 100, 2),
                    confidence=0.95,  # Détection certaine
                    detected_by=AgentType.QUALITY
                )
                issues.append(issue)

        return issues

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        context: AgentContext
    ) -> list[QualityIssue]:
        """Détecte les anomalies statistiques."""
        issues = []

        try:
            # Entraîner et détecter
            results = self.anomaly_detector.fit_detect(df)

            for result in results:
                if result.anomaly_count > 0:
                    # Déterminer la sévérité basée sur le pourcentage
                    if result.anomaly_percentage > 20:
                        severity = Severity.HIGH
                    elif result.anomaly_percentage > 5:
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW

                    issue = QualityIssue(
                        issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                        issue_type=IssueType.ANOMALY,
                        severity=severity,
                        column=result.column,
                        row_indices=result.anomaly_indices[:100],
                        description=(
                            f"Détecté {result.anomaly_count} anomalies dans '{result.column}' "
                            f"({result.anomaly_percentage:.1f}% des valeurs)"
                        ),
                        details={
                            "anomaly_count": result.anomaly_count,
                            "anomaly_percentage": result.anomaly_percentage,
                            "anomaly_values_sample": result.anomaly_values[:10],
                            "anomaly_scores_sample": [
                                round(s, 4) for s in result.anomaly_scores[:10]
                            ],
                            "model_id": result.model_id
                        },
                        affected_count=result.anomaly_count,
                        affected_percentage=result.anomaly_percentage,
                        confidence=0.75,  # Anomaly detection a une incertitude
                        detected_by=AgentType.QUALITY
                    )
                    issues.append(issue)

        except Exception as e:
            # Log l'erreur mais continue
            context.metadata["anomaly_detection_error"] = str(e)

        return issues

    def _detect_type_issues(
        self,
        df: pd.DataFrame,
        context: AgentContext
    ) -> list[QualityIssue]:
        """Détecte les problèmes de type de données."""
        issues = []

        for col in df.columns:
            series = df[col]

            # Vérifier les colonnes object qui pourraient être numériques
            if series.dtype == "object":
                # Tenter la conversion numérique
                numeric_converted = pd.to_numeric(series, errors="coerce")
                convertible = numeric_converted.notna().sum()
                non_null = series.notna().sum()

                if convertible > 0 and convertible < non_null * 0.9:
                    # Certaines valeurs sont numériques, d'autres non
                    non_numeric_mask = series.notna() & numeric_converted.isna()
                    non_numeric_indices = df[non_numeric_mask].index.tolist()
                    non_numeric_values = series[non_numeric_mask].head(10).tolist()

                    issue = QualityIssue(
                        issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=Severity.MEDIUM,
                        column=col,
                        row_indices=non_numeric_indices[:100],
                        description=(
                            f"Colonne '{col}' contient un mélange de valeurs numériques "
                            f"et non-numériques ({len(non_numeric_indices)} incohérences)"
                        ),
                        details={
                            "numeric_count": int(convertible),
                            "non_numeric_count": len(non_numeric_indices),
                            "non_numeric_samples": [str(v) for v in non_numeric_values]
                        },
                        affected_count=len(non_numeric_indices),
                        affected_percentage=round(
                            100 * len(non_numeric_indices) / non_null, 2
                        ) if non_null > 0 else 0,
                        confidence=0.85,
                        detected_by=AgentType.QUALITY
                    )
                    issues.append(issue)

        return issues

    def _detect_drift(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        context: AgentContext
    ) -> list[QualityIssue]:
        """Détecte le drift par rapport à une référence."""
        issues = []

        try:
            self.drift_detector.set_reference(reference_df)
            results = self.drift_detector.detect(df)

            for result in results:
                if result.has_drift:
                    # Mapper la sévérité du drift
                    severity_map = {
                        DriftSeverity.LOW: Severity.LOW,
                        DriftSeverity.MEDIUM: Severity.MEDIUM,
                        DriftSeverity.HIGH: Severity.HIGH,
                        DriftSeverity.CRITICAL: Severity.CRITICAL,
                    }
                    severity = severity_map.get(result.severity, Severity.MEDIUM)

                    issue = QualityIssue(
                        issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                        issue_type=IssueType.DRIFT,
                        severity=severity,
                        column=result.column,
                        row_indices=[],  # Le drift affecte toute la colonne
                        description=result.interpretation,
                        details={
                            "drift_score": result.drift_score,
                            "p_value": result.p_value,
                            "test_used": result.test_used,
                            "reference_stats": result.reference_stats,
                            "current_stats": result.current_stats
                        },
                        affected_count=len(df),
                        affected_percentage=100.0,
                        confidence=1 - (result.p_value or 0.5),  # Confiance basée sur p-value
                        detected_by=AgentType.QUALITY
                    )
                    issues.append(issue)

        except Exception as e:
            context.metadata["drift_detection_error"] = str(e)

        return issues

    def _validate_against_rules(
        self,
        df: pd.DataFrame,
        context: AgentContext
    ) -> list[QualityIssue]:
        """Valide le dataset contre les règles métier."""
        issues = []

        # Pour chaque colonne, chercher les règles applicables
        # et vérifier les violations

        # Exemple: vérifier les IDs uniques
        for col in df.columns:
            col_lower = col.lower()

            # Heuristique: colonnes ID doivent être uniques
            if "id" in col_lower or col_lower.endswith("_id"):
                duplicates = df[col].duplicated()
                dup_count = duplicates.sum()

                if dup_count > 0:
                    dup_indices = df[duplicates].index.tolist()

                    issue = QualityIssue(
                        issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                        issue_type=IssueType.CONSTRAINT_VIOLATION,
                        severity=Severity.HIGH,
                        column=col,
                        row_indices=dup_indices[:100],
                        description=(
                            f"Colonne '{col}' (identifiant) contient {dup_count} "
                            f"valeurs dupliquées"
                        ),
                        details={
                            "duplicate_count": int(dup_count),
                            "rule_violated": "unique_id_constraint",
                            "duplicate_values": df[col][duplicates].head(10).tolist()
                        },
                        affected_count=int(dup_count),
                        affected_percentage=round(100 * dup_count / len(df), 2),
                        confidence=0.9,
                        detected_by=AgentType.QUALITY
                    )
                    issues.append(issue)

        return issues

    def _calculate_quality_confidence(
        self,
        df: pd.DataFrame,
        issues: list[QualityIssue]
    ) -> Any:
        """Calcule la confiance globale dans l'analyse de qualité."""
        # Agréger les confiances des issues
        if issues:
            avg_confidence = sum(i.confidence for i in issues) / len(issues)
            signal_scores = [i.confidence for i in issues]
        else:
            avg_confidence = 0.9  # Haute confiance si pas de problèmes
            signal_scores = [0.9]

        return self._calculate_confidence(
            data_quality=avg_confidence,
            sample_size=len(df),
            signal_scores=signal_scores,
            rule_coverage=0.8  # Estimation de la couverture des règles
        )

    def _summarize_issues(self, issues: list[QualityIssue]) -> str:
        """Génère un résumé des problèmes détectés."""
        if not issues:
            return "Aucun problème de qualité détecté"

        by_severity = {}
        by_type = {}

        for issue in issues:
            sev = issue.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            typ = issue.issue_type.value
            by_type[typ] = by_type.get(typ, 0) + 1

        parts = [f"Total: {len(issues)} problèmes"]
        parts.append(f"Par sévérité: {by_severity}")
        parts.append(f"Par type: {by_type}")

        return " | ".join(parts)

    def analyze_with_llm(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> tuple[AgentContext, str]:
        """
        Analyse les problèmes avec interprétation LLM.

        En plus de la détection automatique, demande au LLM
        d'interpréter les problèmes et de prioriser.

        Args:
            context: Contexte
            df: DataFrame

        Returns:
            Tuple (contexte, analyse LLM)
        """
        # D'abord, exécuter la détection standard
        context = self.execute(context, df)

        # Préparer le prompt
        issues_summary = json.dumps([
            {
                "type": i.issue_type.value,
                "severity": i.severity.value,
                "column": i.column,
                "description": i.description,
                "affected_percentage": i.affected_percentage
            }
            for i in context.issues[:20]  # Limiter
        ], indent=2)

        prompt = f"""Voici les problèmes de qualité détectés dans un dataset:

{issues_summary}

Analyse ces problèmes et fournis:
1. Une priorisation (quel problème traiter en premier)
2. Les impacts potentiels de chaque problème
3. Des recommandations générales

Sois concis et actionnable."""

        response = self._invoke_llm(prompt, include_tools=False)
        llm_analysis = response.content

        context.metadata["quality_llm_analysis"] = llm_analysis

        return context, llm_analysis
