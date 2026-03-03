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
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.agents.base import AgentResult, BaseAgent

logger = logging.getLogger(__name__)
from src.core.models import (
    AgentContext,
    AgentType,
    IssueType,
    QualityIssue,
    Severity,
)
from src.memory.chroma_store import get_chroma_store
from src.ml.anomaly_detector import AnomalyDetector
from src.ml.drift_detector import DriftDetector, DriftSeverity
from src.tools.anomaly import create_anomaly_tools
from src.tools.rules import create_rules_tools


@dataclass
class RuleContext:
    """Contexte de règles pour une colonne (Active RAG — F25)."""
    rules: list[str] = field(default_factory=list)
    null_threshold_override: float | None = None
    severity_override: "Severity | None" = None
    format_tolerance_override: float | None = None


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

    # Valeurs considérées comme pseudo-nulls (masquées comme données valides)
    _PSEUDO_NULL_VALUES: frozenset = frozenset({
        "n/a", "na", "null", "none", "-", "--", "?", "??",
        "nan", "#n/a", "#na", "missing", "unknown", "inconnu",
        "nd", "nr", "nc", "n.a.", "n.a", "not available",
    })

    # Patterns de validation de format par type de colonne (détection via nom)
    _FORMAT_PATTERNS: dict = {
        "email": (
            re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
            ["email", "mail", "courriel", "e_mail", "e-mail"],
        ),
        "phone": (
            re.compile(r"^(\+?33|0033|0)[1-9](\s?\d{2}){4}$"),
            ["phone", "tel", "telephone", "mobile", "portable", "fax"],
        ),
        "url": (
            re.compile(r"^https?://[^\s]{3,}$"),
            ["url", "web", "site", "link", "href", "website"],
        ),
        "postal_fr": (
            re.compile(r"^\d{5}$"),
            ["zip", "postal", "cp", "code_postal", "codepostal", "zipcode"],
        ),
        "siret": (
            re.compile(r"^\d{14}$"),
            ["siret"],
        ),
        "siren": (
            re.compile(r"^\d{9}$"),
            ["siren"],
        ),
    }

    def __init__(self) -> None:
        """Initialise le Quality Agent."""
        super().__init__(
            agent_type=AgentType.QUALITY,
            tools=create_rules_tools()  # Tools pour règles métier
        )

        self.anomaly_detector = AnomalyDetector()
        self.drift_detector = DriftDetector()
        # Accès direct au store pour la validation de règles métier dans
        # _validate_against_rules (sans passer par le LLM).
        self.store = get_chroma_store()
        # Cache RuleContext par (col_name, col_type) pour la session courante (F25)
        self._rule_context_cache: dict[str, RuleContext] = {}
        # Ajustements de confiance issus du FeedbackProcessor (F26)
        self._confidence_adjustments: dict[str, float] = self._load_confidence_adjustments()

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

    @staticmethod
    def _load_confidence_adjustments() -> dict[str, float]:
        """Charge les ajustements de confiance issus du feedback (F26, best-effort)."""
        try:
            import json as _json
            from pathlib import Path as _Path
            fp = _Path("./data/feedback_stats.json")
            if fp.exists():
                data = _json.loads(fp.read_text(encoding="utf-8"))
                return data.get("confidence_adjustments", {})
        except Exception:
            pass
        return {}

    def _get_rule_context(self, col_name: str, col_type: str) -> RuleContext:
        """
        Requête Active RAG : récupère les règles pertinentes pour une colonne.
        Résultats mis en cache par (col_name, col_type) pendant la session (F25).
        """
        cache_key = f"{col_name}:{col_type}"
        if cache_key in self._rule_context_cache:
            return self._rule_context_cache[cache_key]

        ctx = RuleContext()
        try:
            rules = self.store.get_relevant_rules(col_name, col_type, top_k=3)
            # Seuil minimal de pertinence (même que _validate_against_rules)
            relevant = [r for r in rules if r.get("similarity", 0) >= 0.55]
            if not relevant:
                self._rule_context_cache[cache_key] = ctx
                return ctx

            ctx.rules = [r["text"] for r in relevant]
            combined = " ".join(ctx.rules).lower()

            # Parsing des overrides
            if any(kw in combined for kw in ("obligatoire", "required", "non null", "not null")):
                ctx.null_threshold_override = 0.01
            elif any(kw in combined for kw in ("optionnel", "nullable", "peut être null")):
                ctx.null_threshold_override = 0.8

            if any(kw in combined for kw in ("identifiant unique", "clé primaire", "primary key", "unique key")):
                ctx.severity_override = Severity.CRITICAL

            if any(kw in combined for kw in ("format strict", "strict format")):
                ctx.format_tolerance_override = 0.0

        except Exception:
            pass  # Best-effort — ne jamais bloquer l'analyse

        self._rule_context_cache[cache_key] = ctx
        return ctx

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
        # Réinitialise le cache de règles par session (F25)
        self._rule_context_cache = {}
        start_time = time.time()

        issues = []

        # 1. Détection des valeurs manquantes
        null_issues = self._detect_missing_values(df, context)
        issues.extend(null_issues)

        # 2. Détection des pseudo-nulls (v0.4)
        issues.extend(self._detect_pseudo_nulls(df, context))

        # 3. Détection des anomalies (si activé)
        if kwargs.get("detect_anomalies", True):
            anomaly_issues = self._detect_anomalies(df, context)
            issues.extend(anomaly_issues)

        # 4. Détection des problèmes de type
        type_issues = self._detect_type_issues(df, context)
        issues.extend(type_issues)

        # 5. Doublons complets (v0.4)
        issues.extend(self._detect_duplicate_rows(df, context))

        # 6. Validation format (email, téléphone, etc.) (v0.4)
        issues.extend(self._detect_format_issues(df, context))

        # 7. Détection du drift (si référence fournie)
        if kwargs.get("detect_drift", False):
            reference_df = kwargs.get("reference_df")
            if reference_df is not None:
                drift_issues = self._detect_drift(df, reference_df, context)
                issues.extend(drift_issues)

        # 8. Validation contre les règles métier
        rule_issues = self._validate_against_rules(df, context)
        issues.extend(rule_issues)

        _tmap: dict[str, int] = {}
        for _iss in issues:
            _tmap[_iss.issue_type.value] = _tmap.get(_iss.issue_type.value, 0) + 1
        logger.info(
            "%d issues — %s (%dms)",
            len(issues),
            " ".join(f"{k}={v}" for k, v in sorted(_tmap.items())) or "none",
            int((time.time() - start_time) * 1000),
        )

        # Mettre à jour le contexte
        context.issues = issues
        # Score par colonne (v0.4)
        context.metadata["column_scores"] = self._compute_column_scores(df, issues)
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
                # Active RAG : récupère les overrides de règles (F25)
                col_type = str(df[col].dtype)
                rule_ctx = self._get_rule_context(col, col_type)
                null_high = rule_ctx.null_threshold_override or self.NULL_THRESHOLD_HIGH
                null_med = rule_ctx.null_threshold_override or self.NULL_THRESHOLD_MEDIUM
                null_low = rule_ctx.null_threshold_override or self.NULL_THRESHOLD_LOW

                # Déterminer la sévérité (override possible via règle)
                if rule_ctx.severity_override:
                    severity = rule_ctx.severity_override
                elif null_pct >= null_high:
                    severity = Severity.HIGH
                elif null_pct >= null_med:
                    severity = Severity.MEDIUM
                elif null_pct >= null_low:
                    severity = Severity.LOW
                else:
                    continue  # Trop peu de nulls pour signaler

                # Identifier les indices des lignes avec nulls
                null_indices = df[df[col].isna()].index.tolist()

                details: dict[str, Any] = {
                    "null_count": int(null_count),
                    "null_percentage": round(null_pct * 100, 2),
                    "total_rows": len(df)
                }
                if rule_ctx.rules:
                    details["applied_rules"] = rule_ctx.rules

                issue = QualityIssue(
                    issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                    issue_type=IssueType.MISSING_VALUES,
                    severity=severity,
                    column=col,
                    row_indices=null_indices[:100],  # Limiter pour performance
                    description=f"Colonne '{col}' contient {null_count} valeurs manquantes ({null_pct:.1%})",
                    details=details,
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

    def _detect_duplicate_rows(
        self,
        df: pd.DataFrame,
        context: AgentContext,
    ) -> list[QualityIssue]:
        """Détecte les lignes entièrement dupliquées."""
        if len(df) < 2:
            return []

        dup_mask = df.duplicated(keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count == 0:
            return []

        pct = round(dup_count / len(df) * 100, 2)
        severity = (
            Severity.HIGH if pct > 10
            else Severity.MEDIUM if pct > 1
            else Severity.LOW
        )
        return [QualityIssue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            issue_type=IssueType.DUPLICATE,
            severity=severity,
            column=None,
            row_indices=df[dup_mask].index.tolist()[:100],
            description=f"{dup_count} lignes entièrement dupliquées ({pct:.1f}% du dataset)",
            details={
                "duplicate_count": dup_count,
                "unique_rows": len(df) - dup_count // 2,
            },
            affected_count=dup_count,
            affected_percentage=pct,
            confidence=1.0,
            detected_by=AgentType.QUALITY,
        )]

    def _detect_pseudo_nulls(
        self,
        df: pd.DataFrame,
        context: AgentContext,
    ) -> list[QualityIssue]:
        """Détecte les pseudo-nulls masqués comme données valides ('N/A', 'null', '-', etc.)."""
        issues = []
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            normalized = df[col].dropna().astype(str).str.strip().str.lower()
            mask_full = (
                df[col].notna()
                & df[col].astype(str).str.strip().str.lower().isin(self._PSEUDO_NULL_VALUES)
            )
            count = int(mask_full.sum())
            if count == 0:
                continue
            pct = round(count / len(df) * 100, 2)
            severity = (
                Severity.HIGH if pct > 30
                else Severity.MEDIUM if pct > 10
                else Severity.LOW
            )
            found_values = (
                df.loc[mask_full, col].astype(str).value_counts().head(5).to_dict()
            )
            issues.append(QualityIssue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                issue_type=IssueType.MISSING_VALUES,
                severity=severity,
                column=col,
                row_indices=df[mask_full].index.tolist()[:100],
                description=(
                    f"'{col}' contient {count} pseudo-nulls ({pct:.1f}%) — "
                    f"valeurs masquées comme données valides"
                ),
                details={"pseudo_null_values": found_values, "pseudo_null_count": count},
                affected_count=count,
                affected_percentage=pct,
                confidence=0.92,
                detected_by=AgentType.QUALITY,
            ))
        return issues

    def _detect_format_issues(
        self,
        df: pd.DataFrame,
        context: AgentContext,
    ) -> list[QualityIssue]:
        """Détecte les incohérences de format (emails, téléphones, codes postaux, etc.)."""
        issues = []
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            col_lower = col.lower()
            matched_format: str | None = None
            matched_pattern = None
            for fmt_name, (pattern, keywords) in self._FORMAT_PATTERNS.items():
                if any(kw in col_lower for kw in keywords):
                    matched_format = fmt_name
                    matched_pattern = pattern
                    break
            if matched_pattern is None:
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            # Ignore pseudo-nulls before format check
            valid_strings = non_null.astype(str).str.strip()
            valid_strings = valid_strings[~valid_strings.str.lower().isin(self._PSEUDO_NULL_VALUES)]
            if len(valid_strings) == 0:
                continue
            invalid_mask = ~valid_strings.str.match(matched_pattern)
            invalid_count = int(invalid_mask.sum())
            if invalid_count == 0:
                continue
            pct = round(invalid_count / len(valid_strings) * 100, 2)
            if pct < 5:  # Moins de 5% → vraisemblablement des exceptions légitimes
                continue
            severity = (
                Severity.HIGH if pct > 50
                else Severity.MEDIUM if pct > 20
                else Severity.LOW
            )
            examples = valid_strings[invalid_mask].head(3).tolist()
            issues.append(QualityIssue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                issue_type=IssueType.FORMAT_ERROR,
                severity=severity,
                column=col,
                row_indices=[],
                description=(
                    f"'{col}' : {invalid_count} valeurs au format {matched_format} "
                    f"invalide ({pct:.1f}%)"
                ),
                details={
                    "format_expected": matched_format,
                    "invalid_count": invalid_count,
                    "examples": examples,
                },
                affected_count=invalid_count,
                affected_percentage=pct,
                confidence=0.87,
                detected_by=AgentType.QUALITY,
            ))
        return issues

    def _compute_column_scores(
        self,
        df: pd.DataFrame,
        issues: list[QualityIssue],
    ) -> dict[str, float]:
        """Calcule un score de qualité individuel (0-100) par colonne."""
        scores: dict[str, float] = {col: 100.0 for col in df.columns}
        deductions = {
            Severity.CRITICAL: 40,
            Severity.HIGH: 25,
            Severity.MEDIUM: 12,
            Severity.LOW: 5,
        }
        for issue in issues:
            if issue.column is None:
                continue
            if issue.column not in scores:
                continue
            scores[issue.column] = max(0.0, scores[issue.column] - deductions.get(issue.severity, 0))
        return {col: round(score, 1) for col, score in scores.items()}

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
        """
        Valide le dataset contre les règles métier.

        Deux niveaux de validation :
        1. Heuristique embarquée (unicité des IDs) — fonctionne toujours,
           même sans règles dans ChromaDB.
        2. Règles sémantiques via RAG (ChromaDB) — enrichit la détection
           quand la base de règles est alimentée.
        """
        issues = []

        # --- Niveau 1 : heuristique sur les colonnes identifiants ---
        for col in df.columns:
            col_lower = col.lower()
            if "id" in col_lower or col_lower.endswith("_id"):
                duplicates = df[col].duplicated()
                dup_count = duplicates.sum()
                if dup_count > 0:
                    dup_indices = df[duplicates].index.tolist()
                    issues.append(QualityIssue(
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
                    ))

        # --- Niveau 2 : règles sémantiques via ChromaDB (RAG) ---
        # On interroge la base pour chaque colonne ; si une règle pertinente
        # existe et que des valeurs la violent, on crée un QualityIssue.
        try:
            severity_map = {
                "critical": Severity.CRITICAL,
                "high": Severity.HIGH,
                "medium": Severity.MEDIUM,
                "low": Severity.LOW,
            }

            for col in df.columns:
                query = f"column {col} constraint validation rule"
                rules = self.store.search_rules(query=query, n_results=3)

                for rule in rules:
                    # Seuil de pertinence : on ignore les règles trop éloignées.
                    if rule.get("similarity", 0) < 0.55:
                        continue

                    rule_text = rule["text"].lower()
                    rule_meta = rule.get("metadata", {})
                    severity = severity_map.get(
                        rule_meta.get("severity", "medium"), Severity.MEDIUM
                    )

                    # Vérification : unicité explicitement requise par la règle
                    if "unique" in rule_text or "unicité" in rule_text:
                        duplicates = df[col].duplicated()
                        dup_count = int(duplicates.sum())
                        if dup_count > 0:
                            issues.append(QualityIssue(
                                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                                issue_type=IssueType.CONSTRAINT_VIOLATION,
                                severity=severity,
                                column=col,
                                row_indices=df[duplicates].index.tolist()[:100],
                                description=(
                                    f"Règle métier '{rule['id']}' exige l'unicité "
                                    f"sur '{col}' ({dup_count} doublons détectés)"
                                ),
                                details={
                                    "rule_id": rule["id"],
                                    "rule_text": rule["text"],
                                    "duplicate_count": dup_count,
                                    "similarity": round(rule.get("similarity", 0), 4),
                                },
                                affected_count=dup_count,
                                affected_percentage=round(100 * dup_count / len(df), 2),
                                confidence=round(rule.get("similarity", 0.6), 4),
                                detected_by=AgentType.QUALITY
                            ))

                    # Vérification : valeurs non-nulles requises
                    elif "not null" in rule_text or "non null" in rule_text or "obligatoire" in rule_text:
                        null_count = int(df[col].isna().sum())
                        if null_count > 0:
                            issues.append(QualityIssue(
                                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                                issue_type=IssueType.CONSTRAINT_VIOLATION,
                                severity=severity,
                                column=col,
                                row_indices=df[df[col].isna()].index.tolist()[:100],
                                description=(
                                    f"Règle métier '{rule['id']}' interdit les nulls "
                                    f"sur '{col}' ({null_count} valeurs manquantes)"
                                ),
                                details={
                                    "rule_id": rule["id"],
                                    "rule_text": rule["text"],
                                    "null_count": null_count,
                                    "similarity": round(rule.get("similarity", 0), 4),
                                },
                                affected_count=null_count,
                                affected_percentage=round(100 * null_count / len(df), 2),
                                confidence=round(rule.get("similarity", 0.6), 4),
                                detected_by=AgentType.QUALITY
                            ))

        except Exception as e:
            # ChromaDB indisponible → on se contente de l'heuristique niveau 1.
            context.metadata["rules_validation_error"] = str(e)

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

    async def execute_async(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> AgentContext:
        """
        Version asynchrone de execute() — détections en parallèle.

        Lance les 4 vérifications de qualité indépendantes en concurrence
        via asyncio.to_thread, réduisant la latence d'environ 40-50%.

        Identique à execute() en termes de résultats, mais les checks
        missing_values / anomalies / type_issues / rules_validation
        s'exécutent dans des threads séparés simultanement.
        """
        import asyncio
        import time

        start_time = time.time()

        # Tâches indépendantes — peuvent tourner en parallèle
        tasks: list = [
            asyncio.to_thread(self._detect_missing_values, df, context),
            asyncio.to_thread(self._detect_pseudo_nulls, df, context),
            asyncio.to_thread(self._detect_type_issues, df, context),
            asyncio.to_thread(self._detect_duplicate_rows, df, context),
            asyncio.to_thread(self._detect_format_issues, df, context),
            asyncio.to_thread(self._validate_against_rules, df, context),
        ]

        if kwargs.get("detect_anomalies", True):
            tasks.append(asyncio.to_thread(self._detect_anomalies, df, context))

        if kwargs.get("detect_drift", False):
            reference_df = kwargs.get("reference_df")
            if reference_df is not None:
                tasks.append(
                    asyncio.to_thread(self._detect_drift, df, reference_df, context)
                )

        results = await asyncio.gather(*tasks)

        issues: list = []
        for result_list in results:
            issues.extend(result_list)

        _tmap: dict[str, int] = {}
        for _iss in issues:
            _tmap[_iss.issue_type.value] = _tmap.get(_iss.issue_type.value, 0) + 1
        logger.info(
            "%d issues — %s (%dms)",
            len(issues),
            " ".join(f"{k}={v}" for k, v in sorted(_tmap.items())) or "none",
            int((time.time() - start_time) * 1000),
        )

        # 9. LLM semantic check (opt-in F23 — skippé si F27 a déjà tourné)
        from src.core.config import settings as _settings
        if _settings.enable_llm_checks and "semantic_types" not in context.metadata:
            try:
                llm_issues = await self._detect_semantic_anomalies_llm(df, context)
                issues.extend(llm_issues)
            except Exception:
                pass

        # 10. Semantic-type validators (F28 — lit context.metadata["semantic_types"])
        try:
            semantic_issues = self._validate_semantic_types(df, context)
            issues.extend(semantic_issues)
        except Exception:  # noqa: BLE001
            pass

        context.issues = issues
        # Score par colonne (v0.4)
        context.metadata["column_scores"] = self._compute_column_scores(df, issues)
        context.current_step = "quality_checked"
        context.iteration += 1

        confidence = self._calculate_quality_confidence(df, issues)
        processing_time = int((time.time() - start_time) * 1000)

        self._log_decision(
            context=context,
            action="detect_quality_issues_async",
            reasoning=f"Détecté {len(issues)} problèmes (parallel checks)",
            input_summary=f"Dataset: {len(df)} lignes, {len(tasks)} checks parallèles",
            output_summary=self._summarize_issues(issues),
            confidence=confidence.overall_score,
            processing_time_ms=processing_time,
        )

        return context

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

    def _validate_semantic_types(
        self,
        df: pd.DataFrame,
        context: AgentContext,
    ) -> list[QualityIssue]:
        """
        Applique des règles métier basées sur les types sémantiques LLM (F28 — v0.8).

        Lit context.metadata["semantic_types"] (populé par SemanticProfilerAgent).
        Applique des validateurs ciblés selon le semantic_type de chaque colonne.
        Skip silencieusement si semantic_types est absent ou vide.

        Règles appliquées :
        - monetary_amount : valeurs négatives → ANOMALY MEDIUM
        - percentage       : hors [0, 100] → ANOMALY MEDIUM
        - age              : hors [0, 150] → ANOMALY MEDIUM
        - email/phone/url/postal_code : format check via regex existant si nom de col ne matche pas
        """
        semantic_types: dict = context.metadata.get("semantic_types", {})
        if not semantic_types:
            return []

        issues: list[QualityIssue] = []

        # Regexes réutilisées depuis _FORMAT_PATTERNS
        format_re = {k: v[0] for k, v in self._FORMAT_PATTERNS.items()}

        for col, sem_info in semantic_types.items():
            if col not in df.columns:
                continue
            sem_type = sem_info.get("semantic_type", "")
            confidence = float(sem_info.get("confidence", 0.0))
            if confidence < 0.7:
                continue

            base_details = {
                "semantic_type": sem_type,
                "detection_method": "semantic",
                "confidence": confidence,
            }
            col_series = df[col].dropna()
            total = len(df)
            if total == 0:
                continue

            # ── Règle : monetary_amount → pas de valeurs négatives ────────────
            if sem_type == "monetary_amount":
                try:
                    numeric = pd.to_numeric(col_series, errors="coerce").dropna()
                    neg_mask = numeric < 0
                    if neg_mask.any():
                        neg_count = int(neg_mask.sum())
                        issues.append(QualityIssue(
                            issue_id=f"sem_neg_{col}_{uuid.uuid4().hex[:8]}",
                            issue_type=IssueType.ANOMALY,
                            severity=Severity.MEDIUM,
                            column=col,
                            description=(
                                f"Colonne '{col}' (monetary_amount) contient "
                                f"{neg_count} valeur(s) négative(s)."
                            ),
                            details={**base_details, "negative_count": neg_count},
                            affected_count=neg_count,
                            affected_percentage=round(neg_count / total * 100, 2),
                            confidence=confidence,
                            detected_by=AgentType.QUALITY,
                        ))
                except Exception:  # noqa: BLE001
                    pass

            # ── Règle : percentage → [0, 100] ─────────────────────────────────
            elif sem_type == "percentage":
                try:
                    numeric = pd.to_numeric(col_series, errors="coerce").dropna()
                    out_mask = (numeric < 0) | (numeric > 100)
                    if out_mask.any():
                        out_count = int(out_mask.sum())
                        issues.append(QualityIssue(
                            issue_id=f"sem_pct_{col}_{uuid.uuid4().hex[:8]}",
                            issue_type=IssueType.ANOMALY,
                            severity=Severity.MEDIUM,
                            column=col,
                            description=(
                                f"Colonne '{col}' (percentage) contient "
                                f"{out_count} valeur(s) hors plage [0, 100]."
                            ),
                            details={**base_details, "out_of_range_count": out_count},
                            affected_count=out_count,
                            affected_percentage=round(out_count / total * 100, 2),
                            confidence=confidence,
                            detected_by=AgentType.QUALITY,
                        ))
                except Exception:  # noqa: BLE001
                    pass

            # ── Règle : age → [0, 150] ────────────────────────────────────────
            elif sem_type == "age":
                try:
                    numeric = pd.to_numeric(col_series, errors="coerce").dropna()
                    out_mask = (numeric < 0) | (numeric > 150)
                    if out_mask.any():
                        out_count = int(out_mask.sum())
                        issues.append(QualityIssue(
                            issue_id=f"sem_age_{col}_{uuid.uuid4().hex[:8]}",
                            issue_type=IssueType.ANOMALY,
                            severity=Severity.MEDIUM,
                            column=col,
                            description=(
                                f"Colonne '{col}' (age) contient "
                                f"{out_count} valeur(s) hors plage [0, 150]."
                            ),
                            details={**base_details, "out_of_range_count": out_count},
                            affected_count=out_count,
                            affected_percentage=round(out_count / total * 100, 2),
                            confidence=confidence,
                            detected_by=AgentType.QUALITY,
                        ))
                except Exception:  # noqa: BLE001
                    pass

            # ── Règle : email/phone/url/postal_code → format via regex ────────
            elif sem_type in ("email", "phone", "url", "postal_code"):
                regex_key = "postal_fr" if sem_type == "postal_code" else sem_type
                pattern = format_re.get(regex_key)
                if pattern is None:
                    continue
                str_series = col_series.astype(str)
                invalid_mask = ~str_series.str.match(pattern, na=False)
                invalid_count = int(invalid_mask.sum())
                if invalid_count == 0:
                    continue
                invalid_pct = invalid_count / total * 100
                # Seuil 5% pour éviter trop de bruit (cohérent avec _detect_format_issues)
                if invalid_pct < 5.0:
                    continue
                issues.append(QualityIssue(
                    issue_id=f"sem_fmt_{col}_{uuid.uuid4().hex[:8]}",
                    issue_type=IssueType.FORMAT_ERROR,
                    severity=Severity.MEDIUM,
                    column=col,
                    description=(
                        f"Colonne '{col}' (type sémantique: {sem_type}) : "
                        f"{invalid_count} valeur(s) au format invalide ({invalid_pct:.1f}%)."
                    ),
                    details={**base_details, "invalid_count": invalid_count},
                    affected_count=invalid_count,
                    affected_percentage=round(invalid_pct, 2),
                    confidence=round(confidence * 0.9, 3),
                    detected_by=AgentType.QUALITY,
                ))

        return issues

    async def _detect_semantic_anomalies_llm(
        self,
        df: pd.DataFrame,
        context: AgentContext,
    ) -> list[QualityIssue]:
        """
        Détection sémantique via Claude (F23 — opt-in ENABLE_LLM_CHECKS=True).

        Boucle sur les colonnes `object`, échantillonne 20 valeurs,
        appelle Claude avec le tool `flag_anomaly`. Max 5 colonnes, timeout 10s,
        fallback silencieux si l'API échoue.
        """
        import asyncio as _asyncio

        from src.core.config import settings

        if not settings.enable_llm_checks:
            return []

        try:
            import anthropic
        except ImportError:
            return []

        object_cols = list(df.select_dtypes(include="object").columns)[:5]
        if not object_cols:
            return []

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        tools: list[dict] = [
            {
                "name": "flag_anomaly",
                "description": (
                    "Signale une anomalie sémantique détectée dans un échantillon de valeurs "
                    "(format incohérent, valeur impossible, pattern suspect, etc.)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string", "description": "Nom de la colonne"},
                        "value": {"type": "string", "description": "Valeur anormale"},
                        "reason": {"type": "string", "description": "Raison de l'anomalie"},
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                        },
                    },
                    "required": ["column", "value", "reason", "severity"],
                },
            }
        ]

        severity_map = {
            "low": Severity.LOW,
            "medium": Severity.MEDIUM,
            "high": Severity.HIGH,
            "critical": Severity.CRITICAL,
        }

        issues: list[QualityIssue] = []

        for col in object_cols:
            non_null = df[col].dropna()
            if non_null.empty:
                continue

            sample_size = min(20, len(non_null))
            sample = non_null.sample(sample_size, random_state=42).astype(str).tolist()

            prompt = (
                f"Analyze these sample values from column '{col}' and identify semantic anomalies "
                f"(inconsistent formats, impossible values, suspicious patterns).\n"
                f"Values: {sample}\n"
                f"Call flag_anomaly for each anomaly you find. "
                f"If values look normal, do not call the tool."
            )

            try:
                response = await _asyncio.wait_for(
                    client.messages.create(
                        model=settings.llm_check_model,
                        max_tokens=512,
                        tools=tools,  # type: ignore[arg-type]
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=10.0,
                )

                for block in response.content:
                    if block.type == "tool_use" and block.name == "flag_anomaly":
                        inp = block.input
                        issues.append(QualityIssue(
                            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                            issue_type=IssueType.ANOMALY,
                            severity=severity_map.get(inp.get("severity", "medium"), Severity.MEDIUM),
                            column=col,
                            row_indices=[],
                            description=(
                                f"[LLM] {inp.get('reason', 'Anomalie sémantique')} "
                                f"(valeur: {inp.get('value')})"
                            ),
                            details={
                                "detection_method": "llm",
                                "model": settings.llm_check_model,
                                "flagged_value": inp.get("value"),
                            },
                            affected_count=1,
                            affected_percentage=round(100.0 / max(len(df), 1), 4),
                            confidence=0.7,
                            detected_by=AgentType.QUALITY,
                        ))

            except Exception:
                # Fallback silencieux — LLM check best-effort
                pass

        return issues
