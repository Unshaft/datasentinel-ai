"""
Détecteur de drift (dérive) de données.

Ce module détecte les changements de distribution entre un dataset
de référence et un dataset actuel. Le drift indique que les données
ont changé de manière significative, ce qui peut impacter:
- La performance des modèles ML entraînés sur les anciennes données
- La validité des règles métier
- La cohérence des analyses temporelles

Méthodes implémentées:
- KS-test (Kolmogorov-Smirnov): pour variables continues
- Chi-squared test: pour variables catégorielles
- PSI (Population Stability Index): métrique industrielle standard
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.core.exceptions import DriftDetectionError, InsufficientDataError


class DriftSeverity(str, Enum):
    """Niveau de sévérité du drift détecté."""

    NONE = "none"           # Pas de drift significatif
    LOW = "low"             # Drift mineur, à surveiller
    MEDIUM = "medium"       # Drift modéré, investigation recommandée
    HIGH = "high"           # Drift important, action requise
    CRITICAL = "critical"   # Drift critique, données potentiellement inutilisables


@dataclass
class DriftResult:
    """Résultat de détection de drift pour une colonne."""

    column: str
    has_drift: bool
    severity: DriftSeverity
    drift_score: float          # Score normalisé 0-1
    p_value: float | None       # P-value du test statistique
    test_used: str              # Nom du test utilisé
    reference_stats: dict[str, float]
    current_stats: dict[str, float]
    interpretation: str         # Explication en langage naturel

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "column": self.column,
            "has_drift": self.has_drift,
            "severity": self.severity.value,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "test_used": self.test_used,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
            "interpretation": self.interpretation,
        }


class DriftDetector:
    """
    Détecteur de drift entre datasets de référence et actuel.

    Le drift est détecté en comparant les distributions statistiques
    des colonnes entre deux datasets. Plusieurs métriques sont
    calculées pour une évaluation robuste.

    Attributes:
        p_value_threshold: Seuil pour considérer un drift significatif
        psi_threshold: Seuil PSI pour drift (standard: 0.25)
    """

    # Seuils PSI standards (industrie bancaire/finance)
    PSI_LOW = 0.1       # Changement mineur
    PSI_MEDIUM = 0.25   # Changement modéré
    PSI_HIGH = 0.5      # Changement majeur

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.25,
        min_samples: int = 30
    ) -> None:
        """
        Initialise le détecteur de drift.

        Args:
            p_value_threshold: Seuil p-value (défaut: 0.05 = 95% confiance)
            psi_threshold: Seuil PSI pour drift significatif
            min_samples: Minimum d'échantillons requis par dataset
        """
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples

        self._reference_data: pd.DataFrame | None = None
        self._reference_stats: dict[str, dict] = {}

    def set_reference(self, df: pd.DataFrame) -> "DriftDetector":
        """
        Définit le dataset de référence.

        Args:
            df: DataFrame de référence

        Returns:
            Self pour chaînage

        Raises:
            InsufficientDataError: Si pas assez de données
        """
        if len(df) < self.min_samples:
            raise InsufficientDataError(
                model_name="DriftDetector",
                required=self.min_samples,
                actual=len(df)
            )

        self._reference_data = df.copy()
        self._reference_stats = self._compute_stats(df)
        return self

    def detect(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None
    ) -> list[DriftResult]:
        """
        Détecte le drift entre la référence et le dataset actuel.

        Args:
            df: Dataset actuel à comparer
            columns: Colonnes à vérifier (None = intersection)

        Returns:
            Liste de résultats de drift

        Raises:
            DriftDetectionError: Si pas de référence définie
        """
        if self._reference_data is None:
            raise DriftDetectionError(
                column="*",
                reason="Aucune référence définie. Appelez set_reference() d'abord."
            )

        if len(df) < self.min_samples:
            raise InsufficientDataError(
                model_name="DriftDetector",
                required=self.min_samples,
                actual=len(df)
            )

        # Déterminer les colonnes à analyser
        if columns is None:
            columns = list(
                set(self._reference_data.columns) & set(df.columns)
            )

        results = []
        for column in columns:
            try:
                result = self._detect_column_drift(df, column)
                results.append(result)
            except Exception as e:
                # Log l'erreur mais continue avec les autres colonnes
                results.append(DriftResult(
                    column=column,
                    has_drift=False,
                    severity=DriftSeverity.NONE,
                    drift_score=0.0,
                    p_value=None,
                    test_used="error",
                    reference_stats={},
                    current_stats={},
                    interpretation=f"Erreur d'analyse: {str(e)}"
                ))

        return results

    def _detect_column_drift(
        self,
        df: pd.DataFrame,
        column: str
    ) -> DriftResult:
        """Détecte le drift pour une colonne spécifique."""
        ref_series = self._reference_data[column].dropna()
        cur_series = df[column].dropna()

        # Déterminer le type de colonne
        is_numeric = pd.api.types.is_numeric_dtype(ref_series)

        if is_numeric:
            return self._detect_numeric_drift(column, ref_series, cur_series)
        else:
            return self._detect_categorical_drift(column, ref_series, cur_series)

    def _detect_numeric_drift(
        self,
        column: str,
        ref_series: pd.Series,
        cur_series: pd.Series
    ) -> DriftResult:
        """Détecte le drift pour une colonne numérique avec KS-test et PSI."""
        # Test de Kolmogorov-Smirnov
        ks_stat, p_value = stats.ks_2samp(ref_series, cur_series)

        # Calcul du PSI
        psi = self._calculate_psi(ref_series, cur_series)

        # Statistiques descriptives
        ref_stats = {
            "mean": float(ref_series.mean()),
            "std": float(ref_series.std()),
            "median": float(ref_series.median()),
            "min": float(ref_series.min()),
            "max": float(ref_series.max()),
        }
        cur_stats = {
            "mean": float(cur_series.mean()),
            "std": float(cur_series.std()),
            "median": float(cur_series.median()),
            "min": float(cur_series.min()),
            "max": float(cur_series.max()),
        }

        # Déterminer la sévérité
        severity = self._determine_severity_numeric(p_value, psi)
        has_drift = severity != DriftSeverity.NONE

        # Interprétation
        interpretation = self._interpret_numeric_drift(
            column, ref_stats, cur_stats, p_value, psi, severity
        )

        return DriftResult(
            column=column,
            has_drift=has_drift,
            severity=severity,
            drift_score=min(1.0, psi / self.PSI_HIGH),
            p_value=float(p_value),
            test_used="KS-test + PSI",
            reference_stats=ref_stats,
            current_stats=cur_stats,
            interpretation=interpretation
        )

    def _detect_categorical_drift(
        self,
        column: str,
        ref_series: pd.Series,
        cur_series: pd.Series
    ) -> DriftResult:
        """Détecte le drift pour une colonne catégorielle avec Chi-squared."""
        # Obtenir toutes les catégories
        all_categories = set(ref_series.unique()) | set(cur_series.unique())

        # Fréquences observées
        ref_counts = ref_series.value_counts()
        cur_counts = cur_series.value_counts()

        # Aligner sur toutes les catégories
        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
        cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]

        # Test Chi-squared
        # Ajouter un petit epsilon pour éviter division par zéro
        ref_freq = np.array(ref_freq) + 1e-10
        cur_freq = np.array(cur_freq) + 1e-10

        # Normaliser
        ref_freq = ref_freq / ref_freq.sum()
        cur_freq = cur_freq / cur_freq.sum()

        # Chi-squared test
        chi2, p_value = stats.chisquare(cur_freq, f_exp=ref_freq)

        # Statistiques
        ref_stats = {
            "n_categories": len(ref_series.unique()),
            "top_category": str(ref_series.mode().iloc[0]) if len(ref_series.mode()) > 0 else None,
            "top_frequency": float(ref_counts.iloc[0] / len(ref_series)) if len(ref_counts) > 0 else 0,
        }
        cur_stats = {
            "n_categories": len(cur_series.unique()),
            "top_category": str(cur_series.mode().iloc[0]) if len(cur_series.mode()) > 0 else None,
            "top_frequency": float(cur_counts.iloc[0] / len(cur_series)) if len(cur_counts) > 0 else 0,
        }

        # Sévérité basée sur p-value
        severity = self._determine_severity_categorical(p_value)
        has_drift = severity != DriftSeverity.NONE

        interpretation = self._interpret_categorical_drift(
            column, ref_stats, cur_stats, p_value, severity
        )

        return DriftResult(
            column=column,
            has_drift=has_drift,
            severity=severity,
            drift_score=min(1.0, chi2 / 100),  # Normalisation approximative
            p_value=float(p_value),
            test_used="Chi-squared",
            reference_stats=ref_stats,
            current_stats=cur_stats,
            interpretation=interpretation
        )

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10
    ) -> float:
        """
        Calcule le Population Stability Index (PSI).

        PSI mesure le changement de distribution entre deux populations.
        Formule: PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)

        Seuils standards:
        - PSI < 0.1: Pas de changement significatif
        - 0.1 <= PSI < 0.25: Changement modéré
        - PSI >= 0.25: Changement significatif

        Args:
            reference: Série de référence
            current: Série actuelle
            n_bins: Nombre de buckets

        Returns:
            Score PSI
        """
        # Créer des bins basés sur la référence
        try:
            _, bin_edges = pd.qcut(reference, q=n_bins, retbins=True, duplicates="drop")
        except ValueError:
            # Pas assez de valeurs uniques pour les quantiles
            _, bin_edges = pd.cut(reference, bins=n_bins, retbins=True)

        # Compter dans chaque bin
        ref_counts = pd.cut(reference, bins=bin_edges, include_lowest=True).value_counts()
        cur_counts = pd.cut(current, bins=bin_edges, include_lowest=True).value_counts()

        # Convertir en proportions (avec correction pour éviter log(0))
        epsilon = 1e-10
        ref_pct = (ref_counts / len(reference)).values + epsilon
        cur_pct = (cur_counts / len(current)).values + epsilon

        # Calcul PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def _determine_severity_numeric(
        self,
        p_value: float,
        psi: float
    ) -> DriftSeverity:
        """Détermine la sévérité pour une colonne numérique."""
        if p_value >= self.p_value_threshold and psi < self.PSI_LOW:
            return DriftSeverity.NONE
        elif psi < self.PSI_LOW:
            return DriftSeverity.LOW
        elif psi < self.PSI_MEDIUM:
            return DriftSeverity.MEDIUM
        elif psi < self.PSI_HIGH:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _determine_severity_categorical(self, p_value: float) -> DriftSeverity:
        """Détermine la sévérité pour une colonne catégorielle."""
        if p_value >= self.p_value_threshold:
            return DriftSeverity.NONE
        elif p_value >= 0.01:
            return DriftSeverity.LOW
        elif p_value >= 0.001:
            return DriftSeverity.MEDIUM
        elif p_value >= 0.0001:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def _interpret_numeric_drift(
        self,
        column: str,
        ref_stats: dict,
        cur_stats: dict,
        p_value: float,
        psi: float,
        severity: DriftSeverity
    ) -> str:
        """Génère une interprétation en langage naturel."""
        if severity == DriftSeverity.NONE:
            return f"Aucun drift significatif détecté pour '{column}'."

        mean_change = ((cur_stats["mean"] - ref_stats["mean"]) / ref_stats["mean"]) * 100
        direction = "augmenté" if mean_change > 0 else "diminué"

        return (
            f"Drift {severity.value} détecté pour '{column}': "
            f"la moyenne a {direction} de {abs(mean_change):.1f}% "
            f"(référence: {ref_stats['mean']:.2f}, actuel: {cur_stats['mean']:.2f}). "
            f"PSI={psi:.3f}, p-value={p_value:.4f}."
        )

    def _interpret_categorical_drift(
        self,
        column: str,
        ref_stats: dict,
        cur_stats: dict,
        p_value: float,
        severity: DriftSeverity
    ) -> str:
        """Génère une interprétation pour colonnes catégorielles."""
        if severity == DriftSeverity.NONE:
            return f"Aucun drift significatif détecté pour '{column}'."

        cat_diff = cur_stats["n_categories"] - ref_stats["n_categories"]
        cat_change = ""
        if cat_diff != 0:
            cat_change = f" Le nombre de catégories a changé ({cat_diff:+d})."

        return (
            f"Drift {severity.value} détecté pour '{column}': "
            f"la distribution des catégories a changé significativement "
            f"(p-value={p_value:.4f}).{cat_change}"
        )

    def _compute_stats(self, df: pd.DataFrame) -> dict[str, dict]:
        """Calcule les statistiques de référence pour chaque colonne."""
        stats_dict = {}
        for column in df.columns:
            series = df[column].dropna()
            if pd.api.types.is_numeric_dtype(series):
                stats_dict[column] = {
                    "type": "numeric",
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "median": float(series.median()),
                }
            else:
                stats_dict[column] = {
                    "type": "categorical",
                    "n_unique": int(series.nunique()),
                    "mode": str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                }
        return stats_dict

    def get_drift_summary(
        self,
        results: list[DriftResult]
    ) -> dict[str, Any]:
        """
        Génère un résumé global du drift.

        Args:
            results: Liste des résultats de drift

        Returns:
            Dictionnaire avec statistiques globales
        """
        if not results:
            return {
                "total_columns": 0,
                "columns_with_drift": 0,
                "drift_percentage": 0.0,
                "severity_distribution": {},
                "most_drifted_column": None,
            }

        columns_with_drift = [r for r in results if r.has_drift]
        severity_counts = {}
        for r in results:
            severity_counts[r.severity.value] = severity_counts.get(r.severity.value, 0) + 1

        most_drifted = max(results, key=lambda r: r.drift_score) if results else None

        return {
            "total_columns": len(results),
            "columns_with_drift": len(columns_with_drift),
            "drift_percentage": round(100 * len(columns_with_drift) / len(results), 1),
            "severity_distribution": severity_counts,
            "most_drifted_column": most_drifted.column if most_drifted else None,
            "max_drift_score": most_drifted.drift_score if most_drifted else 0.0,
            "columns_by_severity": {
                r.column: r.severity.value for r in results if r.has_drift
            }
        }

    @property
    def has_reference(self) -> bool:
        """Vérifie si une référence est définie."""
        return self._reference_data is not None
