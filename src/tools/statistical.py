"""
Tools statistiques pour les agents LangChain.

Ces outils permettent aux agents d'analyser les données
de manière statistique. Ils wrappent les fonctionnalités
pandas/numpy dans une interface LangChain.
"""

import json
from typing import Any, Type

import numpy as np
import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class DataProfileInput(BaseModel):
    """Input pour le tool de profiling."""

    columns: list[str] | None = Field(
        default=None,
        description="Colonnes à profiler (None = toutes)"
    )
    include_samples: bool = Field(
        default=True,
        description="Inclure des échantillons de valeurs"
    )


class CorrelationInput(BaseModel):
    """Input pour le tool de corrélation."""

    method: str = Field(
        default="pearson",
        description="Méthode de corrélation: pearson, spearman, ou kendall"
    )
    threshold: float = Field(
        default=0.7,
        description="Seuil pour signaler les corrélations fortes"
    )


class StatisticalProfileTool(BaseTool):
    """
    Tool pour générer un profil statistique d'un DataFrame.

    Calcule les statistiques descriptives pour chaque colonne:
    - Count, nulls, uniques
    - Mean, std, min, max, quartiles (numériques)
    - Mode, fréquences (catégorielles)
    """

    name: str = "statistical_profile"
    description: str = """
    Génère un profil statistique complet du dataset.
    Retourne les statistiques descriptives pour chaque colonne:
    count, nulls, uniques, mean, std, min, max, quartiles.
    Utilisez cet outil pour comprendre la structure et la distribution des données.
    """
    args_schema: Type[BaseModel] = DataProfileInput

    # DataFrame injecté par l'agent
    dataframe: pd.DataFrame | None = None

    def _run(
        self,
        columns: list[str] | None = None,
        include_samples: bool = True
    ) -> str:
        """Exécute le profiling statistique."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        df = self.dataframe

        # Sélection des colonnes
        if columns:
            df = df[columns]

        profiles = []

        for col in df.columns:
            series = df[col]
            profile = {
                "name": col,
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "null_count": int(series.isna().sum()),
                "null_percentage": round(100 * series.isna().mean(), 2),
                "unique_count": int(series.nunique()),
                "unique_percentage": round(100 * series.nunique() / len(series), 2),
            }

            # Stats numériques
            if pd.api.types.is_numeric_dtype(series):
                profile["inferred_type"] = "numeric"
                desc = series.describe()
                profile.update({
                    "mean": round(float(desc["mean"]), 4) if not pd.isna(desc["mean"]) else None,
                    "std": round(float(desc["std"]), 4) if not pd.isna(desc["std"]) else None,
                    "min": float(desc["min"]) if not pd.isna(desc["min"]) else None,
                    "max": float(desc["max"]) if not pd.isna(desc["max"]) else None,
                    "q25": float(desc["25%"]) if "25%" in desc else None,
                    "q50": float(desc["50%"]) if "50%" in desc else None,
                    "q75": float(desc["75%"]) if "75%" in desc else None,
                })
            else:
                profile["inferred_type"] = "categorical"
                if series.nunique() > 0:
                    mode_val = series.mode()
                    profile["mode"] = str(mode_val.iloc[0]) if len(mode_val) > 0 else None
                    value_counts = series.value_counts()
                    profile["top_values"] = {
                        str(k): int(v) for k, v in value_counts.head(5).items()
                    }

            # Échantillons
            if include_samples:
                sample_values = series.dropna().head(5).tolist()
                profile["sample_values"] = [
                    str(v) if not isinstance(v, (int, float)) else v
                    for v in sample_values
                ]

            profiles.append(profile)

        result = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "total_null_count": int(df.isna().sum().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "columns": profiles
        }

        return json.dumps(result, indent=2, default=str)


class CorrelationAnalysisTool(BaseTool):
    """
    Tool pour analyser les corrélations entre colonnes numériques.

    Identifie les paires de colonnes fortement corrélées
    qui pourraient indiquer de la redondance ou des relations importantes.
    """

    name: str = "correlation_analysis"
    description: str = """
    Analyse les corrélations entre colonnes numériques.
    Retourne les paires de colonnes avec corrélation forte (au-dessus du seuil).
    Utilisez cet outil pour identifier la redondance ou les relations entre variables.
    """
    args_schema: Type[BaseModel] = CorrelationInput

    dataframe: pd.DataFrame | None = None

    def _run(
        self,
        method: str = "pearson",
        threshold: float = 0.7
    ) -> str:
        """Calcule et analyse les corrélations."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        df = self.dataframe

        # Sélectionner uniquement les colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return json.dumps({
                "message": "Moins de 2 colonnes numériques, corrélation non applicable",
                "numeric_columns": list(numeric_df.columns)
            })

        # Calculer la matrice de corrélation
        corr_matrix = numeric_df.corr(method=method)

        # Trouver les paires avec forte corrélation
        strong_correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Éviter les doublons
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            "column_1": col1,
                            "column_2": col2,
                            "correlation": round(float(corr_value), 4),
                            "strength": "positive" if corr_value > 0 else "negative",
                            "interpretation": self._interpret_correlation(corr_value)
                        })

        # Trier par force de corrélation
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        result = {
            "method": method,
            "threshold": threshold,
            "numeric_columns_count": len(numeric_df.columns),
            "strong_correlations_count": len(strong_correlations),
            "strong_correlations": strong_correlations,
        }

        return json.dumps(result, indent=2)

    @staticmethod
    def _interpret_correlation(corr: float) -> str:
        """Interprète la force de la corrélation."""
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            return "Très forte corrélation - possible redondance"
        elif abs_corr >= 0.7:
            return "Forte corrélation - relation significative"
        elif abs_corr >= 0.5:
            return "Corrélation modérée"
        else:
            return "Corrélation faible"


class DistributionAnalysisInput(BaseModel):
    """Input pour l'analyse de distribution."""

    column: str = Field(
        ...,
        description="Nom de la colonne à analyser"
    )
    n_bins: int = Field(
        default=20,
        description="Nombre de bins pour l'histogramme"
    )


class DistributionAnalysisTool(BaseTool):
    """
    Tool pour analyser la distribution d'une colonne.

    Calcule des métriques de distribution comme la skewness
    et la kurtosis, et identifie le type de distribution.
    """

    name: str = "distribution_analysis"
    description: str = """
    Analyse la distribution d'une colonne spécifique.
    Calcule skewness, kurtosis, et identifie le type de distribution.
    Utilisez cet outil pour comprendre la forme des données.
    """
    args_schema: Type[BaseModel] = DistributionAnalysisInput

    dataframe: pd.DataFrame | None = None

    def _run(self, column: str, n_bins: int = 20) -> str:
        """Analyse la distribution d'une colonne."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        if column not in self.dataframe.columns:
            return json.dumps({"error": f"Colonne '{column}' non trouvée"})

        series = self.dataframe[column].dropna()

        if not pd.api.types.is_numeric_dtype(series):
            # Distribution catégorielle
            value_counts = series.value_counts()
            return json.dumps({
                "column": column,
                "type": "categorical",
                "unique_values": int(series.nunique()),
                "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "distribution": {
                    str(k): int(v) for k, v in value_counts.head(10).items()
                }
            }, indent=2)

        # Distribution numérique
        from scipy import stats as scipy_stats

        skewness = float(scipy_stats.skew(series))
        kurtosis = float(scipy_stats.kurtosis(series))

        # Test de normalité (Shapiro-Wilk pour petits échantillons)
        if len(series) <= 5000:
            _, normality_pvalue = scipy_stats.shapiro(series.sample(min(len(series), 5000)))
        else:
            _, normality_pvalue = scipy_stats.normaltest(series.sample(5000))

        # Histogramme simplifié
        hist, bin_edges = np.histogram(series, bins=n_bins)
        histogram = [
            {
                "bin_start": round(float(bin_edges[i]), 4),
                "bin_end": round(float(bin_edges[i + 1]), 4),
                "count": int(hist[i])
            }
            for i in range(len(hist))
        ]

        # Interprétation
        distribution_type = self._identify_distribution_type(skewness, kurtosis, normality_pvalue)

        result = {
            "column": column,
            "type": "numeric",
            "count": len(series),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "normality_pvalue": round(float(normality_pvalue), 6),
            "is_normal": normality_pvalue > 0.05,
            "distribution_type": distribution_type,
            "histogram": histogram
        }

        return json.dumps(result, indent=2)

    @staticmethod
    def _identify_distribution_type(
        skewness: float,
        kurtosis: float,
        normality_pvalue: float
    ) -> str:
        """Identifie le type de distribution."""
        if normality_pvalue > 0.05:
            return "approximately_normal"
        elif abs(skewness) > 1:
            if skewness > 0:
                return "right_skewed"
            else:
                return "left_skewed"
        elif kurtosis > 1:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "non_normal"


# Factory pour créer les tools avec un DataFrame
def create_statistical_tools(df: pd.DataFrame) -> list[BaseTool]:
    """
    Crée les outils statistiques injectés avec un DataFrame.

    Args:
        df: DataFrame à analyser

    Returns:
        Liste d'outils LangChain configurés
    """
    profile_tool = StatisticalProfileTool()
    profile_tool.dataframe = df

    correlation_tool = CorrelationAnalysisTool()
    correlation_tool.dataframe = df

    distribution_tool = DistributionAnalysisTool()
    distribution_tool.dataframe = df

    return [profile_tool, correlation_tool, distribution_tool]
