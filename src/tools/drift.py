"""
Tool de détection de drift pour les agents LangChain.

Wrapper autour du composant ML DriftDetector pour
l'intégration dans le système d'agents.
"""

import json
from typing import Any, Type

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.ml.drift_detector import DriftDetector, DriftResult


class DriftDetectionInput(BaseModel):
    """Input pour la détection de drift."""

    columns: list[str] | None = Field(
        default=None,
        description="Colonnes à analyser (None = intersection avec référence)"
    )
    p_value_threshold: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="Seuil p-value pour significativité (défaut: 0.05)"
    )


class SetReferenceInput(BaseModel):
    """Input pour définir la référence de drift."""

    description: str = Field(
        default="reference dataset",
        description="Description du dataset de référence"
    )


class DriftDetectionTool(BaseTool):
    """
    Tool pour détecter le drift entre datasets.

    Compare la distribution actuelle avec une distribution
    de référence pour identifier les changements significatifs.
    """

    name: str = "detect_drift"
    description: str = """
    Détecte les changements de distribution (drift) par rapport à une référence.
    Utilise KS-test pour les numériques et Chi-squared pour les catégorielles.
    Retourne les colonnes avec drift significatif et la sévérité.
    IMPORTANT: Une référence doit être définie avec set_drift_reference avant utilisation.
    """
    args_schema: Type[BaseModel] = DriftDetectionInput

    dataframe: pd.DataFrame | None = None
    _detector: DriftDetector | None = None

    def _run(
        self,
        columns: list[str] | None = None,
        p_value_threshold: float = 0.05
    ) -> str:
        """Détecte le drift."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        if self._detector is None or not self._detector.has_reference:
            return json.dumps({
                "error": "Aucune référence définie. Utilisez set_drift_reference d'abord."
            })

        try:
            # Mettre à jour le seuil si différent
            self._detector.p_value_threshold = p_value_threshold

            results = self._detector.detect(self.dataframe, columns)
            summary = self._detector.get_drift_summary(results)

            output = {
                "status": "success",
                "p_value_threshold": p_value_threshold,
                "summary": summary,
                "columns_with_drift": [
                    {
                        "column": r.column,
                        "severity": r.severity.value,
                        "drift_score": round(r.drift_score, 4),
                        "p_value": round(r.p_value, 6) if r.p_value else None,
                        "test_used": r.test_used,
                        "interpretation": r.interpretation,
                        "reference_stats": {
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in r.reference_stats.items()
                        },
                        "current_stats": {
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in r.current_stats.items()
                        }
                    }
                    for r in results if r.has_drift
                ],
                "columns_without_drift": [
                    r.column for r in results if not r.has_drift
                ]
            }

            return json.dumps(output, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })

    def set_reference(self, reference_df: pd.DataFrame) -> None:
        """Définit le DataFrame de référence."""
        self._detector = DriftDetector()
        self._detector.set_reference(reference_df)


class SetDriftReferenceTool(BaseTool):
    """
    Tool pour définir le dataset de référence pour la détection de drift.

    La référence représente l'état "normal" des données contre
    lequel les futures données seront comparées.
    """

    name: str = "set_drift_reference"
    description: str = """
    Définit le dataset actuel comme référence pour la détection de drift.
    Après cette action, detect_drift comparera les nouveaux datasets à cette référence.
    Utilisez cet outil une fois pour établir la baseline.
    """
    args_schema: Type[BaseModel] = SetReferenceInput

    dataframe: pd.DataFrame | None = None
    drift_detection_tool: DriftDetectionTool | None = None

    def _run(self, description: str = "reference dataset") -> str:
        """Définit la référence."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        if self.drift_detection_tool is None:
            return json.dumps({"error": "drift_detection_tool non configuré"})

        try:
            self.drift_detection_tool.set_reference(self.dataframe)

            # Statistiques de la référence
            df = self.dataframe
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

            result = {
                "status": "success",
                "message": f"Référence '{description}' définie avec succès",
                "reference_info": {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                },
                "next_step": "Utilisez detect_drift sur un nouveau dataset pour comparer"
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })


class DriftComparisonInput(BaseModel):
    """Input pour comparer deux colonnes spécifiques."""

    column: str = Field(
        ...,
        description="Nom de la colonne à comparer"
    )


class DriftColumnComparisonTool(BaseTool):
    """
    Tool pour analyser le drift en détail sur une colonne spécifique.

    Fournit une analyse approfondie des changements de distribution
    incluant des visualisations textuelles.
    """

    name: str = "analyze_column_drift"
    description: str = """
    Analyse en détail le drift d'une colonne spécifique.
    Fournit une comparaison détaillée entre référence et actuel.
    Utilisez après detect_drift pour investiguer une colonne particulière.
    """
    args_schema: Type[BaseModel] = DriftComparisonInput

    dataframe: pd.DataFrame | None = None
    reference_dataframe: pd.DataFrame | None = None

    def _run(self, column: str) -> str:
        """Analyse détaillée du drift d'une colonne."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame actuel chargé"})

        if self.reference_dataframe is None:
            return json.dumps({"error": "Aucun DataFrame de référence défini"})

        if column not in self.dataframe.columns:
            return json.dumps({"error": f"Colonne '{column}' non trouvée dans le dataset actuel"})

        if column not in self.reference_dataframe.columns:
            return json.dumps({"error": f"Colonne '{column}' non trouvée dans la référence"})

        ref_series = self.reference_dataframe[column].dropna()
        cur_series = self.dataframe[column].dropna()

        is_numeric = pd.api.types.is_numeric_dtype(ref_series)

        if is_numeric:
            result = self._analyze_numeric_drift(column, ref_series, cur_series)
        else:
            result = self._analyze_categorical_drift(column, ref_series, cur_series)

        return json.dumps(result, indent=2)

    def _analyze_numeric_drift(
        self,
        column: str,
        ref_series: pd.Series,
        cur_series: pd.Series
    ) -> dict[str, Any]:
        """Analyse détaillée pour colonne numérique."""
        from scipy import stats

        # Statistiques
        ref_desc = ref_series.describe()
        cur_desc = cur_series.describe()

        # Tests statistiques
        ks_stat, ks_pvalue = stats.ks_2samp(ref_series, cur_series)

        # Changements en pourcentage
        mean_change = ((cur_desc["mean"] - ref_desc["mean"]) / ref_desc["mean"]) * 100 if ref_desc["mean"] != 0 else 0
        std_change = ((cur_desc["std"] - ref_desc["std"]) / ref_desc["std"]) * 100 if ref_desc["std"] != 0 else 0

        # Distribution par quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        ref_quantiles = ref_series.quantile(quantiles).to_dict()
        cur_quantiles = cur_series.quantile(quantiles).to_dict()

        return {
            "column": column,
            "type": "numeric",
            "sample_sizes": {
                "reference": len(ref_series),
                "current": len(cur_series)
            },
            "statistical_test": {
                "test": "Kolmogorov-Smirnov",
                "statistic": round(ks_stat, 6),
                "p_value": round(ks_pvalue, 6),
                "significant": ks_pvalue < 0.05
            },
            "changes": {
                "mean_change_percent": round(mean_change, 2),
                "std_change_percent": round(std_change, 2),
            },
            "reference_stats": {
                "mean": round(float(ref_desc["mean"]), 4),
                "std": round(float(ref_desc["std"]), 4),
                "min": round(float(ref_desc["min"]), 4),
                "max": round(float(ref_desc["max"]), 4),
            },
            "current_stats": {
                "mean": round(float(cur_desc["mean"]), 4),
                "std": round(float(cur_desc["std"]), 4),
                "min": round(float(cur_desc["min"]), 4),
                "max": round(float(cur_desc["max"]), 4),
            },
            "quantile_comparison": {
                f"q{int(q*100)}": {
                    "reference": round(ref_quantiles[q], 4),
                    "current": round(cur_quantiles[q], 4),
                    "change": round(cur_quantiles[q] - ref_quantiles[q], 4)
                }
                for q in quantiles
            }
        }

    def _analyze_categorical_drift(
        self,
        column: str,
        ref_series: pd.Series,
        cur_series: pd.Series
    ) -> dict[str, Any]:
        """Analyse détaillée pour colonne catégorielle."""
        ref_counts = ref_series.value_counts(normalize=True)
        cur_counts = cur_series.value_counts(normalize=True)

        all_categories = set(ref_counts.index) | set(cur_counts.index)
        new_categories = set(cur_counts.index) - set(ref_counts.index)
        removed_categories = set(ref_counts.index) - set(cur_counts.index)

        # Changements de fréquence
        frequency_changes = {}
        for cat in all_categories:
            ref_freq = ref_counts.get(cat, 0)
            cur_freq = cur_counts.get(cat, 0)
            change = cur_freq - ref_freq
            if abs(change) > 0.01:  # Seuil de 1%
                frequency_changes[str(cat)] = {
                    "reference_freq": round(float(ref_freq), 4),
                    "current_freq": round(float(cur_freq), 4),
                    "change": round(float(change), 4)
                }

        return {
            "column": column,
            "type": "categorical",
            "sample_sizes": {
                "reference": len(ref_series),
                "current": len(cur_series)
            },
            "category_counts": {
                "reference": len(ref_counts),
                "current": len(cur_counts),
                "new_categories": list(new_categories)[:10],
                "removed_categories": list(removed_categories)[:10]
            },
            "significant_frequency_changes": dict(
                sorted(
                    frequency_changes.items(),
                    key=lambda x: abs(x[1]["change"]),
                    reverse=True
                )[:10]
            ),
            "top_categories_comparison": {
                "reference_top5": {
                    str(k): round(float(v), 4)
                    for k, v in ref_counts.head(5).items()
                },
                "current_top5": {
                    str(k): round(float(v), 4)
                    for k, v in cur_counts.head(5).items()
                }
            }
        }


# Factory pour créer les tools
def create_drift_tools(df: pd.DataFrame) -> list[BaseTool]:
    """
    Crée les outils de détection de drift avec un DataFrame.

    Args:
        df: DataFrame à analyser

    Returns:
        Liste d'outils configurés
    """
    detection_tool = DriftDetectionTool()
    detection_tool.dataframe = df

    reference_tool = SetDriftReferenceTool()
    reference_tool.dataframe = df
    reference_tool.drift_detection_tool = detection_tool

    comparison_tool = DriftColumnComparisonTool()
    comparison_tool.dataframe = df

    return [detection_tool, reference_tool, comparison_tool]
