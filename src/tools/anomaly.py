"""
Tool de détection d'anomalies pour les agents LangChain.

Wrapper autour du composant ML AnomalyDetector pour
l'intégration dans le système d'agents.
"""

import json
from typing import Any, Type

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.ml.anomaly_detector import AnomalyDetector, AnomalyResult


class AnomalyDetectionInput(BaseModel):
    """Input pour la détection d'anomalies."""

    columns: list[str] | None = Field(
        default=None,
        description="Colonnes à analyser (None = toutes les numériques)"
    )
    contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Proportion attendue d'anomalies (0.01 à 0.5)"
    )


class AnomalyInvestigationInput(BaseModel):
    """Input pour l'investigation d'une anomalie."""

    column: str = Field(
        ...,
        description="Colonne contenant l'anomalie"
    )
    row_index: int = Field(
        ...,
        description="Index de la ligne à investiguer"
    )


class AnomalyDetectionTool(BaseTool):
    """
    Tool pour détecter les anomalies statistiques.

    Utilise Isolation Forest pour identifier les valeurs
    qui s'écartent significativement de la distribution normale.
    """

    name: str = "detect_anomalies"
    description: str = """
    Détecte les anomalies statistiques dans les colonnes numériques.
    Utilise l'algorithme Isolation Forest pour identifier les valeurs aberrantes.
    Retourne les indices et valeurs des anomalies détectées.
    Utilisez cet outil quand vous suspectez des valeurs incorrectes ou inhabituelles.
    """
    args_schema: Type[BaseModel] = AnomalyDetectionInput

    dataframe: pd.DataFrame | None = None
    _last_results: list[AnomalyResult] | None = None

    def _run(
        self,
        columns: list[str] | None = None,
        contamination: float = 0.1
    ) -> str:
        """Exécute la détection d'anomalies."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        df = self.dataframe

        try:
            detector = AnomalyDetector(contamination=contamination)
            results = detector.fit_detect(df, columns)
            self._last_results = results

            # Formater les résultats
            output = {
                "status": "success",
                "contamination_rate": contamination,
                "columns_analyzed": detector.fitted_columns,
                "summary": detector.get_anomaly_summary(results),
                "details": []
            }

            for result in results:
                detail = {
                    "column": result.column,
                    "anomaly_count": result.anomaly_count,
                    "anomaly_percentage": round(result.anomaly_percentage, 2),
                    "anomaly_indices": result.anomaly_indices[:20],  # Limiter
                    "anomaly_values": [
                        round(v, 4) if isinstance(v, float) else v
                        for v in result.anomaly_values[:20]
                    ],
                    "most_extreme_score": (
                        round(min(result.anomaly_scores), 4)
                        if result.anomaly_scores else None
                    )
                }

                # Ajouter contexte
                if result.anomaly_count > 0:
                    col_data = df[result.column].dropna()
                    detail["column_context"] = {
                        "mean": round(float(col_data.mean()), 4),
                        "std": round(float(col_data.std()), 4),
                        "min_normal": round(float(col_data.quantile(0.01)), 4),
                        "max_normal": round(float(col_data.quantile(0.99)), 4),
                    }

                output["details"].append(detail)

            return json.dumps(output, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e)
            })


class AnomalyInvestigationTool(BaseTool):
    """
    Tool pour investiguer une anomalie spécifique.

    Fournit le contexte détaillé autour d'une valeur anormale
    pour aider à comprendre pourquoi elle est flaggée.
    """

    name: str = "investigate_anomaly"
    description: str = """
    Investigue une anomalie spécifique en détail.
    Fournit le contexte de la valeur par rapport à la distribution.
    Utilisez cet outil après detect_anomalies pour comprendre une anomalie.
    """
    args_schema: Type[BaseModel] = AnomalyInvestigationInput

    dataframe: pd.DataFrame | None = None

    def _run(self, column: str, row_index: int) -> str:
        """Investigue une anomalie."""
        if self.dataframe is None:
            return json.dumps({"error": "Aucun DataFrame chargé"})

        df = self.dataframe

        if column not in df.columns:
            return json.dumps({"error": f"Colonne '{column}' non trouvée"})

        if row_index not in df.index:
            return json.dumps({"error": f"Index {row_index} non trouvé"})

        # Valeur à investiguer
        value = df.loc[row_index, column]
        series = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(series):
            return json.dumps({
                "error": "L'investigation d'anomalie est uniquement pour les colonnes numériques"
            })

        # Statistiques de référence
        mean = float(series.mean())
        std = float(series.std())
        median = float(series.median())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        # Calculs de contexte
        z_score = (float(value) - mean) / std if std > 0 else 0
        percentile = float((series < value).mean() * 100)

        # Déterminer pourquoi c'est une anomalie
        reasons = []
        if abs(z_score) > 3:
            reasons.append(f"Z-score élevé: {z_score:.2f} (>3 écarts-types)")
        if value < q1 - 1.5 * iqr:
            reasons.append("En dessous de la borne inférieure (Q1 - 1.5*IQR)")
        if value > q3 + 1.5 * iqr:
            reasons.append("Au-dessus de la borne supérieure (Q3 + 1.5*IQR)")
        if percentile < 1:
            reasons.append(f"Dans le 1er percentile ({percentile:.2f}%)")
        if percentile > 99:
            reasons.append(f"Dans le 99ème percentile ({percentile:.2f}%)")

        # Valeurs voisines pour contexte
        sorted_values = series.sort_values()
        value_rank = (sorted_values < value).sum()
        nearby_indices = range(max(0, value_rank - 2), min(len(sorted_values), value_rank + 3))
        nearby_values = sorted_values.iloc[list(nearby_indices)].tolist()

        result = {
            "row_index": row_index,
            "column": column,
            "value": float(value) if pd.notna(value) else None,
            "context": {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "median": round(median, 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            },
            "analysis": {
                "z_score": round(z_score, 4),
                "percentile": round(percentile, 2),
                "is_outlier_iqr": value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr,
                "distance_from_mean_in_std": round(abs(z_score), 2),
            },
            "reasons_for_anomaly": reasons if reasons else ["Détecté par Isolation Forest mais pas d'indicateur classique"],
            "nearby_values": [round(v, 4) for v in nearby_values],
            "recommendation": self._get_recommendation(z_score, percentile, reasons)
        }

        return json.dumps(result, indent=2)

    @staticmethod
    def _get_recommendation(
        z_score: float,
        percentile: float,
        reasons: list[str]
    ) -> str:
        """Génère une recommandation basée sur l'analyse."""
        if abs(z_score) > 5:
            return "Anomalie extrême - vérifier si erreur de saisie ou de mesure"
        elif abs(z_score) > 3:
            return "Anomalie significative - investiguer la source de données"
        elif len(reasons) > 1:
            return "Plusieurs indicateurs d'anomalie - valider avec le contexte métier"
        else:
            return "Anomalie légère - pourrait être une valeur rare mais valide"


# Factory pour créer les tools
def create_anomaly_tools(df: pd.DataFrame) -> list[BaseTool]:
    """
    Crée les outils de détection d'anomalies avec un DataFrame.

    Args:
        df: DataFrame à analyser

    Returns:
        Liste d'outils configurés
    """
    detection_tool = AnomalyDetectionTool()
    detection_tool.dataframe = df

    investigation_tool = AnomalyInvestigationTool()
    investigation_tool.dataframe = df

    return [detection_tool, investigation_tool]
