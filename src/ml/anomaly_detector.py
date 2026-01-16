"""
Détecteur d'anomalies basé sur Isolation Forest.

Ce module implémente la détection d'anomalies statistiques dans les données.
L'Isolation Forest est choisi pour:
- Sa capacité à détecter des anomalies sans supervision
- Sa robustesse face aux données de haute dimension
- Son efficacité computationnelle (O(n log n))

L'anomalie detection aide les agents à identifier des valeurs
suspectes qui nécessitent une investigation ou correction.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.core.exceptions import InsufficientDataError, MLError


@dataclass
class AnomalyResult:
    """Résultat de détection d'anomalie pour une colonne."""

    column: str
    anomaly_indices: list[int]
    anomaly_scores: list[float]
    anomaly_values: list[Any]
    anomaly_count: int
    anomaly_percentage: float
    threshold_used: float
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "column": self.column,
            "anomaly_indices": self.anomaly_indices,
            "anomaly_scores": self.anomaly_scores,
            "anomaly_values": self.anomaly_values,
            "anomaly_count": self.anomaly_count,
            "anomaly_percentage": self.anomaly_percentage,
            "threshold_used": self.threshold_used,
            "model_id": self.model_id,
        }


class AnomalyDetector:
    """
    Détecteur d'anomalies utilisant Isolation Forest.

    Isolation Forest fonctionne sur le principe que les anomalies
    sont "peu nombreuses et différentes", donc plus faciles à isoler
    dans un arbre de décision aléatoire.

    Attributes:
        contamination: Proportion attendue d'anomalies (0.01 à 0.5)
        n_estimators: Nombre d'arbres dans la forêt
        random_state: Seed pour reproductibilité
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> None:
        """
        Initialise le détecteur.

        Args:
            contamination: Proportion attendue d'anomalies (défaut: 10%)
            n_estimators: Nombre d'arbres (plus = plus précis, plus lent)
            random_state: Seed pour reproductibilité des résultats
        """
        self._validate_contamination(contamination)

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._models: dict[str, IsolationForest] = {}
        self._scalers: dict[str, StandardScaler] = {}
        self._fitted_columns: set[str] = set()

    @staticmethod
    def _validate_contamination(contamination: float) -> None:
        """Valide le paramètre contamination."""
        if not 0 < contamination < 0.5:
            raise ValueError(
                f"contamination doit être entre 0 et 0.5, reçu: {contamination}"
            )

    def _generate_model_id(self, column: str) -> str:
        """Génère un ID unique pour un modèle entraîné."""
        data = f"{column}_{self.contamination}_{self.n_estimators}_{self.random_state}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def fit(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None
    ) -> "AnomalyDetector":
        """
        Entraîne le détecteur sur les colonnes numériques.

        Args:
            df: DataFrame avec les données
            columns: Colonnes à analyser (None = toutes les numériques)

        Returns:
            Self pour chaînage

        Raises:
            InsufficientDataError: Si pas assez de données
        """
        if len(df) < 10:
            raise InsufficientDataError(
                model_name="IsolationForest",
                required=10,
                actual=len(df)
            )

        # Sélection des colonnes
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for column in columns:
            if column not in df.columns:
                continue

            # Extraction et nettoyage des données
            data = df[column].dropna().values.reshape(-1, 1)

            if len(data) < 10:
                continue  # Pas assez de données pour cette colonne

            # Standardisation pour améliorer la détection
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Entraînement du modèle
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1  # Utilise tous les CPU
            )
            model.fit(data_scaled)

            # Stockage
            self._models[column] = model
            self._scalers[column] = scaler
            self._fitted_columns.add(column)

        return self

    def detect(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None
    ) -> list[AnomalyResult]:
        """
        Détecte les anomalies dans le DataFrame.

        Args:
            df: DataFrame à analyser
            columns: Colonnes à vérifier (None = toutes les entraînées)

        Returns:
            Liste de résultats par colonne

        Raises:
            MLError: Si aucun modèle n'est entraîné
        """
        if not self._fitted_columns:
            raise MLError(
                message="Aucun modèle entraîné. Appelez fit() d'abord.",
                details={"status": "not_fitted"}
            )

        results = []
        check_columns = columns or list(self._fitted_columns)

        for column in check_columns:
            if column not in self._fitted_columns:
                continue

            result = self._detect_column(df, column)
            if result:
                results.append(result)

        return results

    def _detect_column(
        self,
        df: pd.DataFrame,
        column: str
    ) -> AnomalyResult | None:
        """Détecte les anomalies dans une colonne spécifique."""
        if column not in df.columns:
            return None

        model = self._models[column]
        scaler = self._scalers[column]

        # Préparation des données
        series = df[column]
        non_null_mask = series.notna()
        data = series[non_null_mask].values.reshape(-1, 1)

        if len(data) == 0:
            return None

        # Standardisation et prédiction
        data_scaled = scaler.transform(data)
        predictions = model.predict(data_scaled)
        scores = model.decision_function(data_scaled)

        # -1 = anomalie, 1 = normal dans sklearn
        anomaly_mask = predictions == -1

        # Récupération des indices originaux
        original_indices = df.index[non_null_mask].tolist()
        anomaly_indices = [
            original_indices[i]
            for i, is_anomaly in enumerate(anomaly_mask)
            if is_anomaly
        ]
        anomaly_scores = [
            float(scores[i])
            for i, is_anomaly in enumerate(anomaly_mask)
            if is_anomaly
        ]
        anomaly_values = [
            float(data[i][0])
            for i, is_anomaly in enumerate(anomaly_mask)
            if is_anomaly
        ]

        return AnomalyResult(
            column=column,
            anomaly_indices=anomaly_indices,
            anomaly_scores=anomaly_scores,
            anomaly_values=anomaly_values,
            anomaly_count=len(anomaly_indices),
            anomaly_percentage=100.0 * len(anomaly_indices) / len(data),
            threshold_used=self.contamination,
            model_id=self._generate_model_id(column)
        )

    def fit_detect(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None
    ) -> list[AnomalyResult]:
        """
        Entraîne et détecte en une seule opération.

        Pratique pour une analyse one-shot sans réutilisation du modèle.

        Args:
            df: DataFrame à analyser
            columns: Colonnes à analyser

        Returns:
            Liste de résultats d'anomalie
        """
        self.fit(df, columns)
        return self.detect(df, columns)

    def get_anomaly_summary(
        self,
        results: list[AnomalyResult]
    ) -> dict[str, Any]:
        """
        Génère un résumé des anomalies détectées.

        Args:
            results: Résultats de détection

        Returns:
            Dictionnaire avec statistiques globales
        """
        if not results:
            return {
                "total_anomalies": 0,
                "columns_affected": 0,
                "most_affected_column": None,
                "average_anomaly_percentage": 0.0,
            }

        total_anomalies = sum(r.anomaly_count for r in results)
        most_affected = max(results, key=lambda r: r.anomaly_count)
        avg_percentage = np.mean([r.anomaly_percentage for r in results])

        return {
            "total_anomalies": total_anomalies,
            "columns_affected": len(results),
            "most_affected_column": most_affected.column,
            "most_affected_count": most_affected.anomaly_count,
            "average_anomaly_percentage": round(avg_percentage, 2),
            "by_column": {r.column: r.anomaly_count for r in results}
        }

    @property
    def is_fitted(self) -> bool:
        """Vérifie si au moins un modèle est entraîné."""
        return len(self._fitted_columns) > 0

    @property
    def fitted_columns(self) -> list[str]:
        """Retourne la liste des colonnes avec modèle entraîné."""
        return list(self._fitted_columns)
