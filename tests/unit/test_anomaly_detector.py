"""
Tests unitaires pour le détecteur d'anomalies.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.anomaly_detector import AnomalyDetector, AnomalyResult


class TestAnomalyDetector:
    """Tests pour AnomalyDetector."""

    def test_init_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        detector = AnomalyDetector()

        assert detector.contamination == 0.1
        assert detector.n_estimators == 100
        assert detector.random_state == 42
        assert not detector.is_fitted

    def test_init_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        detector = AnomalyDetector(
            contamination=0.05,
            n_estimators=50,
            random_state=123
        )

        assert detector.contamination == 0.05
        assert detector.n_estimators == 50
        assert detector.random_state == 123

    def test_invalid_contamination(self):
        """Test que contamination invalide lève une erreur."""
        with pytest.raises(ValueError):
            AnomalyDetector(contamination=0.0)

        with pytest.raises(ValueError):
            AnomalyDetector(contamination=0.6)

    def test_fit_creates_models(self, sample_numeric_df):
        """Test que fit crée les modèles."""
        detector = AnomalyDetector()
        detector.fit(sample_numeric_df)

        assert detector.is_fitted
        assert "value" in detector.fitted_columns

    def test_fit_ignores_non_numeric(self, sample_dirty_df):
        """Test que fit ignore les colonnes non numériques."""
        detector = AnomalyDetector()
        detector.fit(sample_dirty_df)

        # Seules les colonnes numériques doivent être fittées
        assert "name" not in detector.fitted_columns
        assert "email" not in detector.fitted_columns

    def test_detect_finds_anomalies(self, sample_numeric_df):
        """Test que detect trouve les anomalies."""
        detector = AnomalyDetector(contamination=0.1)
        results = detector.fit_detect(sample_numeric_df)

        assert len(results) > 0

        # Vérifier qu'on a trouvé des anomalies
        total_anomalies = sum(r.anomaly_count for r in results)
        assert total_anomalies > 0

    def test_detect_returns_correct_structure(self, sample_numeric_df):
        """Test la structure des résultats."""
        detector = AnomalyDetector()
        results = detector.fit_detect(sample_numeric_df)

        for result in results:
            assert isinstance(result, AnomalyResult)
            assert isinstance(result.column, str)
            assert isinstance(result.anomaly_indices, list)
            assert isinstance(result.anomaly_count, int)
            assert 0 <= result.anomaly_percentage <= 100

    def test_detect_without_fit_raises_error(self, sample_numeric_df):
        """Test que detect sans fit lève une erreur."""
        detector = AnomalyDetector()

        with pytest.raises(Exception):  # MLError
            detector.detect(sample_numeric_df)

    def test_fit_detect_convenience_method(self, sample_numeric_df):
        """Test la méthode fit_detect."""
        detector = AnomalyDetector()
        results = detector.fit_detect(sample_numeric_df)

        assert detector.is_fitted
        assert len(results) > 0

    def test_anomaly_summary(self, sample_numeric_df):
        """Test le résumé des anomalies."""
        detector = AnomalyDetector()
        results = detector.fit_detect(sample_numeric_df)
        summary = detector.get_anomaly_summary(results)

        assert "total_anomalies" in summary
        assert "columns_affected" in summary
        assert "average_anomaly_percentage" in summary

    def test_empty_results_summary(self):
        """Test le résumé avec résultats vides."""
        detector = AnomalyDetector()
        summary = detector.get_anomaly_summary([])

        assert summary["total_anomalies"] == 0
        assert summary["columns_affected"] == 0

    def test_reproducibility(self, sample_numeric_df):
        """Test que les résultats sont reproductibles avec même seed."""
        detector1 = AnomalyDetector(random_state=42)
        results1 = detector1.fit_detect(sample_numeric_df)

        detector2 = AnomalyDetector(random_state=42)
        results2 = detector2.fit_detect(sample_numeric_df)

        # Les anomalies détectées doivent être identiques
        assert results1[0].anomaly_indices == results2[0].anomaly_indices

    def test_to_dict_serialization(self, sample_numeric_df):
        """Test la sérialisation en dict."""
        detector = AnomalyDetector()
        results = detector.fit_detect(sample_numeric_df)

        for result in results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "column" in result_dict
            assert "anomaly_count" in result_dict
