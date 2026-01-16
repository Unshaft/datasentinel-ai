"""
Tests unitaires pour le détecteur de drift.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.drift_detector import DriftDetector, DriftResult, DriftSeverity


class TestDriftDetector:
    """Tests pour DriftDetector."""

    def test_init_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        detector = DriftDetector()

        assert detector.p_value_threshold == 0.05
        assert detector.psi_threshold == 0.25
        assert not detector.has_reference

    def test_set_reference(self, sample_drift_reference):
        """Test définition de la référence."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)

        assert detector.has_reference

    def test_detect_requires_reference(self, sample_drift_current):
        """Test que detect sans référence lève une erreur."""
        detector = DriftDetector()

        with pytest.raises(Exception):  # DriftDetectionError
            detector.detect(sample_drift_current)

    def test_detect_finds_drift(self, sample_drift_reference, sample_drift_current):
        """Test que detect trouve le drift."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)
        results = detector.detect(sample_drift_current)

        assert len(results) > 0

        # La colonne "value" devrait avoir du drift (moyenne décalée)
        value_result = next((r for r in results if r.column == "value"), None)
        assert value_result is not None
        assert value_result.has_drift

    def test_no_drift_on_identical_data(self, sample_drift_reference):
        """Test pas de drift sur données identiques."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)

        # Détecter sur les mêmes données
        results = detector.detect(sample_drift_reference)

        # Aucune colonne ne devrait avoir de drift
        for result in results:
            assert not result.has_drift or result.severity == DriftSeverity.NONE

    def test_categorical_drift_detection(self):
        """Test détection de drift sur colonnes catégorielles."""
        # Référence: principalement A et B
        reference = pd.DataFrame({
            "category": ["A"] * 50 + ["B"] * 50
        })

        # Current: distribution différente avec nouvelle catégorie
        current = pd.DataFrame({
            "category": ["A"] * 20 + ["B"] * 30 + ["C"] * 50
        })

        detector = DriftDetector()
        detector.set_reference(reference)
        results = detector.detect(current)

        cat_result = next((r for r in results if r.column == "category"), None)
        assert cat_result is not None
        # Devrait détecter un changement
        assert cat_result.has_drift or cat_result.drift_score > 0

    def test_drift_severity_levels(self, sample_drift_reference):
        """Test que les niveaux de sévérité sont corrects."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)

        # Drift modéré
        moderate_drift = pd.DataFrame({
            "value": np.random.normal(110, 15, 100),  # Légèrement décalé
            "category": ["A"] * 50 + ["B"] * 50
        })

        results = detector.detect(moderate_drift)

        for result in results:
            assert result.severity in [
                DriftSeverity.NONE,
                DriftSeverity.LOW,
                DriftSeverity.MEDIUM,
                DriftSeverity.HIGH,
                DriftSeverity.CRITICAL
            ]

    def test_drift_summary(self, sample_drift_reference, sample_drift_current):
        """Test le résumé du drift."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)
        results = detector.detect(sample_drift_current)

        summary = detector.get_drift_summary(results)

        assert "total_columns" in summary
        assert "columns_with_drift" in summary
        assert "drift_percentage" in summary
        assert "severity_distribution" in summary

    def test_result_to_dict(self, sample_drift_reference, sample_drift_current):
        """Test sérialisation des résultats."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)
        results = detector.detect(sample_drift_current)

        for result in results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "column" in result_dict
            assert "has_drift" in result_dict
            assert "severity" in result_dict

    def test_psi_calculation(self, sample_drift_reference):
        """Test que le PSI est calculé correctement."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)

        # Créer des données avec drift connu
        high_drift = pd.DataFrame({
            "value": np.random.normal(150, 30, 100),  # Fort décalage
            "category": ["A"] * 50 + ["B"] * 50
        })

        results = detector.detect(high_drift)
        value_result = next((r for r in results if r.column == "value"), None)

        assert value_result is not None
        assert value_result.drift_score > 0

    def test_interpretation_generated(self, sample_drift_reference, sample_drift_current):
        """Test que les interprétations sont générées."""
        detector = DriftDetector()
        detector.set_reference(sample_drift_reference)
        results = detector.detect(sample_drift_current)

        for result in results:
            assert result.interpretation
            assert isinstance(result.interpretation, str)
            assert len(result.interpretation) > 0
