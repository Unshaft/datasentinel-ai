"""
Fixtures pytest partagées pour les tests.

Ce module définit les fixtures utilisables dans tous les tests.
"""

import os
from typing import Generator

import pandas as pd
import pytest

# Configurer les variables d'environnement pour les tests
os.environ["ANTHROPIC_API_KEY"] = "sk-test-key-for-testing"
os.environ["ENVIRONMENT"] = "development"
os.environ["CHROMA_PERSIST_PATH"] = "./test_data/chroma"


@pytest.fixture
def sample_clean_df() -> pd.DataFrame:
    """DataFrame propre sans problèmes de qualité."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 32],
        "salary": [50000, 60000, 55000, 70000, 65000],
        "department": ["IT", "HR", "IT", "Finance", "HR"]
    })


@pytest.fixture
def sample_dirty_df() -> pd.DataFrame:
    """DataFrame avec divers problèmes de qualité."""
    return pd.DataFrame({
        "id": [1, 2, 2, 4, 5],  # Duplicates
        "name": ["Alice", "Bob", None, "Diana", ""],  # Nulls et empty
        "age": [25, 30, 35, 200, 28],  # Anomalie (200)
        "salary": [50000, -5000, 55000, 70000, None],  # Négatif et null
        "email": [
            "alice@test.com",
            "invalid-email",  # Format invalide
            "charlie@test.com",
            "diana@test.com",
            "eve@test.com"
        ]
    })


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    """DataFrame avec colonnes numériques pour tests ML."""
    import numpy as np

    np.random.seed(42)

    # Données normales
    normal_data = np.random.normal(100, 15, 100)

    # Ajouter quelques anomalies
    anomalies = [250, 300, -50, 350]
    data = np.concatenate([normal_data, anomalies])

    return pd.DataFrame({
        "value": data,
        "category": ["A"] * 50 + ["B"] * 50 + ["C"] * 4
    })


@pytest.fixture
def sample_drift_reference() -> pd.DataFrame:
    """DataFrame de référence pour tests de drift."""
    import numpy as np

    np.random.seed(42)

    return pd.DataFrame({
        "value": np.random.normal(100, 15, 100),
        "category": ["A"] * 50 + ["B"] * 50
    })


@pytest.fixture
def sample_drift_current() -> pd.DataFrame:
    """DataFrame avec drift pour tests."""
    import numpy as np

    np.random.seed(123)

    # Drift: moyenne décalée de 100 à 120
    return pd.DataFrame({
        "value": np.random.normal(120, 20, 100),  # Drift!
        "category": ["A"] * 30 + ["B"] * 50 + ["C"] * 20  # Nouvelle catégorie
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """DataFrame vide pour tests de cas limites."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """DataFrame avec une seule ligne."""
    return pd.DataFrame({
        "id": [1],
        "value": [100]
    })


@pytest.fixture
def large_null_df() -> pd.DataFrame:
    """DataFrame avec beaucoup de valeurs nulles."""
    return pd.DataFrame({
        "id": range(100),
        "mostly_null": [None] * 90 + list(range(10)),
        "half_null": [None if i % 2 == 0 else i for i in range(100)],
        "no_null": list(range(100))
    })


# Fixtures pour les tests d'intégration

@pytest.fixture
def mock_chroma_store():
    """Mock du ChromaStore pour tests sans DB."""
    from unittest.mock import MagicMock

    store = MagicMock()
    store.search_rules.return_value = []
    store.get_all_rules.return_value = []
    store.add_rule.return_value = "rule_test_123"
    store.get_stats.return_value = {
        "rules_count": 0,
        "decisions_count": 0,
        "feedback_count": 0
    }

    return store


@pytest.fixture
def mock_llm_response():
    """Mock de réponse LLM pour tests sans API."""
    from unittest.mock import MagicMock

    response = MagicMock()
    response.content = "This is a mock LLM response for testing."

    return response
