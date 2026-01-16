"""
Utilitaires pour le chargement de données.

Ce module fournit des fonctions pour charger des datasets
depuis différentes sources (fichiers, JSON, etc.).
"""

import io
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import settings
from src.core.exceptions import DataLoadError, DataSizeExceededError, EmptyDataError


def load_from_dict(data: dict[str, list[Any]]) -> pd.DataFrame:
    """
    Charge un DataFrame depuis un dictionnaire.

    Args:
        data: Dictionnaire {colonne: [valeurs]}

    Returns:
        DataFrame

    Raises:
        DataLoadError: Si le format est invalide
    """
    try:
        df = pd.DataFrame(data)
        _validate_dataframe(df, "dict")
        return df
    except Exception as e:
        raise DataLoadError(
            source="dict",
            reason=str(e),
            original_error=e
        )


def load_from_csv_string(csv_content: str) -> pd.DataFrame:
    """
    Charge un DataFrame depuis une chaîne CSV.

    Args:
        csv_content: Contenu CSV sous forme de string

    Returns:
        DataFrame

    Raises:
        DataLoadError: Si le parsing échoue
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        _validate_dataframe(df, "csv_string")
        return df
    except Exception as e:
        raise DataLoadError(
            source="csv_string",
            reason=str(e),
            original_error=e
        )


def load_from_file(file_path: str | Path) -> pd.DataFrame:
    """
    Charge un DataFrame depuis un fichier.

    Supporte: CSV, Excel, JSON, Parquet

    Args:
        file_path: Chemin vers le fichier

    Returns:
        DataFrame

    Raises:
        DataLoadError: Si le fichier ne peut pas être lu
    """
    path = Path(file_path)

    if not path.exists():
        raise DataLoadError(
            source=str(path),
            reason="Fichier non trouvé"
        )

    # Vérifier la taille
    file_size = path.stat().st_size
    if file_size > settings.max_upload_size:
        raise DataSizeExceededError(
            actual_size=file_size,
            max_size=settings.max_upload_size,
            unit="bytes"
        )

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif suffix == ".json":
            df = pd.read_json(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise DataLoadError(
                source=str(path),
                reason=f"Format non supporté: {suffix}"
            )

        _validate_dataframe(df, str(path))
        return df

    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(
            source=str(path),
            reason=str(e),
            original_error=e
        )


def _validate_dataframe(df: pd.DataFrame, source: str) -> None:
    """
    Valide un DataFrame chargé.

    Args:
        df: DataFrame à valider
        source: Nom de la source pour les messages d'erreur

    Raises:
        EmptyDataError: Si le DataFrame est vide
        DataSizeExceededError: Si trop de lignes
    """
    if df.empty:
        raise EmptyDataError(source=source)

    if settings.max_rows_analyze > 0 and len(df) > settings.max_rows_analyze:
        raise DataSizeExceededError(
            actual_size=len(df),
            max_size=settings.max_rows_analyze,
            unit="rows"
        )


def sample_dataframe(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """
    Échantillonne un DataFrame si nécessaire.

    Args:
        df: DataFrame source
        max_rows: Nombre max de lignes (None = config)

    Returns:
        DataFrame (potentiellement échantillonné)
    """
    max_rows = max_rows or settings.max_rows_analyze

    if max_rows > 0 and len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)

    return df


def get_memory_usage(df: pd.DataFrame) -> dict[str, float]:
    """
    Calcule l'utilisation mémoire d'un DataFrame.

    Args:
        df: DataFrame

    Returns:
        Dict avec total et par colonne (en MB)
    """
    memory = df.memory_usage(deep=True)

    return {
        "total_mb": round(memory.sum() / 1024 / 1024, 2),
        "by_column": {
            col: round(memory[col] / 1024 / 1024, 4)
            for col in df.columns
        }
    }
