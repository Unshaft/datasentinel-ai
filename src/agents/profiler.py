"""
Agent Profiler - Analyse et profilage des données.

Cet agent est responsable de la première analyse du dataset:
- Extraction du schéma et des types
- Calcul des statistiques descriptives
- Identification des patterns dans les données
- Génération d'un profil complet pour les autres agents
"""

import hashlib
import json
import time
import uuid
from typing import Any

import pandas as pd

from src.agents.base import AgentResult, BaseAgent
from src.core.models import (
    AgentContext,
    AgentType,
    ColumnProfile,
    DataProfile,
)
from src.tools.statistical import create_statistical_tools


class ProfilerAgent(BaseAgent):
    """
    Agent spécialisé dans le profilage des données.

    Rôle: Établir la "carte d'identité" complète d'un dataset
    avant toute analyse de qualité.

    Le profil généré sert de base pour:
    - La détection d'anomalies (valeurs de référence)
    - La détection de drift (baseline)
    - La validation des règles métier (types et contraintes)
    """

    def __init__(self) -> None:
        """Initialise le Profiler Agent."""
        super().__init__(
            agent_type=AgentType.PROFILER,
            tools=[]  # Tools seront injectés à l'exécution
        )

    @property
    def system_prompt(self) -> str:
        """Prompt système du Profiler."""
        return """Tu es un Data Profiler Agent expert dans l'analyse exploratoire des données.

Ton rôle est d'analyser un dataset et d'en extraire un profil complet comprenant:
1. Les métadonnées générales (dimensions, mémoire)
2. Le profil de chaque colonne (types, statistiques, distributions)
3. Les patterns et relations entre colonnes
4. Les premières observations sur la qualité

Tu dois être:
- PRÉCIS: Chaque statistique doit être exacte
- EXHAUSTIF: Ne manquer aucune colonne ou caractéristique importante
- OBJECTIF: Rapporter les faits sans interprétation excessive
- CONCIS: Aller à l'essentiel tout en étant complet

Format de sortie attendu:
- Un résumé structuré du dataset
- Les statistiques clés par colonne
- Les observations préliminaires sur la qualité

Tu ne dois PAS:
- Proposer de corrections (c'est le rôle du Corrector)
- Valider contre des règles métier (c'est le rôle du Validator)
- Détecter les anomalies en détail (c'est le rôle du Quality Agent)
"""

    def execute(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> AgentContext:
        """
        Exécute le profilage du dataset.

        Args:
            context: Contexte de la session
            df: DataFrame à profiler
            **kwargs: Arguments additionnels

        Returns:
            Contexte mis à jour avec le profil
        """
        start_time = time.time()

        # Générer le profil
        profile = self._generate_profile(context.dataset_id, df)

        # Mettre à jour le contexte
        context.profile = profile
        context.current_step = "profiled"
        context.iteration += 1

        # Calculer la confiance
        confidence = self._calculate_profile_confidence(df, profile)

        # Générer un résumé pour le logging
        summary = self._generate_summary(profile)

        # Logger la décision
        processing_time = int((time.time() - start_time) * 1000)
        self._log_decision(
            context=context,
            action="profile_dataset",
            reasoning=f"Profilé {profile.row_count} lignes x {profile.column_count} colonnes",
            input_summary=f"Dataset: {profile.source}",
            output_summary=summary,
            confidence=confidence.overall_score,
            processing_time_ms=processing_time
        )

        return context

    def _generate_profile(
        self,
        dataset_id: str,
        df: pd.DataFrame
    ) -> DataProfile:
        """
        Génère le profil complet du DataFrame.

        Args:
            dataset_id: Identifiant du dataset
            df: DataFrame à analyser

        Returns:
            Profil complet
        """
        # Profiler chaque colonne
        column_profiles = []
        for col in df.columns:
            profile = self._profile_column(df, col)
            column_profiles.append(profile)

        # Calculer le hash pour la détection de drift future
        data_hash = self._compute_data_hash(df)

        return DataProfile(
            dataset_id=dataset_id,
            source=f"dataframe_{dataset_id}",
            row_count=len(df),
            column_count=len(df.columns),
            memory_size_bytes=df.memory_usage(deep=True).sum(),
            columns=column_profiles,
            data_hash=data_hash
        )

    def _profile_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """
        Profile une colonne spécifique.

        Args:
            df: DataFrame source
            column: Nom de la colonne

        Returns:
            Profil de la colonne
        """
        series = df[column]
        dtype = str(series.dtype)

        # Statistiques de base
        count = int(series.count())
        null_count = int(series.isna().sum())
        total = len(series)
        null_percentage = round(100 * null_count / total, 2) if total > 0 else 0
        unique_count = int(series.nunique())
        unique_percentage = round(100 * unique_count / total, 2) if total > 0 else 0

        # Inférer le type sémantique
        inferred_type = self._infer_semantic_type(series)

        # Initialiser le profil
        profile_data = {
            "name": column,
            "dtype": dtype,
            "inferred_type": inferred_type,
            "count": count,
            "null_count": null_count,
            "null_percentage": null_percentage,
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
        }

        # Statistiques numériques
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            profile_data.update({
                "mean": float(desc["mean"]) if pd.notna(desc.get("mean")) else None,
                "std": float(desc["std"]) if pd.notna(desc.get("std")) else None,
                "min": float(desc["min"]) if pd.notna(desc.get("min")) else None,
                "max": float(desc["max"]) if pd.notna(desc.get("max")) else None,
                "q25": float(desc["25%"]) if "25%" in desc else None,
                "q50": float(desc["50%"]) if "50%" in desc else None,
                "q75": float(desc["75%"]) if "75%" in desc else None,
            })

        # Échantillons
        sample_values = series.dropna().head(5).tolist()
        profile_data["sample_values"] = [
            str(v) if not isinstance(v, (int, float, bool)) else v
            for v in sample_values
        ]

        return ColumnProfile(**profile_data)

    def _infer_semantic_type(self, series: pd.Series) -> str:
        """
        Infère le type sémantique d'une série.

        Args:
            series: Série Pandas

        Returns:
            Type sémantique (integer, float, categorical, datetime, text, etc.)
        """
        dtype = series.dtype

        # Types numériques
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"

        # Types temporels
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # Types booléens
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"

        # Types object - analyse plus poussée
        if dtype == "object":
            return self._infer_object_type(series)

        # Type catégoriel explicite
        if pd.api.types.is_categorical_dtype(dtype):
            return "categorical"

        return "unknown"

    def _infer_object_type(self, series: pd.Series) -> str:
        """
        Infère le type sémantique d'une série object.

        Analyse les valeurs pour déterminer s'il s'agit de:
        - Texte libre
        - Catégorie
        - Date (string)
        - Email
        - etc.
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return "empty"

        n_unique = non_null.nunique()
        n_total = len(non_null)

        # Faible cardinalité -> probablement catégoriel
        if n_unique < 50 or (n_unique / n_total) < 0.05:
            return "categorical"

        # Vérifier si ce sont des dates en string
        sample = non_null.head(100)
        try:
            pd.to_datetime(sample, errors="raise", format="mixed")
            return "datetime_string"
        except (ValueError, TypeError):
            pass

        # Analyser la longueur moyenne des strings
        avg_len = non_null.astype(str).str.len().mean()

        if avg_len > 100:
            return "text_long"
        elif avg_len > 30:
            return "text_short"
        else:
            # Pourrait être un ID ou une catégorie à haute cardinalité
            if n_unique == n_total:
                return "identifier"
            return "categorical_high_cardinality"

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        Calcule un hash représentatif du dataset.

        Le hash est basé sur la structure et un échantillon des données,
        pas sur l'intégralité (pour performance).

        Args:
            df: DataFrame

        Returns:
            Hash MD5
        """
        # Combiner structure et échantillon
        structure = str(list(df.columns)) + str(list(df.dtypes))
        sample = df.head(100).to_json() if len(df) > 0 else ""
        stats = str(df.describe().to_dict()) if len(df) > 0 else ""

        combined = f"{structure}|{sample}|{stats}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _calculate_profile_confidence(
        self,
        df: pd.DataFrame,
        profile: DataProfile
    ) -> Any:
        """
        Calcule la confiance dans le profil généré.

        Args:
            df: DataFrame original
            profile: Profil généré

        Returns:
            Score de confiance
        """
        # La confiance dépend de:
        # - Taille de l'échantillon
        # - Proportion de valeurs nulles
        # - Variété des types

        sample_size = len(df)

        # Score de qualité basé sur les nulls
        total_cells = profile.row_count * profile.column_count
        null_ratio = profile.total_null_count / total_cells if total_cells > 0 else 0
        data_quality = 1.0 - min(null_ratio, 0.5)  # Plafonner l'impact

        return self._calculate_confidence(
            data_quality=data_quality,
            sample_size=sample_size,
            signal_scores=[1.0],  # Profiling est déterministe
            rule_coverage=1.0
        )

    def _generate_summary(self, profile: DataProfile) -> str:
        """
        Génère un résumé textuel du profil.

        Args:
            profile: Profil du dataset

        Returns:
            Résumé formaté
        """
        null_cols = [c.name for c in profile.columns if c.has_nulls]
        numeric_cols = [c.name for c in profile.columns if c.is_numeric]

        summary_parts = [
            f"Dataset: {profile.row_count} lignes x {profile.column_count} colonnes",
            f"Mémoire: {profile.memory_size_bytes / 1024 / 1024:.2f} MB",
            f"Colonnes numériques: {len(numeric_cols)}",
            f"Colonnes avec nulls: {len(null_cols)}",
            f"Total valeurs nulles: {profile.total_null_count}",
        ]

        return " | ".join(summary_parts)

    def profile_with_llm_analysis(
        self,
        context: AgentContext,
        df: pd.DataFrame
    ) -> tuple[AgentContext, str]:
        """
        Profile le dataset avec une analyse LLM additionnelle.

        En plus du profil technique, demande au LLM une analyse
        qualitative des données.

        Args:
            context: Contexte de la session
            df: DataFrame

        Returns:
            Tuple (contexte mis à jour, analyse LLM)
        """
        # D'abord, exécuter le profiling standard
        context = self.execute(context, df)
        profile = context.profile

        # Préparer le prompt pour l'analyse LLM
        profile_summary = json.dumps({
            "rows": profile.row_count,
            "columns": profile.column_count,
            "null_percentage": round(
                100 * profile.total_null_count / (profile.row_count * profile.column_count), 2
            ) if profile.row_count * profile.column_count > 0 else 0,
            "columns_info": [
                {
                    "name": c.name,
                    "type": c.inferred_type,
                    "null_pct": c.null_percentage,
                    "unique_pct": c.unique_percentage
                }
                for c in profile.columns
            ]
        }, indent=2)

        analysis_prompt = f"""Voici le profil d'un dataset:

{profile_summary}

Analyse ce dataset et fournis:
1. Une description générale de ce que semble contenir ce dataset
2. Les points d'attention potentiels sur la qualité
3. Des recommandations pour les prochaines étapes d'analyse

Sois concis et factuel."""

        # Invoquer le LLM pour l'analyse
        response = self._invoke_llm(analysis_prompt, include_tools=False)
        llm_analysis = response.content

        # Stocker l'analyse dans les métadonnées du contexte
        context.metadata["llm_analysis"] = llm_analysis

        return context, llm_analysis
