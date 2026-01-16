"""
Script de démonstration de DataSentinel AI.

Ce script montre comment utiliser le système en mode programmatique
(sans passer par l'API).

Usage:
    python scripts/demo.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.agents.orchestrator import OrchestratorAgent, TaskType
from src.core.models import AgentContext


def create_sample_data() -> pd.DataFrame:
    """Crée un dataset d'exemple avec des problèmes de qualité."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 5, 7, 8, 9, 10],  # Duplicate ID
        "name": ["Alice", "Bob", None, "Diana", "Eve", "Frank", "Grace", "", "Ivan", "Julia"],
        "age": [25, 30, 35, 200, 28, 32, 45, 29, -5, 38],  # Anomalies: 200, -5
        "email": [
            "alice@test.com",
            "bob@test.com",
            "invalid-email",  # Format invalide
            "diana@test.com",
            "eve@test.com",
            "frank@test.com",
            "grace@test.com",
            "henry@test.com",
            "ivan@test.com",
            "julia@test.com"
        ],
        "salary": [50000, 60000, 55000, 70000, None, 65000, 80000, None, 45000, 75000],
        "department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "HR", "IT", "Finance"]
    })


def print_section(title: str) -> None:
    """Affiche un titre de section."""
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def main():
    """Exécute la démonstration."""
    print_section("DataSentinel AI - Démonstration")

    # Créer les données
    print("\n📊 Création du dataset d'exemple...")
    df = create_sample_data()
    print(f"   Dimensions: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"   Colonnes: {list(df.columns)}")

    # Créer l'orchestrateur
    print("\n🤖 Initialisation de l'orchestrateur...")
    orchestrator = OrchestratorAgent()

    # Créer le contexte
    context = AgentContext(
        session_id="demo_session_001",
        dataset_id="demo_dataset_001"
    )

    print_section("Étape 1: Analyse complète")

    print("\n🔍 Exécution du pipeline d'analyse...")
    try:
        result = orchestrator.analyze(df, session_id="demo_session_001")

        print(f"\n✅ Analyse terminée!")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Score de qualité: {result['quality_score']}%")
        print(f"   Problèmes détectés: {len(result['issues'])}")

        if result['issues']:
            print("\n📋 Problèmes détectés:")
            for i, issue in enumerate(result['issues'][:5], 1):
                print(f"   {i}. [{issue['severity'].upper()}] {issue['description']}")
                print(f"      Colonne: {issue['column']}, Confiance: {issue['confidence']:.0%}")

        if result.get('needs_human_review'):
            print("\n⚠️ REVUE HUMAINE RECOMMANDÉE")

    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse: {e}")
        print("   Note: Assurez-vous que ANTHROPIC_API_KEY est configuré dans .env")
        return

    print_section("Étape 2: Recommandations")

    print("\n💡 Génération des recommandations de correction...")
    try:
        result = orchestrator.recommend(df, session_id="demo_session_002")

        print(f"\n✅ Recommandations générées!")
        print(f"   Propositions: {len(result['proposals'])}")
        print(f"   Amélioration estimée: +{result['estimated_improvement']}%")

        if result['proposals']:
            print("\n📝 Corrections proposées:")
            for i, prop in enumerate(result['proposals'][:5], 1):
                print(f"   {i}. {prop['description']}")
                print(f"      Type: {prop['correction_type']}")
                print(f"      Confiance: {prop['confidence']:.0%}")
                print(f"      Justification: {prop['justification'][:100]}...")

    except Exception as e:
        print(f"\n❌ Erreur lors des recommandations: {e}")

    print_section("Étape 3: Pipeline complet avec validation")

    print("\n🔄 Exécution du pipeline complet...")
    try:
        result = orchestrator.full_analysis(df, session_id="demo_session_003")

        print(f"\n✅ Pipeline complet terminé!")
        print(f"   Score de qualité: {result['quality_score']}%")
        print(f"   Problèmes: {len(result.get('issues', []))}")
        print(f"   Propositions: {len(result.get('proposals', []))}")
        print(f"   Validations: {len(result.get('validations', []))}")

        approved = result.get('approved_corrections', [])
        print(f"\n✅ Corrections approuvées: {len(approved)}")
        for corr in approved:
            print(f"   - {corr['description']}")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")

    print_section("Fin de la démonstration")
    print("\n🎉 Démonstration terminée!")
    print("   Pour utiliser l'API, lancez: uvicorn src.api.main:app --reload")
    print("   Documentation: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
