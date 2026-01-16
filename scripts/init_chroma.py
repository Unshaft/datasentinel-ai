"""
Script d'initialisation de ChromaDB.

Ce script initialise ChromaDB avec les règles métier par défaut.
À exécuter une fois après l'installation.

Usage:
    python scripts/init_chroma.py
"""

import json
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.chroma_store import ChromaStore


def load_default_rules(store: ChromaStore) -> int:
    """
    Charge les règles par défaut depuis le fichier JSON.

    Args:
        store: Instance ChromaStore

    Returns:
        Nombre de règles chargées
    """
    rules_file = Path("data/rules/default_rules.json")

    if not rules_file.exists():
        print(f"⚠️ Fichier de règles non trouvé: {rules_file}")
        return 0

    with open(rules_file, encoding="utf-8") as f:
        data = json.load(f)

    rules = data.get("rules", [])
    loaded = 0

    for rule in rules:
        try:
            store.add_rule(
                rule_id=rule["id"],
                rule_text=rule["text"],
                rule_type=rule["type"],
                metadata={
                    "severity": rule.get("severity", "medium"),
                    "category": rule.get("category", "general"),
                    "source": "default"
                }
            )
            loaded += 1
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {rule['id']}: {e}")

    return loaded


def main():
    """Point d'entrée principal."""
    print("🚀 Initialisation de ChromaDB pour DataSentinel AI...")
    print("-" * 50)

    try:
        # Initialiser le store
        store = ChromaStore()
        stats = store.get_stats()

        print(f"📁 Chemin: {store.persist_path}")
        print(f"📊 Stats actuelles:")
        print(f"   - Règles: {stats['rules_count']}")
        print(f"   - Décisions: {stats['decisions_count']}")
        print(f"   - Feedbacks: {stats['feedback_count']}")
        print()

        # Charger les règles si collection vide
        if stats["rules_count"] == 0:
            print("📝 Chargement des règles par défaut...")
            loaded = load_default_rules(store)
            print(f"✅ {loaded} règles chargées")
        else:
            print("ℹ️ Des règles existent déjà, skip du chargement")

        # Stats finales
        print()
        print("-" * 50)
        final_stats = store.get_stats()
        print("📊 Stats finales:")
        print(f"   - Règles: {final_stats['rules_count']}")
        print(f"   - Décisions: {final_stats['decisions_count']}")
        print(f"   - Feedbacks: {final_stats['feedback_count']}")

        print()
        print("✅ Initialisation terminée!")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
