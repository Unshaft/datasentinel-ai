"""
Client ChromaDB pour DataSentinel AI.

Ce module fournit une interface unifiée pour interagir avec ChromaDB,
gérant les trois collections principales:
- business_rules: Règles métier pour la validation
- decision_history: Historique des décisions pour l'apprentissage
- user_feedback: Feedbacks utilisateur pour l'amélioration continue

ChromaDB est choisi pour:
- Simplicité d'installation (pas de serveur externe requis)
- Support natif des embeddings
- Persistance locale facile
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.core.exceptions import ChromaDBError


class ChromaStore:
    """
    Store centralisé pour ChromaDB.

    Gère la connexion et les opérations CRUD sur les collections.
    Utilise le pattern Singleton pour une instance unique.

    Attributes:
        persist_path: Chemin de persistance des données
        client: Client ChromaDB
    """

    _instance: "ChromaStore | None" = None

    def __new__(cls, persist_path: Path | None = None) -> "ChromaStore":
        """Implémente le pattern Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_path: Path | None = None) -> None:
        """
        Initialise le store ChromaDB.

        Args:
            persist_path: Chemin de persistance (défaut: config)
        """
        if self._initialized:
            return

        self.persist_path = persist_path or settings.chroma_persist_path
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Configuration ChromaDB
        chroma_settings = ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
        )

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=chroma_settings
            )
        except Exception as e:
            raise ChromaDBError(
                operation="init",
                collection="*",
                reason=str(e),
                original_error=e
            )

        # Initialiser les collections
        self._init_collections()
        self._initialized = True

    def _init_collections(self) -> None:
        """Initialise les collections avec les bons paramètres."""
        # Collection pour les règles métier
        self.rules_collection = self.client.get_or_create_collection(
            name=settings.chroma_rules_collection,
            metadata={
                "description": "Business rules for data validation",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )

        # Collection pour l'historique des décisions
        self.decisions_collection = self.client.get_or_create_collection(
            name=settings.chroma_decisions_collection,
            metadata={
                "description": "History of agent decisions",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )

        # Collection pour les feedbacks utilisateur
        self.feedback_collection = self.client.get_or_create_collection(
            name=settings.chroma_feedback_collection,
            metadata={
                "description": "User feedback on decisions",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )

    # =========================================================================
    # BUSINESS RULES
    # =========================================================================

    def add_rule(
        self,
        rule_id: str,
        rule_text: str,
        rule_type: str,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Ajoute une règle métier à la collection.

        Args:
            rule_id: Identifiant unique de la règle
            rule_text: Texte de la règle en langage naturel
            rule_type: Type de règle (constraint, validation, format, etc.)
            metadata: Métadonnées additionnelles

        Returns:
            ID de la règle ajoutée
        """
        meta = metadata or {}
        meta.update({
            "rule_type": rule_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "active": True
        })

        try:
            self.rules_collection.add(
                ids=[rule_id],
                documents=[rule_text],
                metadatas=[meta]
            )
            return rule_id
        except Exception as e:
            raise ChromaDBError(
                operation="add_rule",
                collection=settings.chroma_rules_collection,
                reason=str(e),
                original_error=e
            )

    def search_rules(
        self,
        query: str,
        n_results: int = 5,
        rule_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Recherche des règles pertinentes par similarité sémantique.

        Args:
            query: Texte de recherche
            n_results: Nombre de résultats max
            rule_type: Filtrer par type de règle

        Returns:
            Liste de règles avec scores de similarité
        """
        where_filter = {"active": True}
        if rule_type:
            where_filter["rule_type"] = rule_type

        try:
            # ChromaDB lève une exception si la collection est vide.
            if self.rules_collection.count() == 0:
                return []

            # n_results ne peut pas dépasser le nombre de documents présents.
            actual_n = min(n_results, self.rules_collection.count())

            results = self.rules_collection.query(
                query_texts=[query],
                n_results=actual_n,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )

            rules = []
            if results["ids"] and results["ids"][0]:
                for i, rule_id in enumerate(results["ids"][0]):
                    rules.append({
                        "id": rule_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "similarity": 1 - (results["distances"][0][i] if results["distances"] else 0)
                    })

            return rules
        except Exception as e:
            raise ChromaDBError(
                operation="search_rules",
                collection=settings.chroma_rules_collection,
                reason=str(e),
                original_error=e
            )

    def get_relevant_rules(
        self,
        col_name: str,
        col_type: str,
        sample_values: list[Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retourne les règles pertinentes pour une colonne spécifique (Active RAG — F25).

        Args:
            col_name: Nom de la colonne
            col_type: Type de la colonne (numeric, object, datetime, …)
            sample_values: Quelques valeurs exemple pour enrichir la requête
            top_k: Nombre de règles à retourner

        Returns:
            Liste de règles avec similarité, triées par pertinence
        """
        # Requête sémantique enrichie
        samples_str = ""
        if sample_values:
            samples_str = f" values: {', '.join(str(v) for v in sample_values[:5])}"
        query = f"{col_name} {col_type}{samples_str}"

        try:
            return self.search_rules(query, n_results=top_k)
        except Exception:
            return []

    def get_all_rules(self, rule_type: str | None = None) -> list[dict[str, Any]]:
        """
        Récupère toutes les règles actives.

        Args:
            rule_type: Filtrer par type (optionnel)

        Returns:
            Liste de toutes les règles
        """
        where_filter = {"active": True}
        if rule_type:
            where_filter["rule_type"] = rule_type

        try:
            results = self.rules_collection.get(
                where=where_filter,
                include=["documents", "metadatas"]
            )

            rules = []
            if results["ids"]:
                for i, rule_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] or {}
                    rules.append({
                        "id": rule_id,
                        "text": results["documents"][i],
                        "metadata": meta,
                        # Champs aplatis pour faciliter la sérialisation
                        "rule_type": meta.get("rule_type", "constraint"),
                        "severity": meta.get("severity", "medium"),
                        "category": meta.get("category", "general"),
                        "active": meta.get("active", True),
                    })

            return rules
        except Exception as e:
            raise ChromaDBError(
                operation="get_all_rules",
                collection=settings.chroma_rules_collection,
                reason=str(e),
                original_error=e
            )

    def deactivate_rule(self, rule_id: str) -> None:
        """Désactive une règle sans la supprimer."""
        try:
            self.rules_collection.update(
                ids=[rule_id],
                metadatas=[{"active": False, "deactivated_at": datetime.now(timezone.utc).isoformat()}]
            )
        except Exception as e:
            raise ChromaDBError(
                operation="deactivate_rule",
                collection=settings.chroma_rules_collection,
                reason=str(e),
                original_error=e
            )

    # =========================================================================
    # DECISION HISTORY
    # =========================================================================

    def log_decision(
        self,
        decision_id: str,
        agent_type: str,
        action: str,
        reasoning: str,
        context: dict[str, Any],
        confidence: float
    ) -> str:
        """
        Enregistre une décision d'agent pour apprentissage futur.

        Args:
            decision_id: ID unique de la décision
            agent_type: Type d'agent ayant pris la décision
            action: Action décidée
            reasoning: Raisonnement derrière la décision
            context: Contexte de la décision
            confidence: Score de confiance

        Returns:
            ID de la décision enregistrée
        """
        # Créer un texte descriptif pour la recherche sémantique
        document = f"Agent: {agent_type}. Action: {action}. Reasoning: {reasoning}"

        # ChromaDB n'accepte pas None comme valeur de métadonnée (str/int/float/bool
        # uniquement). was_correct est omis ici et n'est ajouté qu'après feedback.
        metadata = {
            "agent_type": agent_type,
            "action": action,
            "confidence": confidence,
            "context_hash": hashlib.md5(str(context).encode()).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self.decisions_collection.add(
                ids=[decision_id],
                documents=[document],
                metadatas=[metadata]
            )
            return decision_id
        except Exception as e:
            raise ChromaDBError(
                operation="log_decision",
                collection=settings.chroma_decisions_collection,
                reason=str(e),
                original_error=e
            )

    def find_similar_decisions(
        self,
        query: str,
        agent_type: str | None = None,
        n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Recherche des décisions passées similaires.

        Utile pour:
        - Améliorer la confiance si décisions similaires étaient correctes
        - Apprendre des erreurs passées

        Args:
            query: Description de la situation actuelle
            agent_type: Filtrer par type d'agent
            n_results: Nombre de résultats

        Returns:
            Décisions similaires avec métadonnées
        """
        where_filter = {}
        if agent_type:
            where_filter["agent_type"] = agent_type

        try:
            if self.decisions_collection.count() == 0:
                return []

            actual_n = min(n_results, self.decisions_collection.count())

            results = self.decisions_collection.query(
                query_texts=[query],
                n_results=actual_n,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            decisions = []
            if results["ids"] and results["ids"][0]:
                for i, dec_id in enumerate(results["ids"][0]):
                    decisions.append({
                        "id": dec_id,
                        "description": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": 1 - (results["distances"][0][i] if results["distances"] else 0)
                    })

            return decisions
        except Exception as e:
            raise ChromaDBError(
                operation="find_similar_decisions",
                collection=settings.chroma_decisions_collection,
                reason=str(e),
                original_error=e
            )

    def get_decision_accuracy(self, agent_type: str | None = None) -> dict[str, Any]:
        """
        Calcule la précision historique des décisions.

        Args:
            agent_type: Filtrer par type d'agent

        Returns:
            Statistiques de précision
        """
        where_filter = {}
        if agent_type:
            where_filter["agent_type"] = agent_type

        try:
            results = self.decisions_collection.get(
                where=where_filter if where_filter else None,
                include=["metadatas"]
            )

            total = 0
            correct = 0
            incorrect = 0
            unknown = 0

            if results["metadatas"]:
                for meta in results["metadatas"]:
                    total += 1
                    was_correct = meta.get("was_correct")
                    if was_correct is True:
                        correct += 1
                    elif was_correct is False:
                        incorrect += 1
                    else:
                        unknown += 1

            accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else None

            return {
                "total_decisions": total,
                "correct": correct,
                "incorrect": incorrect,
                "unknown": unknown,
                "accuracy": accuracy,
                "agent_type": agent_type
            }
        except Exception as e:
            raise ChromaDBError(
                operation="get_decision_accuracy",
                collection=settings.chroma_decisions_collection,
                reason=str(e),
                original_error=e
            )

    # =========================================================================
    # USER FEEDBACK
    # =========================================================================

    def add_feedback(
        self,
        feedback_id: str,
        target_id: str,
        target_type: str,
        is_correct: bool,
        user_correction: str | None = None,
        comments: str | None = None
    ) -> str:
        """
        Enregistre un feedback utilisateur.

        Args:
            feedback_id: ID unique du feedback
            target_id: ID de l'élément concerné (issue, proposal, decision)
            target_type: Type d'élément
            is_correct: La décision était-elle correcte?
            user_correction: Correction suggérée
            comments: Commentaires additionnels

        Returns:
            ID du feedback
        """
        document = f"Feedback on {target_type} {target_id}. "
        if user_correction:
            document += f"Correction: {user_correction}. "
        if comments:
            document += f"Comments: {comments}"

        metadata = {
            "target_id": target_id,
            "target_type": target_type,
            "is_correct": is_correct,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self.feedback_collection.add(
                ids=[feedback_id],
                documents=[document],
                metadatas=[metadata]
            )

            # Mettre à jour la décision si c'en est une
            if target_type == "decision":
                self._update_decision_correctness(target_id, is_correct)

            return feedback_id
        except Exception as e:
            raise ChromaDBError(
                operation="add_feedback",
                collection=settings.chroma_feedback_collection,
                reason=str(e),
                original_error=e
            )

    def _update_decision_correctness(
        self,
        decision_id: str,
        was_correct: bool
    ) -> None:
        """Met à jour le statut de correction d'une décision."""
        try:
            self.decisions_collection.update(
                ids=[decision_id],
                metadatas=[{
                    "was_correct": was_correct,
                    "feedback_received_at": datetime.now(timezone.utc).isoformat()
                }]
            )
        except Exception:
            pass  # Ignore si la décision n'existe pas

    def get_feedback_for_target(self, target_id: str) -> list[dict[str, Any]]:
        """Récupère tous les feedbacks pour un élément."""
        try:
            results = self.feedback_collection.get(
                where={"target_id": target_id},
                include=["documents", "metadatas"]
            )

            feedbacks = []
            if results["ids"]:
                for i, fb_id in enumerate(results["ids"]):
                    feedbacks.append({
                        "id": fb_id,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })

            return feedbacks
        except Exception as e:
            raise ChromaDBError(
                operation="get_feedback_for_target",
                collection=settings.chroma_feedback_collection,
                reason=str(e),
                original_error=e
            )

    def search_similar_feedback(
        self,
        query: str,
        n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Recherche des feedbacks similaires.

        Utile pour apprendre des corrections utilisateur sur cas similaires.

        Args:
            query: Description du cas actuel
            n_results: Nombre de résultats

        Returns:
            Feedbacks similaires
        """
        try:
            if self.feedback_collection.count() == 0:
                return []

            actual_n = min(n_results, self.feedback_collection.count())

            results = self.feedback_collection.query(
                query_texts=[query],
                n_results=actual_n,
                include=["documents", "metadatas", "distances"]
            )

            feedbacks = []
            if results["ids"] and results["ids"][0]:
                for i, fb_id in enumerate(results["ids"][0]):
                    feedbacks.append({
                        "id": fb_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": 1 - (results["distances"][0][i] if results["distances"] else 0)
                    })

            return feedbacks
        except Exception as e:
            raise ChromaDBError(
                operation="search_similar_feedback",
                collection=settings.chroma_feedback_collection,
                reason=str(e),
                original_error=e
            )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Retourne des statistiques sur les collections."""
        return {
            "rules_count": self.rules_collection.count(),
            "decisions_count": self.decisions_collection.count(),
            "feedback_count": self.feedback_collection.count(),
            "persist_path": str(self.persist_path),
        }

    def reset(self) -> None:
        """Réinitialise toutes les collections (attention: perte de données)."""
        self.client.delete_collection(settings.chroma_rules_collection)
        self.client.delete_collection(settings.chroma_decisions_collection)
        self.client.delete_collection(settings.chroma_feedback_collection)
        self._init_collections()


# Instance globale pour faciliter l'import
def get_chroma_store() -> ChromaStore:
    """Retourne l'instance singleton du ChromaStore."""
    return ChromaStore()
