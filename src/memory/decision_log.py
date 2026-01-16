"""
Gestionnaire de log des décisions.

Ce module fournit une interface de haut niveau pour enregistrer
et récupérer les décisions des agents. Il utilise ChromaStore
pour la persistance mais ajoute une couche de logique métier.

L'historique des décisions permet:
- L'apprentissage contextuel (cas similaires passés)
- L'explicabilité (pourquoi cette décision)
- L'amélioration continue via les feedbacks
"""

import uuid
from datetime import datetime
from typing import Any

from src.core.models import AgentDecision, AgentType
from src.memory.chroma_store import ChromaStore, get_chroma_store


class DecisionLogger:
    """
    Logger de décisions pour les agents.

    Encapsule la logique de logging et de récupération des décisions
    avec une interface type-safe utilisant les modèles Pydantic.

    Attributes:
        store: Instance ChromaStore
    """

    def __init__(self, store: ChromaStore | None = None) -> None:
        """
        Initialise le logger.

        Args:
            store: Instance ChromaStore (défaut: singleton global)
        """
        self.store = store or get_chroma_store()

    def log(
        self,
        agent_type: AgentType,
        session_id: str,
        action: str,
        reasoning: str,
        input_summary: str,
        output_summary: str,
        confidence: float,
        processing_time_ms: int,
        metadata: dict[str, Any] | None = None
    ) -> AgentDecision:
        """
        Enregistre une décision d'agent.

        Args:
            agent_type: Type de l'agent
            session_id: ID de la session
            action: Action décidée
            reasoning: Raisonnement
            input_summary: Résumé des entrées
            output_summary: Résumé des sorties
            confidence: Score de confiance
            processing_time_ms: Temps de traitement
            metadata: Métadonnées additionnelles

        Returns:
            AgentDecision créée et enregistrée
        """
        decision_id = f"dec_{uuid.uuid4().hex[:12]}"

        decision = AgentDecision(
            decision_id=decision_id,
            agent_type=agent_type,
            session_id=session_id,
            action=action,
            reasoning=reasoning,
            input_summary=input_summary,
            output_summary=output_summary,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            created_at=datetime.utcnow()
        )

        # Contexte pour la recherche
        context = {
            "session_id": session_id,
            "input_summary": input_summary,
            "output_summary": output_summary,
            **(metadata or {})
        }

        # Enregistrer dans ChromaDB
        self.store.log_decision(
            decision_id=decision_id,
            agent_type=agent_type.value,
            action=action,
            reasoning=reasoning,
            context=context,
            confidence=confidence
        )

        return decision

    def find_similar(
        self,
        current_situation: str,
        agent_type: AgentType | None = None,
        n_results: int = 5,
        min_similarity: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Trouve des décisions passées similaires.

        Args:
            current_situation: Description de la situation actuelle
            agent_type: Filtrer par type d'agent
            n_results: Nombre max de résultats
            min_similarity: Seuil de similarité minimum

        Returns:
            Liste de décisions similaires avec métadonnées
        """
        decisions = self.store.find_similar_decisions(
            query=current_situation,
            agent_type=agent_type.value if agent_type else None,
            n_results=n_results
        )

        # Filtrer par similarité minimum
        return [
            d for d in decisions
            if d.get("similarity", 0) >= min_similarity
        ]

    def get_historical_accuracy(
        self,
        agent_type: AgentType | None = None
    ) -> float | None:
        """
        Calcule la précision historique d'un type d'agent.

        Args:
            agent_type: Type d'agent (None = tous)

        Returns:
            Taux de précision (0-1) ou None si pas assez de données
        """
        stats = self.store.get_decision_accuracy(
            agent_type=agent_type.value if agent_type else None
        )
        return stats.get("accuracy")

    def get_decisions_for_session(
        self,
        session_id: str
    ) -> list[dict[str, Any]]:
        """
        Récupère toutes les décisions d'une session.

        Args:
            session_id: ID de la session

        Returns:
            Liste des décisions
        """
        # Recherche par session_id dans le contexte
        # Note: ChromaDB ne supporte pas la recherche par métadonnées imbriquées,
        # donc on fait une recherche sémantique avec le session_id
        return self.store.find_similar_decisions(
            query=f"session {session_id}",
            n_results=100  # Limite haute pour récupérer toutes les décisions
        )

    def calculate_confidence_adjustment(
        self,
        similar_decisions: list[dict[str, Any]]
    ) -> float:
        """
        Calcule un ajustement de confiance basé sur les décisions similaires.

        Si des décisions similaires ont été confirmées comme correctes,
        on augmente la confiance. Si incorrectes, on la diminue.

        Args:
            similar_decisions: Décisions similaires trouvées

        Returns:
            Facteur d'ajustement (-0.2 à +0.2)
        """
        if not similar_decisions:
            return 0.0

        # Compter les décisions avec feedback
        correct_count = 0
        incorrect_count = 0
        total_with_feedback = 0

        for decision in similar_decisions:
            was_correct = decision.get("metadata", {}).get("was_correct")
            similarity = decision.get("similarity", 0.5)

            if was_correct is True:
                correct_count += similarity  # Pondérer par similarité
                total_with_feedback += 1
            elif was_correct is False:
                incorrect_count += similarity
                total_with_feedback += 1

        if total_with_feedback == 0:
            return 0.0

        # Calculer le ratio
        total_weighted = correct_count + incorrect_count
        if total_weighted == 0:
            return 0.0

        accuracy_ratio = correct_count / total_weighted

        # Convertir en ajustement (-0.2 à +0.2)
        # 0.5 ratio = 0 ajustement (neutre)
        # 1.0 ratio = +0.2 ajustement
        # 0.0 ratio = -0.2 ajustement
        adjustment = (accuracy_ratio - 0.5) * 0.4

        return round(adjustment, 3)


# Instance globale
def get_decision_logger() -> DecisionLogger:
    """Retourne une instance du DecisionLogger."""
    return DecisionLogger()
