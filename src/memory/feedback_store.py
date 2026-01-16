"""
Gestionnaire de feedbacks utilisateur.

Ce module gère l'enregistrement et l'exploitation des feedbacks
utilisateur pour améliorer les décisions futures du système.

Le feedback loop permet:
- Correction des erreurs du système
- Apprentissage des préférences utilisateur
- Amélioration continue de la confiance
"""

import uuid
from datetime import datetime
from typing import Any

from src.core.models import FeedbackRequest, FeedbackResponse
from src.memory.chroma_store import ChromaStore, get_chroma_store


class FeedbackStore:
    """
    Store pour les feedbacks utilisateur.

    Gère l'enregistrement des feedbacks et leur exploitation
    pour améliorer les décisions futures.

    Attributes:
        store: Instance ChromaStore
    """

    def __init__(self, store: ChromaStore | None = None) -> None:
        """
        Initialise le store.

        Args:
            store: Instance ChromaStore (défaut: singleton global)
        """
        self.store = store or get_chroma_store()

    def record_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """
        Enregistre un feedback utilisateur.

        Args:
            request: Requête de feedback

        Returns:
            Réponse confirmant l'enregistrement
        """
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"

        # Enregistrer dans ChromaDB
        self.store.add_feedback(
            feedback_id=feedback_id,
            target_id=request.target_id,
            target_type=request.target_type,
            is_correct=request.is_correct,
            user_correction=request.user_correction,
            comments=request.comments
        )

        # Déterminer l'impact du feedback
        impact = self._determine_impact(request)

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="recorded",
            message="Feedback enregistré avec succès",
            impact=impact
        )

    def _determine_impact(self, request: FeedbackRequest) -> str:
        """Détermine comment le feedback sera utilisé."""
        if request.target_type == "decision":
            if request.is_correct:
                return (
                    "Ce feedback positif augmentera la confiance du système "
                    "pour des décisions similaires futures."
                )
            else:
                return (
                    "Ce feedback négatif ajustera le comportement du système. "
                    "Les décisions similaires seront analysées avec plus de prudence."
                )
        elif request.target_type == "proposal":
            if request.is_correct:
                return "Cette correction sera proposée en priorité pour des cas similaires."
            else:
                return (
                    "Le système évitera de proposer cette correction "
                    "pour des situations similaires."
                )
        elif request.target_type == "issue":
            if request.is_correct:
                return "Ce type de problème sera détecté avec plus de confiance."
            else:
                return (
                    "Le seuil de détection sera ajusté pour réduire "
                    "les faux positifs similaires."
                )
        else:
            return "Ce feedback sera pris en compte pour améliorer le système."

    def get_feedback_stats(self) -> dict[str, Any]:
        """
        Calcule des statistiques sur les feedbacks reçus.

        Returns:
            Statistiques globales
        """
        # Récupérer tous les feedbacks
        # Note: Cette implémentation est simplifiée, en production
        # on utiliserait une requête plus efficace
        all_feedback = []

        for target_type in ["decision", "proposal", "issue"]:
            results = self.store.feedback_collection.get(
                where={"target_type": target_type},
                include=["metadatas"]
            )
            if results["metadatas"]:
                all_feedback.extend(results["metadatas"])

        total = len(all_feedback)
        positive = sum(1 for f in all_feedback if f.get("is_correct") is True)
        negative = sum(1 for f in all_feedback if f.get("is_correct") is False)

        by_type = {}
        for target_type in ["decision", "proposal", "issue"]:
            type_feedback = [f for f in all_feedback if f.get("target_type") == target_type]
            by_type[target_type] = {
                "total": len(type_feedback),
                "positive": sum(1 for f in type_feedback if f.get("is_correct") is True),
                "negative": sum(1 for f in type_feedback if f.get("is_correct") is False),
            }

        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": positive / total if total > 0 else None,
            "by_type": by_type
        }

    def find_relevant_feedback(
        self,
        context: str,
        target_type: str | None = None,
        n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Trouve des feedbacks pertinents pour le contexte actuel.

        Args:
            context: Description du contexte actuel
            target_type: Type de cible (optionnel)
            n_results: Nombre de résultats

        Returns:
            Feedbacks pertinents
        """
        feedbacks = self.store.search_similar_feedback(
            query=context,
            n_results=n_results
        )

        if target_type:
            feedbacks = [
                f for f in feedbacks
                if f.get("metadata", {}).get("target_type") == target_type
            ]

        return feedbacks

    def learn_from_feedback(
        self,
        context: str,
        target_type: str
    ) -> dict[str, Any]:
        """
        Extrait des insights des feedbacks pour améliorer les décisions.

        Args:
            context: Contexte de la décision actuelle
            target_type: Type de cible

        Returns:
            Insights et recommandations
        """
        relevant = self.find_relevant_feedback(
            context=context,
            target_type=target_type,
            n_results=10
        )

        if not relevant:
            return {
                "has_relevant_feedback": False,
                "recommendation": None,
                "confidence_adjustment": 0.0
            }

        # Analyser les feedbacks
        positive_count = sum(
            1 for f in relevant
            if f.get("metadata", {}).get("is_correct") is True
        )
        negative_count = sum(
            1 for f in relevant
            if f.get("metadata", {}).get("is_correct") is False
        )

        total = positive_count + negative_count
        if total == 0:
            return {
                "has_relevant_feedback": True,
                "feedback_count": len(relevant),
                "recommendation": "Pas assez de feedbacks avec résultat pour ajuster",
                "confidence_adjustment": 0.0
            }

        success_rate = positive_count / total

        # Calculer l'ajustement de confiance
        # success_rate de 0.5 = ajustement neutre (0)
        # success_rate de 1.0 = ajustement positif (+0.15)
        # success_rate de 0.0 = ajustement négatif (-0.15)
        confidence_adjustment = (success_rate - 0.5) * 0.3

        # Générer une recommandation
        if success_rate >= 0.8:
            recommendation = "Historique très positif - haute confiance recommandée"
        elif success_rate >= 0.6:
            recommendation = "Historique généralement positif - confiance normale"
        elif success_rate >= 0.4:
            recommendation = "Historique mitigé - vérification recommandée"
        else:
            recommendation = "Historique négatif - prudence requise, envisager escalade"

        # Extraire les corrections suggérées par les utilisateurs
        user_corrections = [
            f.get("content", "")
            for f in relevant
            if f.get("metadata", {}).get("is_correct") is False
            and "Correction:" in f.get("content", "")
        ]

        return {
            "has_relevant_feedback": True,
            "feedback_count": len(relevant),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "success_rate": round(success_rate, 3),
            "confidence_adjustment": round(confidence_adjustment, 3),
            "recommendation": recommendation,
            "past_corrections": user_corrections[:3]  # Top 3 corrections
        }


# Instance globale
def get_feedback_store() -> FeedbackStore:
    """Retourne une instance du FeedbackStore."""
    return FeedbackStore()
