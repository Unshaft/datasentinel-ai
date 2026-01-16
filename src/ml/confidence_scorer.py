"""
Scoring de confiance pour les décisions des agents.

Ce module calcule un score de confiance pour chaque décision prise
par le système. La confiance aide à:
- Décider si une escalade humaine est nécessaire
- Prioriser les corrections proposées
- Expliquer le niveau de certitude des analyses

Le score combine plusieurs facteurs:
- Quantité de données disponibles
- Cohérence des signaux (plusieurs indicateurs convergent)
- Historique de décisions similaires
- Qualité des données d'entrée
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ConfidenceLevel(str, Enum):
    """Niveau de confiance qualitatif."""

    VERY_LOW = "very_low"     # 0.0 - 0.3: Escalade obligatoire
    LOW = "low"               # 0.3 - 0.5: Escalade recommandée
    MEDIUM = "medium"         # 0.5 - 0.7: Vérification suggérée
    HIGH = "high"             # 0.7 - 0.9: Confiance acceptable
    VERY_HIGH = "very_high"   # 0.9 - 1.0: Haute confiance


@dataclass
class ConfidenceScore:
    """Score de confiance avec décomposition."""

    overall_score: float              # Score global 0-1
    level: ConfidenceLevel           # Niveau qualitatif
    factors: dict[str, float]        # Scores par facteur
    weights: dict[str, float]        # Poids de chaque facteur
    explanation: str                  # Explication en langage naturel
    needs_escalation: bool           # Doit être escaladé?
    escalation_reason: str | None    # Raison de l'escalade

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "overall_score": round(self.overall_score, 3),
            "level": self.level.value,
            "factors": {k: round(v, 3) for k, v in self.factors.items()},
            "weights": self.weights,
            "explanation": self.explanation,
            "needs_escalation": self.needs_escalation,
            "escalation_reason": self.escalation_reason,
        }


class ConfidenceScorer:
    """
    Calcule les scores de confiance pour les décisions du système.

    Le scoring est basé sur une combinaison pondérée de facteurs
    qui reflètent la fiabilité de la décision.

    Attributes:
        escalation_threshold: Seuil en dessous duquel escalader
        weights: Poids par défaut des facteurs
    """

    # Poids par défaut pour chaque facteur
    DEFAULT_WEIGHTS = {
        "data_quality": 0.25,      # Qualité des données d'entrée
        "sample_size": 0.20,       # Taille de l'échantillon
        "signal_consistency": 0.25, # Cohérence des signaux
        "historical_accuracy": 0.15, # Précision historique
        "rule_coverage": 0.15,     # Couverture par les règles
    }

    def __init__(
        self,
        escalation_threshold: float = 0.5,
        weights: dict[str, float] | None = None
    ) -> None:
        """
        Initialise le scorer.

        Args:
            escalation_threshold: Seuil pour escalade (défaut: 0.5)
            weights: Poids personnalisés (défaut: DEFAULT_WEIGHTS)
        """
        self.escalation_threshold = escalation_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normaliser les poids pour qu'ils somment à 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate(
        self,
        data_quality_score: float = 1.0,
        sample_size: int = 100,
        signal_scores: list[float] | None = None,
        historical_accuracy: float | None = None,
        rule_coverage: float = 1.0,
        context: dict[str, Any] | None = None
    ) -> ConfidenceScore:
        """
        Calcule le score de confiance global.

        Args:
            data_quality_score: Score de qualité des données (0-1)
            sample_size: Nombre d'échantillons analysés
            signal_scores: Scores de différents signaux/indicateurs
            historical_accuracy: Précision historique sur décisions similaires
            rule_coverage: Proportion de règles métier vérifiées
            context: Contexte additionnel pour l'explication

        Returns:
            Score de confiance complet
        """
        factors = {}

        # Facteur 1: Qualité des données
        factors["data_quality"] = self._clamp(data_quality_score)

        # Facteur 2: Taille de l'échantillon (log scale)
        factors["sample_size"] = self._sample_size_score(sample_size)

        # Facteur 3: Cohérence des signaux
        if signal_scores and len(signal_scores) > 0:
            factors["signal_consistency"] = self._signal_consistency_score(signal_scores)
        else:
            factors["signal_consistency"] = 0.5  # Neutre si pas de signaux

        # Facteur 4: Précision historique
        if historical_accuracy is not None:
            factors["historical_accuracy"] = self._clamp(historical_accuracy)
        else:
            factors["historical_accuracy"] = 0.5  # Neutre si pas d'historique

        # Facteur 5: Couverture des règles
        factors["rule_coverage"] = self._clamp(rule_coverage)

        # Calcul du score global pondéré
        overall_score = sum(
            factors[factor] * self.weights[factor]
            for factor in factors
        )

        # Déterminer le niveau qualitatif
        level = self._score_to_level(overall_score)

        # Vérifier si escalade nécessaire
        needs_escalation = overall_score < self.escalation_threshold
        escalation_reason = self._get_escalation_reason(
            overall_score, factors, needs_escalation
        )

        # Générer l'explication
        explanation = self._generate_explanation(
            overall_score, level, factors, context
        )

        return ConfidenceScore(
            overall_score=overall_score,
            level=level,
            factors=factors,
            weights=self.weights,
            explanation=explanation,
            needs_escalation=needs_escalation,
            escalation_reason=escalation_reason
        )

    def calculate_for_issue(
        self,
        affected_percentage: float,
        detection_strength: float,
        sample_size: int,
        similar_issues_accuracy: float | None = None
    ) -> ConfidenceScore:
        """
        Calcule la confiance pour une détection de problème.

        Args:
            affected_percentage: % de données affectées (0-100)
            detection_strength: Force du signal de détection (0-1)
            sample_size: Taille de l'échantillon
            similar_issues_accuracy: Précision sur issues similaires

        Returns:
            Score de confiance
        """
        # Plus le problème est répandu, plus on est confiant dans sa détection
        # (sauf si très peu de cas -> possiblement faux positifs)
        if affected_percentage < 1:
            data_quality = 0.4  # Très peu de cas, méfiance
        elif affected_percentage < 5:
            data_quality = 0.6
        elif affected_percentage < 20:
            data_quality = 0.8
        else:
            data_quality = 0.9  # Problème systématique, confiant

        return self.calculate(
            data_quality_score=data_quality,
            sample_size=sample_size,
            signal_scores=[detection_strength],
            historical_accuracy=similar_issues_accuracy,
            context={
                "type": "issue_detection",
                "affected_percentage": affected_percentage
            }
        )

    def calculate_for_correction(
        self,
        issue_confidence: float,
        correction_impact: float,
        rules_validated: int,
        rules_total: int,
        similar_corrections_success: float | None = None
    ) -> ConfidenceScore:
        """
        Calcule la confiance pour une proposition de correction.

        Args:
            issue_confidence: Confiance dans le problème détecté
            correction_impact: Impact estimé de la correction (0-1)
            rules_validated: Nombre de règles validées
            rules_total: Nombre total de règles
            similar_corrections_success: Taux de succès de corrections similaires

        Returns:
            Score de confiance
        """
        rule_coverage = rules_validated / rules_total if rules_total > 0 else 0.5

        # La confiance dans la correction dépend de la confiance dans le problème
        adjusted_quality = issue_confidence * 0.8 + correction_impact * 0.2

        return self.calculate(
            data_quality_score=adjusted_quality,
            sample_size=100,  # Fixe pour corrections
            signal_scores=[issue_confidence, correction_impact],
            historical_accuracy=similar_corrections_success,
            rule_coverage=rule_coverage,
            context={
                "type": "correction_proposal",
                "issue_confidence": issue_confidence,
                "rules_validated": rules_validated,
                "rules_total": rules_total
            }
        )

    def _sample_size_score(self, n: int) -> float:
        """
        Convertit la taille d'échantillon en score.

        Utilise une échelle logarithmique car l'apport marginal
        diminue avec la taille.

        Args:
            n: Nombre d'échantillons

        Returns:
            Score entre 0 et 1
        """
        if n <= 0:
            return 0.0
        elif n < 10:
            return 0.2
        elif n < 30:
            return 0.4
        elif n < 100:
            return 0.6
        elif n < 1000:
            return 0.8
        else:
            return 0.95

    def _signal_consistency_score(self, scores: list[float]) -> float:
        """
        Calcule la cohérence entre plusieurs signaux.

        Des signaux similaires (tous hauts ou tous bas) indiquent
        une cohérence, donc une confiance plus élevée.

        Args:
            scores: Liste de scores de signaux (0-1)

        Returns:
            Score de cohérence (0-1)
        """
        if len(scores) < 2:
            return scores[0] if scores else 0.5

        # Écart-type normalisé (inverse = cohérence)
        std = np.std(scores)
        mean = np.mean(scores)

        # Score de cohérence: faible variance = haute cohérence
        consistency = 1.0 - min(std * 2, 1.0)

        # Pondérer par la moyenne (signaux forts et cohérents = meilleur)
        return consistency * 0.6 + mean * 0.4

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convertit un score numérique en niveau qualitatif."""
        if score < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.5:
            return ConfidenceLevel.LOW
        elif score < 0.7:
            return ConfidenceLevel.MEDIUM
        elif score < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _get_escalation_reason(
        self,
        overall_score: float,
        factors: dict[str, float],
        needs_escalation: bool
    ) -> str | None:
        """Détermine la raison de l'escalade si nécessaire."""
        if not needs_escalation:
            return None

        # Identifier le facteur le plus faible
        weakest_factor = min(factors, key=factors.get)
        weakest_score = factors[weakest_factor]

        reasons = {
            "data_quality": "qualité des données insuffisante",
            "sample_size": "échantillon trop petit pour une décision fiable",
            "signal_consistency": "signaux contradictoires",
            "historical_accuracy": "précision historique faible sur cas similaires",
            "rule_coverage": "règles métier insuffisamment vérifiées",
        }

        return f"Escalade requise: {reasons.get(weakest_factor, 'confiance globale insuffisante')} (score: {weakest_score:.2f})"

    def _generate_explanation(
        self,
        overall_score: float,
        level: ConfidenceLevel,
        factors: dict[str, float],
        context: dict[str, Any] | None
    ) -> str:
        """Génère une explication en langage naturel."""
        # Identifier les points forts et faibles
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        strengths = [f for f, s in sorted_factors if s >= 0.7]
        weaknesses = [f for f, s in sorted_factors if s < 0.5]

        factor_names = {
            "data_quality": "qualité des données",
            "sample_size": "taille de l'échantillon",
            "signal_consistency": "cohérence des signaux",
            "historical_accuracy": "historique",
            "rule_coverage": "couverture des règles",
        }

        explanation_parts = [
            f"Confiance {level.value} ({overall_score:.0%})."
        ]

        if strengths:
            strength_names = [factor_names.get(s, s) for s in strengths[:2]]
            explanation_parts.append(
                f"Points forts: {', '.join(strength_names)}."
            )

        if weaknesses:
            weakness_names = [factor_names.get(w, w) for w in weaknesses[:2]]
            explanation_parts.append(
                f"Points faibles: {', '.join(weakness_names)}."
            )

        return " ".join(explanation_parts)

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Limite une valeur entre min et max."""
        return max(min_val, min(max_val, value))

    def adjust_weights(self, new_weights: dict[str, float]) -> None:
        """
        Ajuste les poids des facteurs.

        Permet de personnaliser l'importance relative des facteurs
        selon le contexte d'utilisation.

        Args:
            new_weights: Nouveaux poids (seront normalisés)
        """
        self.weights.update(new_weights)
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
