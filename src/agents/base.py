"""
Classe de base pour tous les agents DataSentinel.

Ce module définit l'interface commune et les fonctionnalités
partagées par tous les agents spécialisés.

Design Pattern: Template Method
- La classe de base définit le squelette de l'algorithme
- Les sous-classes implémentent les étapes spécifiques
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from src.core.config import settings
from src.core.models import AgentContext, AgentDecision, AgentType
from src.memory.decision_log import DecisionLogger, get_decision_logger
from src.ml.confidence_scorer import ConfidenceScore, ConfidenceScorer


class BaseAgent(ABC):
    """
    Classe abstraite de base pour tous les agents.

    Fournit:
    - Interface commune pour l'exécution
    - Gestion du LLM (Claude)
    - Logging des décisions
    - Calcul de confiance

    Attributes:
        agent_type: Type de l'agent (enum)
        llm: Instance du modèle de langage
        tools: Outils disponibles pour cet agent
        decision_logger: Logger pour les décisions
        confidence_scorer: Calculateur de score de confiance
    """

    def __init__(
        self,
        agent_type: AgentType,
        tools: list[BaseTool] | None = None,
        temperature: float | None = None,
        decision_logger: DecisionLogger | None = None
    ) -> None:
        """
        Initialise l'agent.

        Args:
            agent_type: Type d'agent (Profiler, Quality, etc.)
            tools: Liste des outils LangChain disponibles
            temperature: Température du LLM (défaut: config)
            decision_logger: Logger de décisions (défaut: global)
        """
        self.agent_type = agent_type
        self.tools = tools or []
        self.decision_logger = decision_logger or get_decision_logger()
        self.confidence_scorer = ConfidenceScorer(
            escalation_threshold=settings.confidence_threshold
        )

        # Initialisation du LLM
        self.llm = ChatAnthropic(
            model=settings.claude_model,
            temperature=temperature or settings.claude_temperature,
            max_tokens=settings.claude_max_tokens,
            anthropic_api_key=settings.anthropic_api_key,
        )

        # Si des outils sont disponibles, les binder au LLM
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Prompt système définissant le rôle et le comportement de l'agent.

        Doit être implémenté par chaque agent spécialisé.
        """
        pass

    @abstractmethod
    def execute(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        **kwargs: Any
    ) -> AgentContext:
        """
        Exécute la tâche principale de l'agent.

        Args:
            context: Contexte partagé de la session
            df: DataFrame à analyser
            **kwargs: Arguments additionnels spécifiques

        Returns:
            Contexte mis à jour avec les résultats
        """
        pass

    def _invoke_llm(
        self,
        user_message: str,
        include_tools: bool = True
    ) -> Any:
        """
        Invoque le LLM avec le message utilisateur.

        Args:
            user_message: Message/prompt à envoyer
            include_tools: Inclure les outils dans l'appel

        Returns:
            Réponse du LLM
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]

        if include_tools and self.tools:
            return self.llm_with_tools.invoke(messages)
        else:
            return self.llm.invoke(messages)

    def _log_decision(
        self,
        context: AgentContext,
        action: str,
        reasoning: str,
        input_summary: str,
        output_summary: str,
        confidence: float,
        processing_time_ms: int
    ) -> AgentDecision:
        """
        Enregistre une décision prise par l'agent.

        Args:
            context: Contexte de la session
            action: Action décidée
            reasoning: Raisonnement
            input_summary: Résumé des entrées
            output_summary: Résumé des sorties
            confidence: Score de confiance
            processing_time_ms: Temps de traitement

        Returns:
            Décision enregistrée
        """
        return self.decision_logger.log(
            agent_type=self.agent_type,
            session_id=context.session_id,
            action=action,
            reasoning=reasoning,
            input_summary=input_summary,
            output_summary=output_summary,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )

    def _calculate_confidence(
        self,
        data_quality: float = 1.0,
        sample_size: int = 100,
        signal_scores: list[float] | None = None,
        rule_coverage: float = 1.0
    ) -> ConfidenceScore:
        """
        Calcule le score de confiance pour une décision.

        Args:
            data_quality: Score de qualité des données (0-1)
            sample_size: Taille de l'échantillon
            signal_scores: Scores de différents signaux
            rule_coverage: Couverture des règles vérifiées

        Returns:
            Score de confiance complet
        """
        # Récupérer la précision historique de cet agent
        historical_accuracy = self.decision_logger.get_historical_accuracy(
            agent_type=self.agent_type
        )

        return self.confidence_scorer.calculate(
            data_quality_score=data_quality,
            sample_size=sample_size,
            signal_scores=signal_scores,
            historical_accuracy=historical_accuracy,
            rule_coverage=rule_coverage
        )

    def _get_similar_decisions(
        self,
        current_situation: str,
        n_results: int = 5
    ) -> list[dict[str, Any]]:
        """
        Récupère des décisions passées similaires.

        Utile pour ajuster la confiance basée sur l'historique.

        Args:
            current_situation: Description de la situation actuelle
            n_results: Nombre de résultats

        Returns:
            Décisions similaires
        """
        return self.decision_logger.find_similar(
            current_situation=current_situation,
            agent_type=self.agent_type,
            n_results=n_results
        )

    def _format_tools_output(self, tool_outputs: list[dict]) -> str:
        """
        Formate les sorties des outils pour le LLM.

        Args:
            tool_outputs: Liste des résultats d'outils

        Returns:
            Texte formaté
        """
        formatted = []
        for output in tool_outputs:
            tool_name = output.get("tool", "unknown")
            result = output.get("result", "")
            formatted.append(f"=== {tool_name} ===\n{result}")
        return "\n\n".join(formatted)

    @staticmethod
    def generate_id(prefix: str = "agent") -> str:
        """Génère un ID unique."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def measure_time(func):
        """Décorateur pour mesurer le temps d'exécution."""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = int((time.time() - start) * 1000)
            return result, elapsed_ms
        return wrapper


class AgentResult:
    """
    Résultat standardisé d'exécution d'un agent.

    Encapsule le contexte mis à jour, les métriques
    et les informations de debug.
    """

    def __init__(
        self,
        context: AgentContext,
        success: bool,
        confidence: ConfidenceScore,
        processing_time_ms: int,
        decision: AgentDecision | None = None,
        error: str | None = None
    ) -> None:
        self.context = context
        self.success = success
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.decision = decision
        self.error = error

    @property
    def needs_escalation(self) -> bool:
        """Vérifie si le résultat nécessite une escalade."""
        return self.confidence.needs_escalation

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "success": self.success,
            "confidence": self.confidence.to_dict(),
            "processing_time_ms": self.processing_time_ms,
            "needs_escalation": self.needs_escalation,
            "error": self.error
        }
