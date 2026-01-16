"""
Exceptions personnalisées pour DataSentinel AI.

Hiérarchie d'exceptions pour une gestion d'erreurs cohérente
et des messages d'erreur informatifs à travers le système.
"""

from typing import Any


class DataSentinelError(Exception):
    """
    Exception de base pour toutes les erreurs DataSentinel.

    Toutes les exceptions personnalisées héritent de cette classe,
    permettant de capturer toutes les erreurs du système avec
    un seul except.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ) -> None:
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convertit l'exception en dictionnaire pour l'API."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# DATA ERRORS - Erreurs liées aux données
# =============================================================================


class DataError(DataSentinelError):
    """Erreur de base liée aux données."""
    pass


class DataLoadError(DataError):
    """Erreur lors du chargement des données."""

    def __init__(
        self,
        source: str,
        reason: str,
        original_error: Exception | None = None
    ) -> None:
        message = f"Impossible de charger les données depuis '{source}': {reason}"
        super().__init__(
            message=message,
            details={"source": source, "reason": reason},
            original_error=original_error
        )


class DataValidationError(DataError):
    """Erreur de validation des données d'entrée."""

    def __init__(
        self,
        field: str,
        expected: str,
        actual: str
    ) -> None:
        message = f"Validation échouée pour '{field}': attendu {expected}, reçu {actual}"
        super().__init__(
            message=message,
            details={"field": field, "expected": expected, "actual": actual}
        )


class EmptyDataError(DataError):
    """Erreur quand les données sont vides."""

    def __init__(self, source: str) -> None:
        message = f"Le dataset '{source}' est vide"
        super().__init__(message=message, details={"source": source})


class DataSizeExceededError(DataError):
    """Erreur quand les données dépassent la limite."""

    def __init__(
        self,
        actual_size: int,
        max_size: int,
        unit: str = "bytes"
    ) -> None:
        message = f"Taille des données ({actual_size} {unit}) dépasse la limite ({max_size} {unit})"
        super().__init__(
            message=message,
            details={
                "actual_size": actual_size,
                "max_size": max_size,
                "unit": unit
            }
        )


# =============================================================================
# AGENT ERRORS - Erreurs liées aux agents
# =============================================================================


class AgentError(DataSentinelError):
    """Erreur de base liée aux agents."""
    pass


class AgentExecutionError(AgentError):
    """Erreur lors de l'exécution d'un agent."""

    def __init__(
        self,
        agent_name: str,
        step: str,
        reason: str,
        original_error: Exception | None = None
    ) -> None:
        message = f"Agent '{agent_name}' a échoué à l'étape '{step}': {reason}"
        super().__init__(
            message=message,
            details={"agent": agent_name, "step": step, "reason": reason},
            original_error=original_error
        )


class AgentTimeoutError(AgentError):
    """Erreur de timeout d'un agent."""

    def __init__(self, agent_name: str, timeout_seconds: int) -> None:
        message = f"Agent '{agent_name}' a dépassé le timeout de {timeout_seconds}s"
        super().__init__(
            message=message,
            details={"agent": agent_name, "timeout": timeout_seconds}
        )


class AgentMaxIterationsError(AgentError):
    """Erreur quand un agent atteint le maximum d'itérations."""

    def __init__(self, agent_name: str, max_iterations: int) -> None:
        message = f"Agent '{agent_name}' a atteint le maximum de {max_iterations} itérations"
        super().__init__(
            message=message,
            details={"agent": agent_name, "max_iterations": max_iterations}
        )


class OrchestratorError(AgentError):
    """Erreur spécifique à l'orchestrateur."""

    def __init__(self, reason: str, context: dict[str, Any] | None = None) -> None:
        message = f"Erreur d'orchestration: {reason}"
        super().__init__(message=message, details=context or {})


# =============================================================================
# ML ERRORS - Erreurs liées au Machine Learning
# =============================================================================


class MLError(DataSentinelError):
    """Erreur de base liée au ML."""
    pass


class ModelNotFittedError(MLError):
    """Erreur quand un modèle n'est pas entraîné."""

    def __init__(self, model_name: str) -> None:
        message = f"Le modèle '{model_name}' doit être entraîné avant utilisation"
        super().__init__(message=message, details={"model": model_name})


class InsufficientDataError(MLError):
    """Erreur quand il n'y a pas assez de données pour le ML."""

    def __init__(
        self,
        model_name: str,
        required: int,
        actual: int
    ) -> None:
        message = f"'{model_name}' nécessite au moins {required} échantillons, {actual} fournis"
        super().__init__(
            message=message,
            details={"model": model_name, "required": required, "actual": actual}
        )


class DriftDetectionError(MLError):
    """Erreur lors de la détection de drift."""

    def __init__(self, column: str, reason: str) -> None:
        message = f"Impossible de détecter le drift pour '{column}': {reason}"
        super().__init__(message=message, details={"column": column, "reason": reason})


# =============================================================================
# MEMORY ERRORS - Erreurs liées à la mémoire (ChromaDB)
# =============================================================================


class MemoryError(DataSentinelError):
    """Erreur de base liée à la mémoire."""
    pass


class ChromaDBError(MemoryError):
    """Erreur liée à ChromaDB."""

    def __init__(
        self,
        operation: str,
        collection: str,
        reason: str,
        original_error: Exception | None = None
    ) -> None:
        message = f"ChromaDB {operation} sur '{collection}' a échoué: {reason}"
        super().__init__(
            message=message,
            details={"operation": operation, "collection": collection, "reason": reason},
            original_error=original_error
        )


class RuleNotFoundError(MemoryError):
    """Erreur quand une règle métier n'est pas trouvée."""

    def __init__(self, rule_id: str) -> None:
        message = f"Règle métier '{rule_id}' non trouvée"
        super().__init__(message=message, details={"rule_id": rule_id})


# =============================================================================
# API ERRORS - Erreurs liées à l'API
# =============================================================================


class APIError(DataSentinelError):
    """Erreur de base liée à l'API."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        super().__init__(message=message, details=details)


class InvalidRequestError(APIError):
    """Erreur de requête invalide (400)."""

    def __init__(self, reason: str, field: str | None = None) -> None:
        details = {"reason": reason}
        if field:
            details["field"] = field
        super().__init__(
            message=f"Requête invalide: {reason}",
            status_code=400,
            details=details
        )


class NotFoundError(APIError):
    """Erreur ressource non trouvée (404)."""

    def __init__(self, resource_type: str, resource_id: str) -> None:
        super().__init__(
            message=f"{resource_type} '{resource_id}' non trouvé",
            status_code=404,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class SessionNotFoundError(NotFoundError):
    """Erreur session non trouvée."""

    def __init__(self, session_id: str) -> None:
        super().__init__(resource_type="Session", resource_id=session_id)


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(DataSentinelError):
    """Erreur de configuration."""

    def __init__(self, parameter: str, reason: str) -> None:
        message = f"Configuration invalide pour '{parameter}': {reason}"
        super().__init__(message=message, details={"parameter": parameter, "reason": reason})


class MissingAPIKeyError(ConfigurationError):
    """Erreur quand une clé API est manquante."""

    def __init__(self, key_name: str) -> None:
        super().__init__(
            parameter=key_name,
            reason="Clé API requise mais non fournie"
        )
