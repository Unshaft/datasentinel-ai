"""
Configuration du logging pour DataSentinel AI.

Utilise structlog pour un logging structuré et lisible.
"""

import logging
import sys

import structlog

from src.core.config import settings


def setup_logging() -> None:
    """
    Configure le logging pour l'application.

    En développement: format coloré et lisible
    En production: format JSON pour parsing
    """
    # Niveau de log depuis la config
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configuration de base
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level
    )

    # Processeurs structlog
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_development:
        # En dev: format coloré et lisible
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # En prod: format JSON
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Retourne un logger configuré.

    Args:
        name: Nom du logger (généralement __name__)

    Returns:
        Logger structlog
    """
    return structlog.get_logger(name)


# Logger par défaut pour le module
logger = get_logger(__name__)
