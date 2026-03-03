"""
Configuration du logging pour DataSentinel AI.

Utilise structlog pour un logging structuré et lisible.
"""

import logging
import sys
from typing import Any

from src.core.config import settings

try:
    import structlog as _structlog
    _HAS_STRUCTLOG = True
except ImportError:
    _structlog = None  # type: ignore[assignment]
    _HAS_STRUCTLOG = False

# ---------------------------------------------------------------------------
# Colored formatter pour les stdlib loggers (orchestrator, quality, etc.)
# ---------------------------------------------------------------------------

_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Vert
    "WARNING":  "\033[33m",   # Jaune
    "ERROR":    "\033[31m",   # Rouge
    "CRITICAL": "\033[35m",   # Magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """Formatter coloré pour la console : LEVEL  logger_name  message."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        level = f"{color}{_BOLD}{record.levelname:<7}{_RESET}"
        name  = record.name.split(".")[-1][:14]
        return f"{level} {name:<14} {super().format(record)}"


def setup_logging() -> None:
    """
    Configure le logging pour l'application.

    En développement: format coloré et lisible
    En production: format JSON pour parsing
    """
    # Niveau de log depuis la config
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configuration stdlib root logger avec formatter coloré
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("%(message)s"))
    logging.root.handlers = [handler]
    logging.root.setLevel(log_level)

    # Configuration structlog optionnelle (si installé)
    if _HAS_STRUCTLOG:
        shared_processors = [
            _structlog.stdlib.filter_by_level,
            _structlog.stdlib.add_logger_name,
            _structlog.stdlib.add_log_level,
            _structlog.stdlib.PositionalArgumentsFormatter(),
            _structlog.processors.TimeStamper(fmt="iso"),
            _structlog.processors.StackInfoRenderer(),
            _structlog.processors.UnicodeDecoder(),
        ]
        if settings.is_development:
            processors = shared_processors + [_structlog.dev.ConsoleRenderer(colors=True)]
        else:
            processors = shared_processors + [
                _structlog.processors.format_exc_info,
                _structlog.processors.JSONRenderer(),
            ]
        _structlog.configure(
            processors=processors,
            wrapper_class=_structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=_structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> Any:
    """
    Retourne un logger configuré (structlog si disponible, stdlib sinon).

    Args:
        name: Nom du logger (généralement __name__)

    Returns:
        Logger structlog ou stdlib
    """
    if _HAS_STRUCTLOG:
        return _structlog.get_logger(name)
    return logging.getLogger(name)


# Logger par défaut pour le module
logger = get_logger(__name__)
