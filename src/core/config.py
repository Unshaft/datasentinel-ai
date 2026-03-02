"""
Configuration centralisée de DataSentinel AI.

Utilise pydantic-settings pour une gestion robuste des variables d'environnement
avec validation de types et valeurs par défaut.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration principale de l'application.

    Les valeurs sont chargées depuis les variables d'environnement
    ou le fichier .env à la racine du projet.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,  # ex: CORS_ORIGINS= vide → utilise la valeur par défaut
    )

    # ===================
    # API Keys
    # ===================
    anthropic_api_key: str = Field(
        ...,
        description="Clé API Anthropic pour Claude"
    )

    # ===================
    # Application
    # ===================
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environnement d'exécution"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Niveau de logging"
    )

    # ===================
    # API Server
    # ===================
    api_host: str = Field(
        default="0.0.0.0",
        description="Host pour le serveur FastAPI"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port pour le serveur FastAPI"
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        description="Nombre de workers uvicorn"
    )

    # ===================
    # ChromaDB
    # ===================
    chroma_persist_path: Path = Field(
        default=Path("./data/chroma"),
        description="Chemin de persistance ChromaDB"
    )
    chroma_rules_collection: str = Field(
        default="business_rules",
        description="Collection pour les règles métier"
    )
    chroma_decisions_collection: str = Field(
        default="decision_history",
        description="Collection pour l'historique des décisions"
    )
    chroma_feedback_collection: str = Field(
        default="user_feedback",
        description="Collection pour les feedbacks utilisateur"
    )

    # ===================
    # Model Configuration
    # ===================
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Modèle Claude à utiliser"
    )
    claude_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Température pour les réponses"
    )
    claude_max_tokens: int = Field(
        default=4096,
        ge=1,
        le=8192,
        description="Tokens maximum par réponse"
    )

    # ===================
    # Agent Configuration
    # ===================
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Seuil de confiance pour escalade humaine"
    )
    max_agent_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Nombre maximum d'itérations par agent"
    )
    agent_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout par appel agent (secondes)"
    )

    # ===================
    # ML Configuration
    # ===================
    anomaly_contamination: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Taux de contamination pour Isolation Forest"
    )
    drift_pvalue_threshold: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="Seuil p-value pour détection de drift"
    )

    # ===================
    # Data Limits
    # ===================
    max_upload_size: int = Field(
        default=100 * 1024 * 1024,  # 100 MB
        description="Taille maximale de fichier uploadé (bytes)"
    )
    max_rows_analyze: int = Field(
        default=100000,
        ge=0,
        description="Nombre maximum de lignes à analyser (0 = illimité)"
    )

    # ===================
    # Security / Auth
    # ===================
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Origins CORS autorisés"
    )
    api_secret_key: str = Field(
        default="changeme-in-production-use-a-long-random-string",
        description="Clé secrète JWT (doit être longue et aléatoire en production)"
    )
    auth_enabled: bool = Field(
        default=False,
        description="Activer l'authentification JWT (désactivée par défaut en dev)"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="Algorithme de signature JWT"
    )
    jwt_expire_minutes: int = Field(
        default=60,
        ge=1,
        description="Durée de vie du token JWT en minutes"
    )
    api_username: str = Field(
        default="admin",
        description="Nom d'utilisateur pour l'authentification (dev)"
    )
    api_password: str = Field(
        default="changeme",
        description="Mot de passe pour l'authentification (dev)"
    )

    # ===================
    # Redis / Sessions
    # ===================
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="URL de connexion Redis"
    )
    session_ttl: int = Field(
        default=3600,
        ge=60,
        description="Durée de vie des sessions en secondes (défaut 1h)"
    )

    @field_validator("chroma_persist_path", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convertit string en Path."""
        return Path(v) if isinstance(v, str) else v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: str | list) -> list[str]:
        """Parse CORS origins depuis string comma-separated ou liste."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def is_development(self) -> bool:
        """Vérifie si on est en mode développement."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Vérifie si on est en mode production."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Retourne l'instance singleton des settings.

    Utilise lru_cache pour éviter de recharger les settings
    à chaque appel.

    Returns:
        Settings: Instance de configuration
    """
    return Settings()


# Alias pour faciliter l'import
settings = get_settings()
