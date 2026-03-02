# =============================================================================
# DataSentinel AI - Dockerfile
# =============================================================================
# Multi-stage build pour une image optimisée
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Installation des dépendances
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Variables d'environnement pour Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer les dépendances système pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer un environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Image finale légère
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Métadonnées
LABEL maintainer="DataSentinel Team" \
      version="0.4.0" \
      description="DataSentinel AI - Multi-Agent Data Quality System"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Application
    ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Chemins
    APP_HOME=/app \
    CHROMA_PERSIST_PATH=/app/data/chroma

# Créer l'utilisateur non-root
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Créer la structure de répertoires
RUN mkdir -p /app/data/chroma /app/data/rules /app/data/samples && \
    chown -R appuser:appgroup /app

# Copier l'environnement virtuel depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# Copier le code source
COPY --chown=appuser:appgroup . .

# Changer vers l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Commande de démarrage
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
