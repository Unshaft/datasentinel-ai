# DataSentinel AI

> Système multi-agents IA pour la qualité des données

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)

DataSentinel AI est un système d'IA agentique capable d'**analyser**, **détecter** et **corriger** les problèmes de qualité de données de manière autonome et explicable.

---

## Caractéristiques

### Multi-Agents Architecture
- **Orchestrator Agent** : Coordonne l'ensemble du pipeline
- **Profiler Agent** : Analyse statistique et profilage
- **Quality Agent** : Détection d'anomalies et problèmes
- **Corrector Agent** : Proposition de corrections justifiées
- **Validator Agent** : Validation contre les règles métier

### Détection Intelligente
- Valeurs manquantes avec seuils configurables
- Anomalies statistiques (Isolation Forest)
- Drift de distribution (KS-test, PSI)
- Violations de contraintes métier

### Explicabilité
- Chaque décision est justifiée
- Score de confiance transparent
- Historique des décisions pour apprentissage
- Feedback loop pour amélioration continue

### API Production-Ready
- FastAPI avec documentation OpenAPI
- Endpoints RESTful (`/analyze`, `/recommend`, `/explain`, `/feedback`)
- Docker-ready

---

## Quick Start

### Installation

```bash
# Cloner le repository
git clone https://github.com/yourusername/datasentinel-ai.git
cd datasentinel-ai

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Copier et configurer les variables d'environnement
cp .env.example .env
# Éditer .env et ajouter votre ANTHROPIC_API_KEY
```

### Lancer l'API

```bash
# Mode développement
uvicorn src.api.main:app --reload

# Ou via le script
python -m src.api.main
```

L'API sera disponible sur `http://localhost:8000`

- Documentation Swagger: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Exemple d'utilisation

```python
import requests

# Analyser un dataset
response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "data": {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", None, "Diana", "Eve"],
            "age": [25, 30, 35, 200, 28],  # 200 = anomalie
            "salary": [50000, 60000, 55000, 70000, None]
        },
        "detect_anomalies": True
    }
)

result = response.json()
print(f"Score de qualité: {result['quality_score']}%")
print(f"Problèmes détectés: {len(result['issues'])}")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
│              /analyze  /recommend  /explain  /feedback           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT                          │
│    Routing │ Agent Selection │ Aggregation │ Escalation          │
└─────────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   PROFILER   │ │   QUALITY    │ │  CORRECTOR   │ │  VALIDATOR   │
│    Agent     │ │    Agent     │ │    Agent     │ │    Agent     │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY LAYER                              │
│     ChromaDB (Rules) │ Decision Log │ Feedback Store             │
└─────────────────────────────────────────────────────────────────┘
```

Pour une description détaillée, voir [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/analyze` | POST | Analyse un dataset et détecte les problèmes |
| `/recommend` | POST | Propose des corrections pour les problèmes |
| `/explain` | POST | Explique une décision du système |
| `/feedback` | POST | Enregistre un feedback utilisateur |
| `/feedback/rules` | GET/POST | Gère les règles métier |
| `/health` | GET | Vérification de santé |

---

## Structure du Projet

```
datasentinel-ai/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── agents/           # Agents LangChain
│   │   ├── orchestrator.py
│   │   ├── profiler.py
│   │   ├── quality.py
│   │   ├── corrector.py
│   │   └── validator.py
│   ├── tools/            # LangChain Tools
│   ├── memory/           # ChromaDB stores
│   ├── ml/               # Composants ML
│   │   ├── anomaly_detector.py
│   │   ├── drift_detector.py
│   │   └── confidence_scorer.py
│   └── core/             # Config et modèles
├── tests/                # Tests pytest
├── data/
│   └── rules/            # Règles métier par défaut
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Technologies

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| LLM | Claude (Anthropic) | Raisonnement des agents |
| Framework | LangChain | Orchestration des agents |
| Vector Store | ChromaDB | RAG et mémoire |
| ML | Scikit-learn | Détection d'anomalies |
| API | FastAPI | Exposition HTTP |
| Validation | Pydantic | Schémas et validation |

---

## Configuration

Variables d'environnement principales (voir `.env.example`):

```bash
# Requis
ANTHROPIC_API_KEY=sk-ant-xxx

# Optionnel
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CONFIDENCE_THRESHOLD=0.7
ANOMALY_CONTAMINATION=0.1
```

---

## Développement

### Tests

```bash
# Lancer les tests
pytest

# Avec couverture
pytest --cov=src --cov-report=html
```

### Docker

```bash
# Build et lancement
docker-compose up --build

# Mode développement (hot reload)
docker-compose --profile dev up datasentinel-dev
```

---

## Roadmap

- [x] Architecture multi-agents
- [x] Détection d'anomalies (Isolation Forest)
- [x] Détection de drift
- [x] API REST
- [x] RAG pour règles métier
- [x] Feedback loop
- [ ] Interface web
- [ ] Support fichiers (CSV, Parquet)
- [ ] Métriques Prometheus
- [ ] Tests d'intégration complets

---

## Auteur

Projet portfolio démontrant les compétences en:
- Architecture de systèmes IA multi-agents
- Data Science appliquée
- Développement d'APIs production-ready
- Design de systèmes explicables

---

## Licence

MIT
