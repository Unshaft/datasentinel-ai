# DataSentinel AI

> Système multi-agents IA pour la qualité des données — v1.4

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![Claude](https://img.shields.io/badge/Claude-3.5_Sonnet-purple.svg)](https://anthropic.com/)
[![Tests](https://img.shields.io/badge/tests-316_passed-brightgreen.svg)](#tests)

DataSentinel AI est un système d'IA agentique capable d'**analyser**, **détecter** et **corriger** les problèmes de qualité de données de manière autonome et explicable. Il combine un pipeline **ReAct** (Observe → Reason → Act → Reflect), une classification sémantique des colonnes, un RAG actif sur règles métier, et un apprentissage continu par feedback.

---

## Fonctionnalités

### Pipeline multi-agents

- **Orchestrator** — Coordonne le pipeline, choisit le mode d'exécution (standard ou adaptatif ReAct)
- **Profiler Agent** — Analyse statistique complète du dataset (types, nulls, min/max, cardinalité)
- **Quality Agent** — Détection des problèmes : doublons, pseudo-nulls, formats invalides, anomalies ML, drift, violations de contraintes, types sémantiques hors-range
- **Corrector Agent** — Propositions de correction justifiées avec score d'impact estimé
- **Validator Agent** — Validation des corrections contre les règles métier
- **SemanticProfiler Agent** — Classification des colonnes en types métier (email, phone, monetary_amount, employee_id…) — heuristique + LLM optionnel

### Détection intelligente

- Valeurs manquantes et pseudo-nulls (frozenset de 20 tokens : "N/A", "null", "—"…)
- Anomalies statistiques (Isolation Forest)
- Drift de distribution (KS-test + PSI)
- Violations de contraintes (valeurs négatives sur monetary_amount, âge hors [0,150]…)
- Formats invalides (email, téléphone, URL, code postal, SIRET)
- Doublons (par ligne ou par sous-ensemble de colonnes)

### Pipeline ReAct adaptatif (F24/F31)

- **Phase 1 Observe** — Profilage du dataset
- **Phase 2 Reason** — Construction d'un plan d'exécution adapté (skip anomaly si < 30 lignes, skip drift si pas de numeric, etc.)
- **Phase 3 Act** — Exécution des agents selon le plan
- **Phase 4 Reflect** — Détection d'incohérences : `score_vs_critical` (score élevé mais issues CRITICAL), `plan_blind_spot` (plan a ignoré anomaly detection mais issues HIGH trouvées)
- **Phase 5 Observe** — Re-analyse des corrections proposées

### Mémoire et apprentissage

- **Active RAG** (F25) — ChromaDB : les règles métier les plus proches du dataset sont injectées dans le contexte de chaque analyse
- **Feedback learning** (F26) — Chaque feedback crée des règles d'exception ou des exemples de correction dans ChromaDB, ajuste les scores de confiance
- **Mémoire inter-sessions** (F30) — Historique de qualité par dataset (fingerprint hash), tendances, issues récurrentes, suggestions pro-actives
- **Agents métier** (F32) — Profils de domaine personnalisés (RH, Finance, E-Commerce…) avec trigger types, types requis, overrides de sévérité

### API production-ready

- FastAPI avec docs OpenAPI (Swagger + ReDoc)
- JWT optionnel (`AUTH_ENABLED=true`)
- Rate limiting par IP (slowapi)
- Métriques Prometheus (`/metrics`)
- Webhooks avec persistance JSON
- Export PDF (reportlab) et Excel (openpyxl)
- Upload CSV/Parquet avec détection auto du séparateur et de l'encodage
- Jobs asynchrones (HTTP 202 + polling)
- Batch jusqu'à 10 fichiers en parallèle

---

## Quick Start

### Prérequis

```bash
Python 3.10+
```

### Installation

```bash
git clone https://github.com/yourusername/datasentinel-ai.git
cd datasentinel-ai

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Éditez .env : ajoutez ANTHROPIC_API_KEY (requis pour les features LLM optionnelles)
```

### Lancer l'API

```bash
uvicorn src.api.main:app --reload
```

→ API : `http://localhost:8000`
→ Swagger : `http://localhost:8000/docs`
→ Health : `http://localhost:8000/health`

### Lancer l'interface Streamlit

```bash
streamlit run streamlit_app.py
```

→ UI : `http://localhost:8501`

---

## Interface Streamlit (v1.4)

L'interface est organisée en **8 pages** accessibles depuis la navigation latérale :

| Page | Description |
|------|-------------|
| 🏠 **Accueil** | Landing page — statut API, stats rapides, navigation |
| 🔍 **Analyse** | Upload CSV/Parquet · Score · Issues · Corrections · Profil · Schéma sémantique · Onglet ReAct |
| 📦 **Batch** | Analyse parallèle de plusieurs fichiers |
| ⏳ **Jobs** | Soumission et suivi de jobs asynchrones |
| 📊 **Stats** | Tableau de bord agrégé des analyses |
| 📋 **Règles** | CRUD des règles métier (Active RAG) |
| 🏢 **Agents Métier** | Création de profils de domaine (F32) |
| 💬 **Feedback** | Formulaire de feedback sur les issues/propositions |
| 📂 **Historique** | Tendance qualité inter-sessions d'un dataset |

---

## Endpoints API

### Analyse

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/analyze` | POST | Analyse un dataset (JSON ou file_content) |
| `/upload` | POST | Upload CSV/Parquet multipart |
| `/analyze/{id}` | GET | Récupère les résultats d'une session |
| `/analyze/{id}/corrections` | GET | Plan de corrections automatiques |
| `/analyze/{id}/apply-corrections` | POST | Applique les corrections, retourne CSV |
| `/analyze/{id}/comparison` | GET | Comparaison score avant/après corrections |
| `/analyze/{id}/report.pdf` | GET | Export PDF |
| `/analyze/{id}/report.xlsx` | GET | Export Excel (4 onglets) |
| `/analyze/{id}/schema` | GET | Export schéma sémantique JSON |

### Batch & Jobs

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/batch` | POST | Analyse parallèle (max 10 fichiers) |
| `/jobs/analyze` | POST | Soumet un job async → HTTP 202 + job_id |
| `/jobs/{job_id}` | GET | Statut et résultat d'un job |

### Mémoire & Règles

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/feedback` | POST | Enregistre un feedback utilisateur |
| `/feedback/stats` | GET | Statistiques des feedbacks |
| `/feedback/rules` | GET/POST/DELETE | CRUD règles via feedback |
| `/rules` | GET/POST/DELETE | CRUD règles métier ChromaDB |
| `/domain-agents` | GET/POST | CRUD agents métier |
| `/domain-agents/{id}` | GET/DELETE | Détail / suppression |
| `/datasets/{id}/history` | GET | Historique inter-sessions d'un dataset |

### Monitoring

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET/DELETE | Dashboard analytique agrégé |
| `/metrics` | GET | Métriques Prometheus |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Streamlit UI (8 pages)                  │
└──────────────────────────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────────────────────────────────────┐
│              FastAPI — JWT opt-in · Rate limit · Prometheus   │
│   /analyze  /upload  /batch  /jobs  /rules  /domain-agents   │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT                        │
│         Standard async │ Adaptive ReAct (5 phases)           │
└──────────────────────────────────────────────────────────────┘
        │           │            │            │           │
        ▼           ▼            ▼            ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ PROFILER │ │ QUALITY  │ │CORRECTOR │ │VALIDATOR │ │SEMANTIC  │
│          │ │ +ML +LLM │ │          │ │          │ │PROFILER  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        MEMORY LAYER                           │
│  ChromaDB (Rules + RAG) · SessionStore (Redis/InMemory)      │
│  FeedbackStore · DatasetMemory · StatsManager · JobManager   │
└──────────────────────────────────────────────────────────────┘
```

---

## Structure du projet

```
datasentinel-ai/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py       Pipeline ReAct + adaptive planning
│   │   ├── profiler.py           Analyse statistique
│   │   ├── quality.py            Détection + validation sémantique (F28) + domaine (F32)
│   │   ├── corrector.py          Propositions de correction
│   │   ├── validator.py          Validation contre règles
│   │   └── semantic_profiler.py  Classification sémantique (heuristique + LLM)
│   ├── api/
│   │   ├── main.py               FastAPI app — Prometheus, JWT, routers
│   │   ├── auth.py               JWT opt-in
│   │   ├── limiter.py            Rate limiting (slowapi)
│   │   └── routes/               analyze, upload, batch, jobs, feedback, rules,
│   │                             domain_agents, datasets, stats, webhooks, auth
│   ├── core/
│   │   ├── config.py             Settings pydantic-settings
│   │   ├── models.py             Dataclasses partagées
│   │   ├── domain_manager.py     Gestion agents métier (F32)
│   │   ├── dataset_memory.py     Mémoire inter-sessions (F30)
│   │   ├── feedback_processor.py Apprentissage continu (F26)
│   │   ├── job_manager.py        Jobs async (F21)
│   │   ├── stats_manager.py      Stats agrégées (F22)
│   │   └── webhook_manager.py    Webhooks persistants
│   ├── memory/
│   │   ├── chroma_store.py       ChromaDB (règles + RAG)
│   │   ├── session_store.py      Redis + InMemory fallback
│   │   └── decision_log.py       Historique décisions
│   └── ml/
│       ├── anomaly_detector.py   Isolation Forest
│       ├── drift_detector.py     KS-test + PSI
│       └── confidence_scorer.py  Scoring de confiance
├── pages/
│   ├── _helpers.py               Sidebar partagée + utilitaires UI
│   ├── 1_Analyse.py              Upload + résultats + onglet ReAct
│   ├── 2_Batch.py                Batch parallèle
│   ├── 3_Jobs.py                 Jobs asynchrones
│   ├── 4_Stats.py                Dashboard stats
│   ├── 5_Regles.py               CRUD règles
│   ├── 6_Agents.py               Agents métier
│   ├── 7_Feedback.py             Formulaire feedback (NEW)
│   └── 8_Historique.py           Historique dataset (NEW)
├── streamlit_app.py              Landing page
├── tests/                        316 tests (unit + integration)
├── data/                         Persistance JSON (runtime, gitignored)
├── .env.example
└── requirements.txt
```

---

## Configuration

Variables d'environnement principales (`.env`) :

```bash
# LLM (optionnel — fonctionnalités heuristiques actives sans clé)
ANTHROPIC_API_KEY=sk-ant-xxx
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# Features LLM opt-in (false par défaut)
ENABLE_LLM_CHECKS=false

# Auth (false par défaut)
AUTH_ENABLED=false
JWT_SECRET_KEY=changeme

# Redis (InMemory fallback si absent)
REDIS_URL=redis://localhost:6379

# Limites
MAX_UPLOAD_SIZE=104857600   # 100 MB
MAX_ROWS_ANALYZE=0          # 0 = illimité
```

---

## Tests

```bash
# Tous les tests
pytest tests/ -q

# Avec couverture
pytest tests/ --cov=src --cov-report=html

# Résultat actuel : 316 passed, 1 skipped
```

---

## Technologies

| Composant     | Technologie                                    |
|---------------|------------------------------------------------|
| LLM           | Claude 3.5 Sonnet (Anthropic)                  |
| Vector Store  | ChromaDB                                       |
| ML            | Scikit-learn (Isolation Forest, KS-test)       |
| API           | FastAPI + Pydantic v2                          |
| UI            | Streamlit                                      |
| Monitoring    | Prometheus (prometheus-fastapi-instrumentator) |
| Auth          | JWT (python-jose)                              |
| Rate limiting | slowapi                                        |

---

## Projet portfolio

Ce projet démontre :
- Architecture de systèmes IA multi-agents avec pipeline ReAct
- RAG actif (Active RAG) sur base vectorielle ChromaDB
- Apprentissage continu par feedback utilisateur
- Classification sémantique heuristique + LLM
- API production-ready avec auth, rate limiting, métriques, webhooks
- Interface multi-pages Streamlit avec état partagé inter-pages

---

## Licence

MIT
