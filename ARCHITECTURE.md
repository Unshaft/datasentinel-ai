# Architecture de DataSentinel AI

Ce document décrit l'architecture technique du système DataSentinel AI, un système multi-agents pour la qualité des données.

---

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Principes de conception](#principes-de-conception)
3. [Architecture des agents](#architecture-des-agents)
4. [Flux de données](#flux-de-données)
5. [Couche mémoire](#couche-mémoire)
6. [Composants ML](#composants-ml)
7. [Décisions techniques](#décisions-techniques)
8. [Limites connues](#limites-connues)
9. [Pistes d'amélioration](#pistes-damélioration)

---

## Vue d'ensemble

### Diagramme d'architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Gateway                                 │
│                    /analyze  /recommend  /explain  /feedback                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR AGENT                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ Task Router │  │ Agent Picker │  │ Aggregator  │  │ Confidence Check │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   PROFILER   │  │   QUALITY    │  │  CORRECTOR   │  │  VALIDATOR   │
│    AGENT     │  │    AGENT     │  │    AGENT     │  │    AGENT     │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   ChromaDB      │  │  Decision Log   │  │   Feedback Store            │  │
│  │  (Business      │  │  (Past actions  │  │   (User corrections)        │  │
│  │   Rules RAG)    │  │   & outcomes)   │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML COMPONENTS                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Isolation       │  │  Drift          │  │   Confidence                │  │
│  │ Forest          │  │  Detector       │  │   Scorer                    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Principes de conception

### 1. Séparation des responsabilités (SRP)
Chaque agent a une responsabilité unique et bien définie:
- **Profiler**: Comprendre les données
- **Quality**: Détecter les problèmes
- **Corrector**: Proposer des solutions
- **Validator**: Vérifier la conformité

### 2. Un seul LLM, plusieurs personnalités
Le système utilise **un seul modèle Claude** avec des prompts spécialisés par agent. Ceci permet de:
- Réduire les coûts API
- Simplifier la maintenance
- Garantir la cohérence

### 3. Orchestration logique, pas technique
Le "multi-agent" est **architectural**, pas technique:
- L'Orchestrator décide quel "mode" activer
- Chaque agent est un ensemble de prompts + tools + logique
- Pas de communication inter-agents complexe

### 4. Explicabilité native
Chaque décision est accompagnée de:
- Une justification textuelle
- Un score de confiance
- Les facteurs contributifs
- Les alternatives considérées

### 5. Apprentissage continu
Le système apprend via:
- L'historique des décisions (decision_log)
- Les feedbacks utilisateur (feedback_store)
- La recherche sémantique de cas similaires (RAG)

---

## Architecture des agents

### Orchestrator Agent

**Rôle**: Chef d'orchestre qui coordonne tous les autres agents.

**Responsabilités**:
| Composant | Fonction |
|-----------|----------|
| Task Router | Détermine le type de tâche demandée |
| Agent Picker | Sélectionne les agents à activer |
| Aggregator | Fusionne les résultats |
| Confidence Check | Décide si escalade nécessaire |

**Logique de décision**:
```python
if task == "analyze":
    run(Profiler) -> run(Quality)
elif task == "recommend":
    run(Profiler) -> run(Quality) -> run(Corrector)
elif task == "full":
    run(Profiler) -> run(Quality) -> run(Corrector) -> run(Validator)
```

### Profiler Agent

**Rôle**: Établir la carte d'identité du dataset.

**Outputs**:
- `DataProfile`: Métadonnées et dimensions
- `ColumnProfile[]`: Stats par colonne
- `data_hash`: Signature pour drift

**Ne fait PAS**:
- Détection de problèmes (Quality)
- Proposition de corrections (Corrector)

### Quality Agent

**Rôle**: Identifier tous les problèmes de qualité.

**Méthodes de détection**:
| Type | Méthode | Seuil |
|------|---------|-------|
| Missing values | Count + % | Configurable |
| Anomalies | Isolation Forest | contamination=0.1 |
| Type mismatch | Tentative de conversion | N/A |
| Drift | KS-test + PSI | p-value < 0.05 |
| Constraints (niveau 1) | Heuristique colonnes `*_id` | Doublon détecté |
| Constraints (niveau 2) | RAG ChromaDB + règles métier | Similarité ≥ 0.55 |

**Dégradation gracieuse** : si ChromaDB est indisponible, le niveau 1 reste actif.
L'erreur est capturée dans `AgentContext.metadata["rules_validation_error"]`.

**Output**: Liste de `QualityIssue` avec sévérité et confiance.

### Corrector Agent

**Rôle**: Proposer des corrections **sans les appliquer**.

**Types de corrections**:
- Imputation (mean, median, mode)
- Suppression (lignes, colonnes)
- Transformation (clipping, casting)
- Marquage (flag pour revue)

**Principe clé**: Jamais d'application automatique. Toujours avec justification.

### Validator Agent

**Rôle**: Gardien final vérifiant la conformité.

**Validations**:
1. Conformité aux règles métier (RAG)
2. Cohérence des paramètres
3. Simulation d'impact
4. Non-création de nouveaux problèmes

**Output**: `ValidationResult` avec statut et raisons.

---

## Flux de données

### Pipeline d'analyse typique

```
┌──────────┐
│ Dataset  │ (DataFrame)
└────┬─────┘
     │
     ▼
┌──────────┐     ┌─────────────┐
│ Profiler │────▶│ DataProfile │
└────┬─────┘     └─────────────┘
     │
     ▼
┌──────────┐     ┌───────────────┐
│ Quality  │────▶│ QualityIssue[]│
└────┬─────┘     └───────────────┘
     │
     ▼
┌──────────┐     ┌────────────────────┐
│Corrector │────▶│CorrectionProposal[]│
└────┬─────┘     └────────────────────┘
     │
     ▼
┌──────────┐     ┌──────────────────┐
│Validator │────▶│ValidationResult[]│
└────┬─────┘     └──────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│ AgentContext (résultat agrégé)   │
└──────────────────────────────────┘
```

### Modèle de contexte partagé

Tous les agents partagent un `AgentContext`:
```python
class AgentContext:
    session_id: str
    dataset_id: str
    profile: DataProfile | None
    issues: list[QualityIssue]
    proposals: list[CorrectionProposal]
    validations: list[ValidationResult]
    metadata: dict
```

Ce pattern évite le couplage direct entre agents.

---

## Couche mémoire

### ChromaDB Collections

| Collection | Contenu | Usage |
|------------|---------|-------|
| `business_rules` | Règles métier vectorisées | RAG pour validation |
| `decision_history` | Décisions passées + outcomes | Apprentissage contextuel |
| `user_feedback` | Feedbacks utilisateur | Amélioration continue |

### Flux d'apprentissage

```
                    ┌─────────────────┐
                    │ Nouvelle        │
                    │ décision        │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐
│ Recherche       │◀─│ decision_log │─▶│ Calcul ajust.   │
│ similaires      │  │              │  │ confiance       │
└─────────────────┘  └──────────────┘  └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feedback        │
                    │ utilisateur     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Mise à jour     │
                    │ was_correct     │
                    └─────────────────┘
```

---

## Composants ML

### Isolation Forest (Détection d'anomalies)

**Pourquoi ce choix**:
- Non supervisé (pas besoin de labels)
- Robuste aux données de haute dimension
- Complexité O(n log n)

**Configuration**:
```python
AnomalyDetector(
    contamination=0.1,    # 10% anomalies attendues
    n_estimators=100,     # Nombre d'arbres
    random_state=42       # Reproductibilité
)
```

### Drift Detector

**Méthodes**:
| Méthode | Type de données | Métrique |
|---------|-----------------|----------|
| KS-test | Numériques | p-value |
| Chi-squared | Catégorielles | p-value |
| PSI | Tous | Score 0-1 |

**Seuils PSI** (standard industrie):
- < 0.1: Pas de changement
- 0.1-0.25: Changement modéré
- > 0.25: Changement significatif

### Confidence Scorer

**Facteurs**:
| Facteur | Poids | Description |
|---------|-------|-------------|
| data_quality | 25% | Qualité des données d'entrée |
| sample_size | 20% | Taille de l'échantillon |
| signal_consistency | 25% | Cohérence des indicateurs |
| historical_accuracy | 15% | Précision historique |
| rule_coverage | 15% | Couverture des règles |

---

## Décisions techniques

### Pourquoi un seul LLM?

| Multi-LLM | Mono-LLM (choisi) |
|-----------|-------------------|
| Coûts élevés | Coûts maîtrisés |
| Latence cumulée | Latence réduite |
| Complexité de coordination | Simple à maintenir |
| Incohérences possibles | Cohérence garantie |

### Pourquoi ChromaDB?

| Critère | ChromaDB | Alternatives |
|---------|----------|--------------|
| Installation | Locale, simple | Serveur requis |
| Embeddings | Natifs | Configuration |
| Persistance | Fichier | Variable |
| Coût | Gratuit | Variable |

### Pourquoi FastAPI?

- Documentation OpenAPI automatique
- Validation Pydantic native
- Performance asynchrone
- Écosystème Python riche

---

## Limites connues

### Implémentées mais simplifiées

| Fonctionnalité | État | Amélioration possible |
|----------------|------|----------------------|
| Persistance sessions | Non implémentée | Ajouter Redis/PostgreSQL |
| Upload fichiers | Basique | Support multipart + streaming |
| Auth | Non implémentée | Ajouter OAuth2/JWT |
| Rate limiting | Non implémenté | Ajouter slowapi |

### Limitations techniques

1. **Taille des datasets**: Limité par la mémoire RAM
2. **Parallélisation**: Les agents s'exécutent séquentiellement
3. **LLM timeout**: Appels Claude peuvent être lents
4. **Embeddings**: Utilise les embeddings par défaut de ChromaDB

### Limitations fonctionnelles

1. **Types de données**: Focus sur tabulaire (pas de texte/image)
2. **Règles métier**: Interprétation sémantique simplifiée
3. **Corrections**: Propositions mais pas d'application automatique
4. **Multi-tenant**: Non supporté actuellement

---

## Pistes d'amélioration

### Court terme (v0.2) — Livré ✅

- [x] Tests d'intégration complets (97 tests, 100 % pass)
- [x] Support CSV/Parquet upload (`POST /upload`, pyarrow)
- [x] Métriques Prometheus (`GET /metrics`, prometheus-fastapi-instrumentator)
- [x] Authentification JWT (`POST /auth/token`, opt-in `AUTH_ENABLED`)
- [x] Persistance des sessions Redis (`GET /analyze/{session_id}`, fallback in-memory)

### Moyen terme (v0.3)

- [ ] Interface web (Streamlit ou React)
- [ ] Webhooks pour notifications async
- [ ] Parallélisation des agents indépendants (asyncio)
- [ ] Rate limiting (slowapi)

### Long terme (v1.0)

- [ ] Support multi-tenant
- [ ] Fine-tuning du LLM sur les décisions
- [ ] Détection de patterns temporels
- [ ] API GraphQL alternative
- [ ] Déploiement Kubernetes

---

## Glossaire

| Terme | Définition |
|-------|------------|
| **Agent** | Composant avec rôle spécifique et prompt dédié |
| **Orchestrator** | Agent coordinateur des autres agents |
| **RAG** | Retrieval-Augmented Generation |
| **PSI** | Population Stability Index |
| **Drift** | Changement de distribution des données |
| **Escalade** | Remontée pour intervention humaine |
| **Tool** | Fonction appelable par un agent LangChain |

---

*Document mis à jour pour DataSentinel AI v0.1.1*
*(v0.1.0 → v0.1.1 : 4 bugs corrigés, 66 tests unitaires et d'intégration ajoutés)*
