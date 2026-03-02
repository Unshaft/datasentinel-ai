# DataSentinel AI — Journal de développement

Ce fichier documente les décisions techniques prises, les bugs corrigés et le
raisonnement derrière chaque choix. Utile pour la revue de code et les entretiens.

---

## Session 1 — Audit et corrections de bugs (v0.1.0)

### Contexte

Reprise du projet DataSentinel AI v0.1.0 : système multi-agents de qualité de données
basé sur FastAPI, LangChain, Claude (Anthropic) et ChromaDB. Objectif : maîtriser
l'ensemble du code et le rendre fonctionnel end-to-end avant d'avancer vers v0.2.

### Méthodologie d'audit

Lecture complète de tous les fichiers sources dans l'ordre de la chaîne d'appel :

```
config.py → models.py → base.py → orchestrator.py
→ profiler.py → quality.py → corrector.py → validator.py
→ chroma_store.py → anomaly_detector.py → drift_detector.py
→ api/main.py → routes/*.py
```

---

## Bugs identifiés et corrigés

---

### Bug 1 — `KeyError: 'estimated_improvement'` dans `demo.py`

**Fichier :** `src/agents/orchestrator.py` — `_format_recommendation_response()`

**Symptôme :**
```
KeyError: 'estimated_improvement'
```
Le script `scripts/demo.py` accède à `result['estimated_improvement']` après un appel
à `orchestrator.recommend()`. Cette clé n'existait pas dans le dictionnaire retourné.

**Cause racine :**
La méthode `_format_recommendation_response()` formate le résultat de la méthode
haut niveau `recommend()`, mais ne calculait pas `estimated_improvement`.
En revanche, le **route FastAPI** `/recommend` calculait ce champ lui-même — le bug
n'apparaissait donc pas en mode API, uniquement en mode programmatique (demo.py).

C'est un cas classique de **divergence d'interface** : deux chemins d'appel différents
(API vs code direct) produisant des résultats différents.

**Correction :**
Ajout du calcul dans `_format_recommendation_response()`, aligné sur la logique
existante dans le router :

```python
quality_score = context.metadata.get("quality_score", 100)
pending_count = sum(1 for p in context.proposals if p.is_approved is not False)
base["estimated_improvement"] = round(
    min(100 - quality_score, pending_count * 5), 1
)
```

**Logique de calcul :** chaque correction non rejetée améliore le score d'environ
5 points, plafonné par l'écart restant jusqu'à 100 %.

**Ce qu'on apprend :** quand une méthode haut niveau et un endpoint API font la même
chose, le formatage final doit être centralisé — ici dans la méthode, pas dans le router.

---

### Bug 2 — `None` invalide dans les métadonnées ChromaDB

**Fichier :** `src/memory/chroma_store.py` — `log_decision()`

**Symptôme :**
```
chromadb.errors.InvalidArgumentError: metadata values must be str, int, float, or bool
```
Crash systématique dès le premier appel à un agent, car chaque agent logue sa décision.

**Cause racine :**
ChromaDB impose que les valeurs de métadonnées soient de type `str`, `int`, `float`
ou `bool`. Le champ `was_correct` était initialisé à `None` (Python `NoneType`) :

```python
# AVANT — crash ChromaDB
metadata = {
    ...
    "was_correct": None,  # NoneType interdit par ChromaDB
}
```

Ce champ sert à stocker le feedback utilisateur a posteriori : on sait si une
décision était correcte uniquement après retour utilisateur, pas à la création.

**Correction :**
Suppression pure du champ à la création. ChromaDB supporte l'ajout de nouveaux
champs via `update()` après coup — `_update_decision_correctness()` fait déjà ça
correctement. L'absence de la clé est gérée par `.get("was_correct")` qui retourne
`None` en Python, ce qui n'affecte pas `get_decision_accuracy()`.

```python
# APRÈS — pas de None, la clé est ajoutée après feedback
metadata = {
    "agent_type": agent_type,
    "action": action,
    "confidence": confidence,
    "context_hash": ...,
    "created_at": ...,
    # was_correct absent : sera ajouté via _update_decision_correctness()
}
```

**Ce qu'on apprend :** ne jamais stocker une valeur "inconnue pour l'instant" dans
ChromaDB avec `None`. Soit on omet la clé, soit on utilise une chaîne sentinelle
(`"unknown"`). Ici, l'omission est plus propre car `.get()` gère déjà ce cas.

---

### Bug 3 — Crash sur collection ChromaDB vide

**Fichier :** `src/memory/chroma_store.py` — `search_rules()`, `find_similar_decisions()`,
`search_similar_feedback()`

**Symptôme :**
```
chromadb.errors.InvalidArgumentError: Collection is empty
```
Crash au démarrage ou lors de la première analyse si ChromaDB n'a pas encore été
alimenté (base fraîche, tests, premier lancement).

**Cause racine :**
ChromaDB lève une exception si on appelle `.query()` sur une collection vide.
De plus, si `n_results` dépasse le nombre de documents présents, ChromaDB peut
également lever une erreur selon la version.

**Correction :**
Double guard avant chaque `query()` :
1. Guard sur le count (retour rapide si vide)
2. Limitation de `n_results` au nombre réel de documents

```python
if self.rules_collection.count() == 0:
    return []

actual_n = min(n_results, self.rules_collection.count())
results = self.rules_collection.query(..., n_results=actual_n, ...)
```

Appliqué aux trois méthodes de recherche : `search_rules`, `find_similar_decisions`,
`search_similar_feedback`.

**Ce qu'on apprend :** les clients de bases de données vectorielles ne gèrent pas
toujours gracieusement les états vides. Toujours défensif sur les cas limites au démarrage.

---

### Bug 4 — `_validate_against_rules` non branchée sur ChromaDB

**Fichier :** `src/agents/quality.py` — `_validate_against_rules()`

**Symptôme :**
Pas de crash, mais le `QualityAgent` disposait de tools LangChain pour accéder
aux règles ChromaDB (`create_rules_tools()` dans `self.tools`) sans jamais les
utiliser dans la méthode de validation. Seule une heuristique statique (unicité
des colonnes `*_id`) était exécutée. Les règles métier chargées dans ChromaDB
via `scripts/init_chroma.py` étaient ignorées.

**Cause racine :**
Les tools LangChain sont conçus pour être appelés par le LLM (via `bind_tools`).
La méthode `_validate_against_rules` est du code Python synchrone qui ne passe
pas par le LLM — elle n'avait donc pas accès au store ChromaDB directement.

**Correction :**
Ajout d'un accès direct au `ChromaStore` dans `QualityAgent.__init__()` :

```python
self.store = get_chroma_store()
```

Refactoring de `_validate_against_rules` en deux niveaux :

**Niveau 1 — Heuristique embarquée (toujours active) :**
Unicité des colonnes identifiants (`*_id`). Rapide, sans dépendance externe.

**Niveau 2 — RAG sur ChromaDB (actif si base alimentée) :**
Pour chaque colonne, recherche sémantique des règles applicables. Deux patterns
vérifiés automatiquement si une règle pertinente (similarité ≥ 0.55) est trouvée :
- Règles d'unicité (`unique`, `unicité`) → vérifie les doublons
- Règles de complétude (`not null`, `obligatoire`) → vérifie les nulls

Le niveau 2 est entouré d'un `try/except` : si ChromaDB est indisponible,
seul le niveau 1 s'exécute (dégradation gracieuse).

**Ce qu'on apprend :**
- LangChain tools ≠ accès direct Python. Les tools sont pour le LLM, pas pour
  le code déterministe.
- Toujours implémenter une **dégradation gracieuse** : le système doit fonctionner
  même sans sa couche de mémoire.
- Le seuil de similarité (0.55) est un paramètre de compromis : trop bas = faux
  positifs, trop haut = règles ignorées.

---

## Résumé des fichiers modifiés

| Fichier | Nature de la modification |
|---------|--------------------------|
| `src/agents/orchestrator.py` | Ajout calcul `estimated_improvement` dans `_format_recommendation_response` |
| `src/memory/chroma_store.py` | Suppression `was_correct: None` + guards collection vide (×3) |
| `src/agents/quality.py` | Injection `ChromaStore` + refactoring `_validate_against_rules` (2 niveaux) |

---

## Points de vigilance pour la suite

### Architecture
- **Agents séquentiels** : le pipeline s'exécute en série (Profiler → Quality →
  Corrector → Validator). Pour v0.3, paralléliser Profiler+Quality améliorerait
  la latence.
- **Pas de persistance de session** : chaque requête recrée l'`AgentContext` en
  mémoire. Une reprise de session (v0.2 avec Redis) permettrait l'accumulation
  de feedback sur une même analyse.

### Tests manquants (priorité v0.2)
- `tests/integration/` est vide. Priorité : un test end-to-end du pipeline avec
  un DataFrame factice et un mock de l'API Claude.
- Les tests unitaires existants (`test_anomaly_detector.py`, `test_drift_detector.py`)
  ne couvrent pas les agents ni l'orchestrateur.

### ChromaDB
- La collection `business_rules` doit être alimentée via `scripts/init_chroma.py`
  pour que le niveau 2 de `_validate_against_rules` soit actif.
- Le Singleton `ChromaStore` peut poser des problèmes en tests parallèles.
  À refactorer en injection de dépendance si les tests deviennent nombreux.

---

---

## Session 2 — Écriture des tests

### Stratégie de test adoptée

**Principe clé** : le pipeline principal (Profiler → Quality → Corrector → Validator)
n'appelle **pas le LLM**. Claude n'est invoqué que dans les méthodes `*_with_llm()`
(analyse qualitative additionnelle). Le cœur déterministe est donc testable sans
mock LLM, uniquement avec :

- Un mock du `DecisionLogger` (évite les appels ChromaDB pour les logs)
- Un mock du `ChromaStore` (isole la DB vectorielle)

### Fichiers créés

```text
tests/
├── unit/
│   ├── test_chroma_store.py    → Bugs 2 & 3 (None, collection vide)
│   ├── test_quality_agent.py   → Bug 4 (_validate_against_rules 2 niveaux)
│   └── test_orchestrator.py    → Bug 1 (estimated_improvement) + score + escalade
└── integration/
    └── test_pipeline.py        → Pipeline complet end-to-end
```

### Décisions de test notables

**Isolation via monkeypatch, pas d'héritage**
Les agents sont instanciés directement avec `patch()` sur les points d'injection
(`get_chroma_store`, `get_decision_logger`). On évite les sous-classes de test qui
coupleraient les tests à l'implémentation interne.

**Test de dégradation gracieuse (test_level1_works_without_chromadb)**
Un `ChromaStore` qui lève une exception ne doit pas crasher le pipeline. Ce test
vérifie que l'erreur est capturée dans `context.metadata["rules_validation_error"]`
et que le niveau 1 (heuristique) continue de fonctionner.

**Intégrité référentielle (test_proposals_reference_valid_issues)**
Chaque `CorrectionProposal.issue_id` doit référencer un `QualityIssue.issue_id`
existant dans le contexte. Ce test garantit que le Corrector ne génère pas de
propositions "orphelines" déconnectées du contexte réel.

**Test de divergence d'interface (test_recommend_returns_estimated_improvement)**
Ce test aurait détecté le Bug 1 avant qu'il arrive en prod : il appelle directement
`orchestrator.recommend()` (chemin programmatique) plutôt que le router FastAPI.

### Comment lancer les tests

```bash
# Tous les tests
pytest tests/ -v

# Uniquement les unitaires
pytest tests/unit/ -v

# Uniquement les tests d'intégration pipeline
pytest tests/integration/ -v

# Un test spécifique
pytest tests/unit/test_chroma_store.py::TestChromaStoreEmptyCollection -v

# Avec coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Ce qu'on apprend

**Tester les deux chemins d'appel** : API router ET méthode directe. La divergence
entre les deux (Bug 1) est un antipattern classique. Si `estimated_improvement` avait
eu un test unitaire sur `orchestrator.recommend()`, le bug n'aurait pas existé.

**Fixtures partagées** : le `conftest.py` centralise `mock_chroma_store` et
`mock_llm_response` — réutilisés par tous les tests sans duplication.

**Pas de base de données réelle dans les tests unitaires** : le `tmp_path` de pytest
crée un répertoire temporaire propre par test, idéal pour ChromaDB qui écrit sur disque.

**Les contraintes des bibliothèques ML s'appliquent aux fixtures** : `IsolationForest`
requiert au moins 10 échantillons. La fixture `sample_dirty_df` n'en avait que 5 —
un test pré-existant (`test_fit_ignores_non_numeric`) échouait donc avec
`InsufficientDataError`. Fix : étendre la fixture à 10 lignes en conservant les
mêmes caractéristiques (doublons, nulls, anomalies). Résultat final : **66/66 tests passent**.

---

## Session 3 — Implémentation v0.2 (4 features)

### Contexte

Après la stabilisation v0.1.1 (66/66 tests), implémentation des 4 features v0.2
dans l'ordre Upload → Prometheus → JWT → Redis, en une seule session.
Résultat : **97/97 tests passent**, 4 nouveaux fichiers source, 3 nouveaux fichiers de tests.

---

### Feature 1 — Upload CSV/Parquet (`POST /upload`)

**Décision clé** : l'endpoint fait parse + analyse + retourne `AnalyzeResponse` en un seul
appel. Alternative rejetée : retourner juste un `file_id` sans analyse (nécessite
un second appel, valeur réduite).

**Guard de taille** : vérification de `len(content)` avant le parsing — évite de charger
un fichier de 500 MB en mémoire pour ensuite le rejeter.

**Troncature silencieuse** : si `max_rows_analyze > 0`, les lignes excédentaires sont
coupées sans erreur. L'appelant voit `row_count` dans le profil et peut s'en apercevoir.

---

### Feature 2 — Prometheus

**3 lignes de code** : `Instrumentator().instrument(app).expose(app)` après la création de
`app`. Pas de middleware custom, pas de métriques manuelles. Le instrumentator gère
automatiquement `http_requests_total` et `http_request_duration_seconds`.

**Ce qu'on apprend** : en FastAPI, ajouter l'observabilité après les routers (pas avant)
pour que `expose(app)` crée `/metrics` sans conflit de routes.

---

### Feature 3 — Authentification JWT

**Pattern opt-in** : `auth_enabled=False` par défaut. `get_current_user` retourne
`{"user": "anonymous"}` sans vérifier le token quand l'auth est désactivée.
Aucun test existant ne casse car ils n'appellent pas la couche HTTP.

**Protection via `include_router`** : plutôt que d'ajouter `Depends(get_current_user)`
sur chaque endpoint (fragile, oubli possible), on passe `dependencies=_auth_dep` au
niveau du `include_router`. Une seule ligne par router — impossible d'oublier un endpoint.

```python
_auth_dep = [Depends(get_current_user)]
app.include_router(analyze.router, dependencies=_auth_dep)
```

**`auto_error=False` sur `OAuth2PasswordBearer`** : sans ça, FastAPI lève 401 automatiquement
quand le token est absent, même quand `auth_enabled=False`. Avec `auto_error=False`,
le token est `None` et c'est `get_current_user` qui décide du comportement.

**Tests async sans pytest-asyncio** : `get_current_user` est `async def`. Plutôt
qu'ajouter `pytest-asyncio` (une dépendance de plus), on utilise `asyncio.run(coro)`
dans les tests — compatible Python 3.10+ et zéro config.

---

### Feature 4 — Persistance de session Redis

**Pattern Fallback** : `SessionStore._connect()` tente `redis.from_url().ping()`. En cas
d'exception, switche vers `InMemoryFallback` (dict process-level). Même interface
(`set`, `get`, `delete`), comportement identique pour l'appelant.

**Sérialisation Pydantic v2** : `context.model_dump(mode="json")` gère les types complexes
(datetime, Enum) mieux que `json.dumps(context.dict())` (v1 déprécié). Désérialisation
via `AgentContext.model_validate(data)`.

**Persistance best-effort** : le `try/except` autour de `save()` garantit que si Redis
tombe en pleine requête, la réponse HTTP est quand même retournée. Le client perd
juste la possibilité de récupérer la session plus tard — comportement acceptable.

**Tests avec `fakeredis`** : simule un serveur Redis en mémoire, support du TTL réel
(les tests de TTL utilisent `time.sleep(1.1)`). Alternative `unittest.mock.MagicMock`
rejetée : elle ne simule pas le TTL.

---

### Résumé des fichiers v0.2

| Fichier | Type | Description |
|---------|------|-------------|
| `src/api/routes/upload.py` | Nouveau | Upload CSV/Parquet → pipeline |
| `src/api/auth.py` | Nouveau | JWT create/verify/dependency |
| `src/api/routes/auth.py` | Nouveau | `POST /auth/token` |
| `src/memory/session_store.py` | Nouveau | Redis + fallback in-memory |
| `src/api/main.py` | Modifié | Prometheus, upload router, auth router, JWT deps |
| `src/core/config.py` | Modifié | Settings JWT + Redis |
| `src/api/routes/analyze.py` | Modifié | Save session POST, load session GET |
| `requirements.txt` | Modifié | 5 nouvelles dépendances |
| `tests/unit/test_auth.py` | Nouveau | 9 tests JWT |
| `tests/unit/test_session_store.py` | Nouveau | 10 tests Redis/fallback |
| `tests/integration/test_upload.py` | Nouveau | 6 tests upload |

---

## Lexique technique (pour entretien)

| Terme | Explication |
|-------|-------------|
| **RAG** | Retrieval-Augmented Generation : on enrichit le prompt LLM avec des documents récupérés par similarité vectorielle (ici les règles métier dans ChromaDB) |
| **Isolation Forest** | Algorithme ML non supervisé de détection d'anomalies : les points anormaux sont plus faciles à "isoler" dans un arbre aléatoire |
| **PSI** | Population Stability Index : mesure le changement de distribution d'une variable entre deux datasets (< 0.1 = stable, > 0.25 = dérive significative) |
| **AgentContext** | Objet partagé qui circule entre tous les agents, accumulant profil, issues, propositions et validations au fil du pipeline |
| **Template Method** | Pattern de conception utilisé dans `BaseAgent` : la classe de base définit le squelette (`execute`) que chaque agent spécialise |
| **Dégradation gracieuse** | Le système continue de fonctionner (avec moins de fonctionnalités) si un composant est indisponible (ex: ChromaDB vide) |
| **JWT** | JSON Web Token : token signé contenant le payload utilisateur, vérifié sans appel base de données |
| **OAuth2 Password Flow** | Flux d'auth où le client envoie username+password en form-data et reçoit un Bearer token |
| **Prometheus** | Système de monitoring qui scrape `/metrics` toutes les N secondes et stocke les séries temporelles |
| **TTL** | Time To Live : durée de vie d'une clé Redis après laquelle elle est supprimée automatiquement |
| **fakeredis** | Bibliothèque simulant un serveur Redis en mémoire pour les tests unitaires |
