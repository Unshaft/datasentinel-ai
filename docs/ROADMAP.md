# DataSentinel AI — Roadmap et suivi de développement

Ce fichier pilote la progression feature par feature.
Chaque section documente le **pourquoi**, le **comment** et les **critères de validation**.

---

## État actuel — v1.1.0

| Composant | État | Notes |
|-----------|------|-------|
| Pipeline Profiler → Quality → Corrector → Validator | ✅ Opérationnel | ~306 tests passent |
| Upload CSV / Parquet (`POST /upload`) | ✅ v0.2 | `pyarrow`, validation extension + taille |
| Métriques Prometheus (`GET /metrics`) | ✅ v0.2 | Auto-instrumenté via `prometheus-fastapi-instrumentator` |
| Authentification JWT (`POST /auth/token`) | ✅ v0.2 | Opt-in (`AUTH_ENABLED=true`), fallback anonymous en dev |
| Persistance sessions Redis (`GET /analyze/{id}`) | ✅ v0.2 | Fallback in-memory si Redis indisponible |
| Quality checks en parallèle (`asyncio.gather`) | ✅ v0.3 | ~40% de latence en moins sur le pipeline Quality |
| Rate limiting (`slowapi`) | ✅ v0.3 | 30/min sur `/analyze`, 10/min sur `/upload` |
| Webhooks (`POST /webhooks`) | ✅ v0.3 | Notifications async POST JSON après analyse |
| Rapport PDF (`GET /analyze/{id}/report.pdf`) | ✅ v0.3 | `reportlab`, export professionnel |
| Interface Streamlit (`streamlit_app.py`) | ✅ v1.0 | 9 onglets analyse + 4 onglets globaux (stats, jobs, rules, agents métier) |
| Détection doublons complets (Q1) | ✅ v0.4 | `df.duplicated(keep=False)`, sévérité LOW/MEDIUM/HIGH |
| Détection pseudo-nulls (Q2) | ✅ v0.4 | "N/A", "null", "-", "unknown", etc. |
| Validation de format (Q3) | ✅ v0.4 | Email, téléphone, URL, code postal, SIRET/SIREN |
| Score de qualité par colonne (Q4) | ✅ v0.4 | Déductions par sévérité, exposé dans `AnalyzeResponse` |
| Plan de corrections (`GET /analyze/{id}/corrections`) | ✅ v0.4 | JSON auto/manual, score estimé après corrections |
| Export Excel (`GET /analyze/{id}/report.xlsx`) | ✅ v0.4 | `openpyxl`, 4 onglets, couleurs sévérité |
| Persistance JSON webhooks | ✅ v0.4 | `./data/webhooks.json`, rechargé au démarrage |
| Application des corrections (`POST /analyze/{id}/apply-corrections`) | ✅ v0.5 | Applique les corrections auto, retourne CSV propre |
| Persistance DataFrame en session (`save_dataframe` / `load_dataframe`) | ✅ v0.5 | Parquet base64 dans le store, même TTL que la session |
| Analyse en lot (`POST /batch`) | ✅ v0.5 | Jusqu'à 10 fichiers CSV/Parquet en parallèle (asyncio.gather) |
| Comparaison avant/après (`GET /analyze/{id}/comparison`) | ✅ v0.6 | Score delta + issues supprimées / restantes |
| CRUD règles métier (`GET/POST/DELETE /rules`) | ✅ v0.6 | Gestion des règles ChromaDB via API + Streamlit |
| Jobs asynchrones (`POST /jobs/analyze`) | ✅ v0.6 | Analyse non-bloquante pour gros fichiers (> 10k lignes) |
| Tableau de bord analytique (`GET /stats`) | ✅ v0.6 | Compteurs agrégés persistés JSON, top issues, score moyen |
| LLM Quality Check (Claude function calling) | ✅ v0.7 | Détection sémantique opt-in, fallback silencieux |
| Orchestrateur adaptatif (ReAct loop) | ✅ v0.7 | Plan conditionnel selon profil, `reasoning_steps` exposé |
| RAG actif dans la décision | ✅ v0.7 | Seuils dynamiques via ChromaDB, règles dans `issue.details` |
| Feedback qui améliore le comportement | ✅ v0.7 | `confidence_adjustments` persistés, règles d'exception auto |
| Classification sémantique LLM (`SemanticProfilerAgent`) | ✅ v0.8 | 1 appel batch Claude — classifie email/age/montant/etc. par colonne |
| Validation sémantique QualityAgent | ✅ v0.8 | Règles métier auto selon semantic_type (négatifs, hors-plage) |
| Export schéma sémantique (`GET /analyze/{id}/schema`) | ✅ v0.8 | JSON réutilisable avec inferred_type + semantic_type + pattern |
| Logs console colorés | ✅ v0.9 | `ColoredFormatter`, `setup_logging()` dans lifespan FastAPI |
| Custom Domain Agent Builder (`POST /domain-agents`) | ✅ v1.0 | DomainManager + CRUD /domain-agents + _validate_domain_rules + UI Streamlit |
| SemanticProfiler v2 — classifieur heuristique | ✅ v1.1 | `enrich_sync()`, `_heuristic_classify()`, fusion heuristic+LLM via `_merge_results()` |

### Dette technique résolue en v0.4

| Problème | Résolution |
| -------- | ---------- |
| `datetime.utcnow()` déprécié (Python 3.12+) | Remplacé par `datetime.now(timezone.utc)` dans tous les modèles |
| `Class Config` pydantic déprécié | Remplacé par `model_config = ConfigDict(...)` dans `responses.py` |
| Webhooks in-memory (perdus au redémarrage) | Persistance JSON dans `./data/webhooks.json` |

### Observabilité console (v0.9)

| Composant | État |
|---|---|
| `ColoredFormatter` stdlib — `INFO/WARNING/ERROR` colorés | ✅ v0.9 |
| `setup_logging()` appelé au démarrage FastAPI (lifespan) | ✅ v0.9 |
| Logs Pipeline START/END dans les 3 pipelines (orchestrator) | ✅ v0.9 |
| Logs Quality summary post-gather (nb issues par type + timing) | ✅ v0.9 |
| Logs LLM F27 : Classifying N columns + N/M cols classifiées (Xms) | ✅ v0.9 |

### Dette technique restante

| Problème | Impact | Priorité |
| -------- | ------ | -------- |
| Singleton `ChromaStore` problématique en tests parallèles | Flakiness potentielle | Moyenne |

---

## Vision architecturale — Axes d'évolution agentique

*Analyse des gaps entre l'architecture actuelle (v0.9) et un système multi-agent
"production-grade". À prendre en compte pour les versions futures.*

### Ce qui fonctionne bien

- **Orchestrateur vrai chef d'orchestre** : `run_pipeline_adaptive()` avec loop ReAct,
  plan conditionnel selon le profil du dataset, `reasoning_steps` tracés et exposés.
- **RAG actif dans la décision** (F25) : les seuils de qualité sont contextualisés par
  les règles ChromaDB — pas juste un LLM qui répond, mais un agent qui consulte sa mémoire.
- **Feedback loop** (F26) : le comportement évolue à partir des retours utilisateur
  (`confidence_adjustments`, règles d'exception auto). C'est du vrai apprentissage en ligne.
- **Classification sémantique batch** (F27) : 1 seul appel LLM pour N colonnes, vs N appels
  individuels — décision d'efficacité importante.

### Gaps identifiés pour aller plus loin

#### 1. Mémoire inter-sessions (priorité haute)

**Problème** : chaque analyse repart de zéro. L'orchestrateur ne sait pas que le même
dataset a été analysé 10 fois avec toujours les mêmes issues sur les mêmes colonnes.

**Évolution** : profil de dataset persisté par `dataset_id` + détection "ce dataset
ressemble à ceux du secteur RH" → activation automatique de règles sectorielles.

**Feature candidate** : `F30 — DatasetMemory` — clustering de profils par secteur,
suggestions pro-actives, baseline qualité par dataset récurrent.

#### 2. Auto-critique de l'orchestrateur (priorité moyenne)

**Problème** : l'orchestrateur ne remet jamais en question les résultats du `QualityAgent`.
Si le score semble incohérent (ex: 95/100 mais 3 CRITICAL), il finalise quand même.

**Évolution** : phase "Reflect" après "Act" dans le ReAct — vérification de cohérence
score/issues, re-planification si anomalie détectée, log de l'auto-correction.

**Feature candidate** : `F31 — ReAct Reflect` — 5ème phase dans `run_pipeline_adaptive()`
avec validation de cohérence et itération conditionnelle (max 2 iterations).

#### 3. Agents spécialisés par domaine (priorité basse → haute à terme)

**Problème** : 1 `QualityAgent` généraliste. Un dataset RH (colonnes `salary`, `hire_date`,
`department`) et un dataset e-commerce (`order_id`, `sku`, `cart_value`) passent par les
mêmes règles sans différenciation.

**Évolution** : agents spécialisés `HRQualityAgent`, `FinanceQualityAgent` avec règles
métier natives (cohérence salaire/ancienneté, contraintes RGPD sur les PII, etc.),
sélectionnés automatiquement via le `semantic_type` du `SemanticProfilerAgent`.

**Feature candidate** : `F32 — Domain Agents` — routing dans l'orchestrateur selon
`context.metadata["domain"]` détecté par F27.

#### 4. Planning dynamique multi-étapes (priorité basse)

**Problème** : `_build_execution_plan()` applique 6 règles fixes. Le nombre d'itérations
du pipeline est prédéfini, pas adaptatif à la complexité réelle.

**Évolution** : planificateur qui décide du nombre d'itérations selon la densité d'issues
et leur inter-dépendance (ex: corriger les doublons avant de recalculer les nulls).

**Feature candidate** : `F33 — Iterative Planner` — `max_iterations` configurable,
convergence détectée quand le score ne s'améliore plus entre deux passes.

---

## v0.2 — Features livrées

### Feature 1 — Upload CSV/Parquet ✅

**Motivation** : l'API acceptait uniquement du JSON. Les data scientists travaillent
en `.csv` / `.parquet`.

**Décision de design** : `POST /upload` déclenche l'analyse directement et retourne
un `AnalyzeResponse` complet avec `session_id`. Pas besoin d'un endpoint intermédiaire
sans analyse — ça maximise la valeur par appel.

**Fichiers créés/modifiés** :

- `src/api/routes/upload.py` — nouveau router
- `src/api/main.py` — enregistrement du router
- `requirements.txt` — `pyarrow>=14.0.0`
- `tests/integration/test_upload.py` — 6 tests

**Critères de validation** :

- [x] CSV valide → `AnalyzeResponse` avec `session_id`
- [x] Parquet valide → `AnalyzeResponse` avec `session_id`
- [x] Extension non supportée → HTTP 422 avec message explicite
- [x] Fichier trop grand → HTTP 413
- [x] Fichier vide (header only) → HTTP 400
- [x] 6/6 tests d'intégration passent

---

### Feature 2 — Métriques Prometheus ✅

**Motivation** : en production, il faut observer les appels et la latence.

**Décision de design** : `prometheus-fastapi-instrumentator` instrumente FastAPI
automatiquement en 3 lignes. Pas de code métier à modifier.
Expose `http_requests_total` et `http_request_duration_seconds` avec labels
`{method, handler, status_code}`.

**Fichiers modifiés** :

- `requirements.txt` — `prometheus-fastapi-instrumentator>=0.11.0`
- `src/api/main.py` — `Instrumentator().instrument(app).expose(app)`

**Critères de validation** :

- [x] `GET /metrics` retourne du texte format Prometheus
- [x] Les compteurs s'incrémentent après chaque appel API
- [x] Pas de test automatisé (middleware de config) — validé manuellement

---

### Feature 3 — Authentification JWT ✅

**Motivation** : l'API était entièrement ouverte.

**Décision de design** : auth **opt-in** (`AUTH_ENABLED=false` par défaut).
En dev, `get_current_user` retourne `{"user": "anonymous"}` sans vérifier le token
— les 66 tests existants ne cassent pas, pas besoin de fixtures avec token.
En prod, `AUTH_ENABLED=true` dans `.env` active la vérification sur tous les endpoints.

La protection est appliquée au niveau du `include_router` (via `dependencies=`) et non
endpoint par endpoint — une seule ligne par router, aucune modification des fichiers de routes.

**Fichiers créés/modifiés** :

- `src/api/auth.py` — `create_access_token`, `verify_token`, `get_current_user`
- `src/api/routes/auth.py` — `POST /auth/token` (form OAuth2)
- `src/core/config.py` — `auth_enabled`, `jwt_algorithm`, `jwt_expire_minutes`, `api_username`, `api_password`
- `src/api/main.py` — `dependencies=[Depends(get_current_user)]` sur chaque router protégé
- `requirements.txt` — `python-jose[cryptography]>=3.3.0`, `passlib[bcrypt]>=1.7.4`
- `tests/unit/test_auth.py` — 9 tests

**Critères de validation** :

- [x] Avec `AUTH_ENABLED=false` → toujours autorisé (dev)
- [x] Token expiré → HTTP 401
- [x] Token invalide → HTTP 401
- [x] Credentials incorrects → HTTP 401
- [x] `POST /auth/token` avec bons credentials → token Bearer valide
- [x] 9/9 tests unitaires passent

---

### Feature 4 — Persistance de session Redis ✅

**Motivation** : `GET /analyze/{session_id}` retournait 404 (TODO). Redis permet
de récupérer une session sans renvoyer les données.

**Décision de design** :

- **Fallback in-memory** si Redis est indisponible → zéro crash, même comportement pour l'appelant
- Sérialisation : `AgentContext.model_dump(mode="json")` → `model_validate()` (Pydantic v2)
- TTL configurable (`SESSION_TTL=3600` dans `.env`)
- Persistance **best-effort** : une exception dans `save()` ne bloque pas la réponse

**Fichiers créés/modifiés** :

- `src/memory/session_store.py` — `SessionStore` + `InMemoryFallback` + `get_session_store()`
- `src/core/config.py` — `redis_url`, `session_ttl`
- `src/api/routes/analyze.py` — `POST` sauvegarde le contexte, `GET /{id}` le charge
- `src/api/routes/upload.py` — sauvegarde après analyse
- `requirements.txt` — `redis>=5.0.0`, `fakeredis>=2.20.0`
- `tests/unit/test_session_store.py` — 10 tests

**Critères de validation** :

- [x] `POST /analyze` → sauvegarde le contexte sous `session_id`
- [x] `GET /analyze/{session_id}` → retourne les résultats sans renvoyer les données
- [x] Session absente / expirée → HTTP 404
- [x] TTL expiré → session introuvable
- [x] Redis indisponible → fallback in-memory transparent
- [x] 10/10 tests unitaires passent

---

## v0.3 — Features livrées

### Feature 5 — Parallélisation Quality Agent ✅

**Motivation** : le pipeline Quality exécutait 4 checks séquentiellement (missing values,
anomalies, type issues, rules). Ils sont indépendants → parallélisables.

**Décision de design** : `QualityAgent.execute_async()` utilise `asyncio.to_thread` +
`asyncio.gather` pour exécuter les 4 checks en concurrence.
`OrchestratorAgent.run_pipeline_async()` orchestre le pipeline entier en async (Profiler en
premier car Quality dépend du profil, puis Quality en parallèle).
Les endpoints FastAPI `/analyze` et `/upload` passent à `run_pipeline_async`.

**Fichiers modifiés** :

- `src/agents/quality.py` — `execute_async()`
- `src/agents/orchestrator.py` — `run_pipeline_async()`, `_run_quality_check_async()`
- `src/api/routes/analyze.py` — `await orchestrator.run_pipeline_async(...)`
- `src/api/routes/upload.py` — idem

**Critères de validation** :

- [x] `execute_async` détecte les mêmes types d'issues qu'`execute` (sync)
- [x] Aucune régression sur les 97 tests existants
- [x] 3/3 tests unitaires `test_parallel_quality.py` passent

---

### Feature 6 — Rate Limiting (`slowapi`) ✅

**Motivation** : sans rate limiting, l'API est exposée aux abus (scraping, DDoS applicatif).

**Décision de design** : `slowapi` avec `Limiter(key_func=get_remote_address)`.
Limites : 30 req/min sur `POST /analyze`, 10 req/min sur `POST /upload`.
Singleton `limiter` dans `src/api/limiter.py`, importé dans chaque router.
En cas de dépassement : HTTP 429.

**Fichiers créés/modifiés** :

- `src/api/limiter.py` — `Limiter` singleton
- `src/api/routes/analyze.py` — `@limiter.limit("30/minute")`
- `src/api/routes/upload.py` — `@limiter.limit("10/minute")`
- `src/api/main.py` — `app.state.limiter`, handler 429

**Critères de validation** :

- [x] `@limiter.limit` appliqué sur les 2 endpoints lourds
- [x] `RateLimitExceeded` → HTTP 429 (handler configuré)
- [x] Aucune régression sur les tests (limites suffisamment hautes)

---

### Feature 7 — Webhooks ✅

**Motivation** : un système d'analyse asynchrone doit pouvoir notifier des services tiers
(CI/CD, Slack, monitoring) quand une analyse se termine.

**Décision de design** : stockage in-memory (`_webhooks` dict dans `webhook_manager.py`).
`fire_webhooks()` est async, non-bloquant (BackgroundTasks FastAPI).
Un webhook qui échoue ne bloque pas les autres (`return_exceptions=True`).

**Fichiers créés/modifiés** :

- `src/core/webhook_manager.py` — `add_webhook`, `remove_webhook`, `fire_webhooks`
- `src/api/routes/webhooks.py` — `POST /webhooks`, `GET /webhooks`, `DELETE /webhooks/{id}`
- `src/api/routes/analyze.py` — `background_tasks.add_task(fire_webhooks, ...)`
- `src/api/routes/upload.py` — idem
- `src/api/main.py` — `app.include_router(webhooks_router.router)`
- `tests/unit/test_webhook_manager.py` — 9 tests
- `tests/integration/test_webhooks.py` — 6 tests

**Critères de validation** :

- [x] `POST /webhooks` → 201 avec `webhook_id`
- [x] `GET /webhooks` → liste les abonnés
- [x] `DELETE /webhooks/{id}` → 204
- [x] `fire_webhooks` envoie POST JSON à chaque abonné de l'événement
- [x] Webhook défaillant → logué, pas d'exception propagée
- [x] 9/9 tests unitaires + 6/6 tests d'intégration passent

---

### Feature 8 — Rapport PDF ✅

**Motivation** : les data scientists veulent exporter les résultats d'analyse pour les
partager en dehors de l'API.

**Décision de design** : `GET /analyze/{session_id}/report.pdf` charge la session depuis
le store et génère un PDF via `reportlab.platypus.SimpleDocTemplate`.
Import lazy de reportlab dans la fonction (évite l'ImportError si non installé).
Contenu : titre, infos session, score de qualité coloré, tableau des issues, profil.

**Fichiers modifiés** :

- `src/api/routes/analyze.py` — endpoint + `_generate_pdf()`
- `requirements.txt` — `reportlab>=4.0.0`
- `tests/integration/test_reports.py` — 5 tests

**Critères de validation** :

- [x] `GET /analyze/{session_id}/report.pdf` → HTTP 200, `Content-Type: application/pdf`
- [x] Contenu commence par `%PDF` (signature valide)
- [x] Taille > 1 KB
- [x] Session inconnue → HTTP 404
- [x] 5/5 tests d'intégration passent

---

### Feature 9 — Interface Streamlit ✅

**Motivation** : visualiser les résultats sans Swagger/curl.

**Décision de design** : application Streamlit standalone (`streamlit_app.py` à la racine).
Appelle l'API via HTTP (`requests`). Ne dépend d'aucun module interne.
Lancement : `streamlit run streamlit_app.py` (API FastAPI séparée requise).

**Fichiers créés** :

- `streamlit_app.py` — upload, analyse, onglets (Problèmes / Profil / PDF / JSON)
- `requirements.txt` — `streamlit>=1.28.0`, `requests>=2.31.0`

**Critères de validation** :

- [x] Upload CSV → appelle `POST /upload` → affiche score de qualité coloré
- [x] Onglet Problèmes → tableau trié par sévérité
- [x] Onglet Profil → stats par colonne
- [x] Onglet PDF → bouton de téléchargement
- [x] Sidebar health check → vérifie la connexion à l'API

---

## v0.4 — Features livrées

### Feature 10 — Détection des lignes dupliquées (Q1) ✅

**Motivation** : un dataset peut contenir des lignes identiques dupliquées (import en double,
fusion de fichiers). C'est un problème de qualité qui fausse les statistiques.

**Décision de design** : `df.duplicated(keep=False)` marque toutes les occurrences, pas seulement
la seconde. Sévérité proportionnelle au % de doublons : <1% LOW, 1-10% MEDIUM, >10% HIGH.
Confidence=1.0 (détection déterministe).

**Fichiers modifiés** :
- `src/agents/quality.py` — `_detect_duplicate_rows()`, wiring dans `execute()` et `execute_async()`
- `tests/unit/test_quality_v4.py` — 6 tests

**Critères de validation** :
- [x] Aucun doublon → liste vide
- [x] Sévérité LOW / MEDIUM / HIGH selon le %
- [x] DataFrame d'une ligne → aucune issue
- [x] `affected_count` correct

---

### Feature 11 — Détection des pseudo-nulls (Q2) ✅

**Motivation** : des valeurs comme "N/A", "null", "-", "unknown" masquent des données manquantes.
Elles ne sont pas détectées par `df.isna()`.

**Décision de design** : liste fermée de 20 tokens normalisés (lowercase + strip).
Appliqué uniquement aux colonnes `object`. Confidence=0.92.

**Fichiers modifiés** :
- `src/agents/quality.py` — `_PSEUDO_NULL_VALUES`, `_detect_pseudo_nulls()`
- `tests/unit/test_quality_v4.py` — 7 tests

**Critères de validation** :
- [x] "N/A", "null", "none", "-", "MISSING", "UNKNOWN" détectés
- [x] Insensible à la casse
- [x] Colonnes numériques ignorées
- [x] NaN pandas non comptés
- [x] Sévérité correcte

---

### Feature 12 — Validation de format (Q3) ✅

**Motivation** : des colonnes email/téléphone/CP peuvent contenir des valeurs mal formées.
La validation par nom de colonne est rapide et ne nécessite pas de LLM.

**Décision de design** : 6 formats supportés (email, phone FR, URL, code postal, SIRET, SIREN).
Détection par nom de colonne (keywords). Seuil 5% pour éviter les faux positifs sur exceptions
légitimes. Confidence=0.87.

**Fichiers modifiés** :
- `src/agents/quality.py` — `_FORMAT_PATTERNS`, `_detect_format_issues()`
- `tests/unit/test_quality_v4.py` — 6 tests

**Critères de validation** :
- [x] Emails invalides détectés si >5% invalides
- [x] Codes postaux invalides détectés
- [x] Colonne sans keyword correspondant ignorée
- [x] Pseudo-nulls exclus du check format
- [x] Sévérité correcte (>50% HIGH, 20-50% MEDIUM, 5-20% LOW)

---

### Feature 13 — Score de qualité par colonne (Q4) ✅

**Motivation** : le score global masque quelles colonnes sont dégradées.
Un score par colonne permet de prioriser les corrections.

**Décision de design** : déductions sur 100 par sévérité : CRITICAL=-40, HIGH=-25,
MEDIUM=-12, LOW=-5. Floor à 0. Issues sans colonne (ex: doublons) ignorées.

**Fichiers modifiés** :
- `src/agents/quality.py` — `_compute_column_scores()`
- `src/api/schemas/responses.py` — champ `column_scores: dict[str, float]`
- `src/api/routes/analyze.py` — `column_scores=context.metadata.get("column_scores", {})`
- `src/api/routes/upload.py` — idem
- `tests/unit/test_quality_v4.py` — 5 tests

**Critères de validation** :
- [x] Aucune issue → toutes les colonnes à 100.0
- [x] Déductions correctes par sévérité
- [x] Floor à 0 (pas de score négatif)
- [x] Issue sans colonne n'affecte pas les scores
- [x] `column_scores` présent dans `AnalyzeResponse`

---

### Feature 14 — Plan de corrections (`GET /analyze/{id}/corrections`) ✅

**Motivation** : après une analyse, l'utilisateur veut savoir quoi corriger et dans quel ordre.

**Décision de design** : endpoint RESTful sans recalcul. Charge la session depuis le store
et classe les issues en 2 listes : `auto_corrections` (applicable automatiquement) et
`manual_reviews`. Score estimé après application des corrections auto.

**Fichiers modifiés** :
- `src/api/routes/analyze.py` — endpoint + `_AUTO_ACTIONS` mapping
- `tests/integration/test_corrections.py` — 9 tests

**Critères de validation** :
- [x] Session valide → HTTP 200, JSON structuré
- [x] Session inconnue → HTTP 404
- [x] MISSING_VALUES, DUPLICATE, FORMAT_ERROR → auto
- [x] ANOMALY, CONSTRAINT_VIOLATION → manual
- [x] `estimated_score_after_auto` ≥ `quality_score` et ≤ 100
- [x] Chaque entrée contient les champs requis

---

### Feature 15 — Export Excel (`GET /analyze/{id}/report.xlsx`) ✅

**Motivation** : le PDF est bon pour partager, mais l'Excel permet la manipulation des données.

**Décision de design** : 4 onglets (`openpyxl`) : Résumé, Issues (couleurs sévérité),
Profil colonnes, Score par colonne (coloré par niveau). Import lazy (`openpyxl`) pour
ne pas crasher si la lib n'est pas installée (HTTP 503 à la place).

**Fichiers modifiés** :
- `src/api/routes/analyze.py` — endpoint + `_generate_excel()`
- `requirements.txt` — `openpyxl>=3.1.0`
- `tests/integration/test_reports.py` — 5 tests ajoutés (classe `TestXlsxReport`)

**Critères de validation** :
- [x] HTTP 200, Content-Type Excel
- [x] Contenu commence par `PK` (signature ZIP valide)
- [x] Taille > 1 KB
- [x] Session inconnue → HTTP 404

---

### Feature 16 — Persistance JSON des webhooks ✅

**Motivation** : les webhooks enregistrés étaient perdus au redémarrage (stockage in-memory).

**Décision de design** : sérialisation JSON dans `./data/webhooks.json`.
`_load_from_disk()` est appelé au niveau module (démarrage automatique).
`_save_to_disk()` est appelé à chaque `add_webhook` et `remove_webhook`.
Erreurs de lecture/écriture : logguées, non propagées (best-effort).

**Fichiers modifiés** :
- `src/core/webhook_manager.py` — `_PERSIST_PATH`, `_load_from_disk()`, `_save_to_disk()`
- `tests/unit/test_webhook_manager.py` — 5 tests ajoutés (classe `TestJsonPersistence`)

**Critères de validation** :
- [x] `add_webhook` → fichier JSON écrit
- [x] `remove_webhook` → fichier JSON mis à jour
- [x] `_load_from_disk` → webhooks rechargés depuis le fichier
- [x] Fichier inexistant → pas d'exception
- [x] Fichier corrompu → pas d'exception

---

---

## v0.5 — Features livrées

### Feature 17 — Application des corrections (`POST /analyze/{id}/apply-corrections`) ✅

**Motivation** : après détection et planification, l'utilisateur veut télécharger
un fichier propre directement depuis l'interface.

**Décision de design** :

- Le DataFrame original est stocké en base64+parquet dans le session store (même TTL que la session)
- Corrections appliquées : DUPLICATE → `drop_duplicates`, MISSING_VALUES → pseudo-nulls → NaN + imputation (médiane/mode), FORMAT_ERROR → `str.strip()`, TYPE_MISMATCH → `pd.to_numeric`
- Retourne un CSV `StreamingResponse` avec les headers `X-Rows-Before`, `X-Rows-After`, `X-Corrections-Count`
- Si les données originales ont expiré → HTTP 422 explicatif
- Chaque colonne est traitée une seule fois (set `processed` pour éviter les doublons)

**Fichiers créés/modifiés** :

- `src/memory/session_store.py` — `save_dataframe()`, `load_dataframe()`, `delete()` nettoie les deux clés
- `src/api/routes/analyze.py` — endpoint + `_PSEUDO_NULL_TOKENS`; sauvegarde df dans `POST /analyze`
- `src/api/routes/upload.py` — sauvegarde df après pipeline
- `streamlit_app.py` — bouton "Appliquer et télécharger le CSV corrigé" dans l'onglet Corrections
- `tests/integration/test_apply_corrections.py` — 8 tests
- `tests/unit/test_session_store.py` — 5 tests ajoutés (`TestSaveLoadDataframe`)

**Critères de validation** :

- [x] Session valide + DataFrame disponible → HTTP 200, CSV correct
- [x] Doublons supprimés (`drop_duplicates`)
- [x] Nulls numériques imputés par la médiane
- [x] Pseudo-nulls (N/A, null, -) remplacés avant imputation
- [x] Session inconnue → HTTP 404
- [x] DataFrame expiré → HTTP 422 avec message explicatif
- [x] Headers `X-Rows-Before`, `X-Rows-After`, `X-Corrections-Count` présents
- [x] Aucune correction auto → CSV retourné sans modification

---

### Feature 18 — Analyse en lot (`POST /batch`) ✅

**Motivation** : les data engineers traitent des dizaines de fichiers par jour. Analyser
fichier par fichier est lent et répétitif.

**Décision de design** :

- Maximum 10 fichiers par requête (protection contre les abus)
- `asyncio.gather` pour l'analyse parallèle — chaque fichier passe par le pipeline complet
- Si un fichier échoue (extension invalide, CSV illisible, vide), les autres continuent
- Chaque résultat contient `filename`, `session_id`, `quality_score`, `issues_count`, `status`, `error`
- Rate limit : 5 req/min (plus strict car chaque requête peut lancer 10 analyses)
- Persistance best-effort : DataFrame + contexte sauvegardés pour chaque fichier réussi

**Fichiers créés/modifiés** :

- `src/api/routes/batch.py` — nouveau router `POST /batch`
- `src/api/schemas/responses.py` — `BatchResultItem`, `BatchAnalyzeResponse`
- `src/api/main.py` — `app.include_router(batch.router)`, version `"0.5.0"`
- `streamlit_app.py` — onglet Batch avec `accept_multiple_files=True`
- `tests/integration/test_batch.py` — 8 tests + fixture `reset_rate_limiter`

**Critères de validation** :

- [x] 1 fichier CSV → HTTP 200, 1 résultat, status="success"
- [x] Chaque résultat contient `filename`, `status`, `quality_score`, `issues_count`, `session_id`
- [x] 3 fichiers → 3 résultats
- [x] Aucun fichier → HTTP 400 ou 422
- [x] Extension invalide → résultat en erreur, pas d'arrêt global
- [x] `session_ids` uniques pour chaque fichier réussi
- [x] `quality_score` entre 0 et 100
- [x] `succeeded + failed == total`

---

## v0.6 — Features livrées

> Thème : **Intelligibilité et pilotage** — rendre le système plus transparent,
> gérable et observable sans toucher au code source.

### Feature 19 — Comparaison avant/après corrections (`GET /analyze/{id}/comparison`)

**Motivation** : après `POST /analyze/{id}/apply-corrections`, l'utilisateur veut
mesurer concrètement l'amélioration : quel score avant, quel score après, quelles
issues ont disparu. Actuellement il faut lancer manuellement une deuxième analyse.

**Décision de design** :

- L'endpoint charge le DataFrame corrigé depuis la session (stocké après apply-corrections
  sous une clé `df_corrected:{session_id}`) ou applique les corrections à la volée depuis le DataFrame original.
- Relance le pipeline Quality uniquement (pas de reprofilage complet) sur le DataFrame corrigé
  → score_after + issues_after.
- Retourne un objet de comparaison : `score_before`, `score_after`, `delta`, `issues_removed`,
  `issues_remaining`, `columns_improved`.
- Pas de persistance supplémentaire : stateless, calcul à la demande.

**Fichiers à créer/modifier** :

- `src/api/routes/analyze.py` — `GET /{session_id}/comparison`
- `src/api/schemas/responses.py` — `ComparisonResponse` avec before/after/delta
- `streamlit_app.py` — bouton "Voir l'amélioration" dans l'onglet Corrections
- `tests/integration/test_comparison.py` — 6 tests

**Critères de validation** :

- [ ] Session valide + DataFrame disponible → HTTP 200, `score_after` ≥ `score_before`
- [ ] `delta = score_after - score_before` (peut être 0 si aucune issue auto)
- [ ] `issues_removed` liste les issue_types qui ont disparu
- [ ] `issues_remaining` liste les issues persistantes
- [ ] Session inconnue → HTTP 404
- [ ] DataFrame non disponible → HTTP 422

---

### Feature 20 — CRUD règles métier (`GET/POST/DELETE /rules`)

**Motivation** : les règles ChromaDB s'ajoutent actuellement en JSON ou via le champ
`custom_rules` de `POST /analyze`. Il n'y a aucun moyen de les lister, les supprimer
ou les modifier via l'API.

**Décision de design** :

- `GET /rules` → liste toutes les règles actives (id, texte, type, sévérité, catégorie)
- `POST /rules` → ajoute une règle (même schéma qu'`AddRuleRequest`)
- `DELETE /rules/{rule_id}` → supprime une règle par ID
- La collection ChromaDB est la source de vérité (pas de base SQL supplémentaire)
- Rate limit : 60 req/min sur GET, 20 req/min sur POST/DELETE (modificateurs)

**Fichiers à créer/modifier** :

- `src/api/routes/rules.py` — nouveau router `/rules`
- `src/api/main.py` — `app.include_router(rules.router)`
- `streamlit_app.py` — onglet "📋 Règles" avec formulaire CRUD
- `tests/integration/test_rules.py` — 8 tests

**Critères de validation** :

- [ ] `GET /rules` → liste les règles (liste vide si aucune)
- [ ] `POST /rules` avec texte valide → 201 Created, règle listée ensuite
- [ ] `DELETE /rules/{id}` existant → 204 No Content
- [ ] `DELETE /rules/{id}` inconnu → 404
- [ ] Règle ajoutée via Streamlit → visible dans `GET /rules`
- [ ] Règle supprimée → n'apparaît plus dans `GET /rules`
- [ ] Règle active influence le résultat de `POST /analyze` (constraint_violation)

---

### Feature 21 — Jobs asynchrones pour gros fichiers (`POST /jobs/analyze`)

**Motivation** : pour les fichiers > 50 000 lignes, l'analyse dépasse souvent le timeout
HTTP de 30 s (Streamlit, proxies). Un système de jobs permet un traitement non-bloquant.

**Décision de design** :

- `POST /jobs/analyze` — accepte un fichier (comme `/upload`), retourne immédiatement
  un `job_id` + statut `pending`.
- Traitement via `asyncio.create_task` dans le même process (pas de Celery pour garder
  le déploiement simple). Adapté pour des workloads < 500 k lignes ; pour plus, documenter
  le passage à ARQ/Celery.
- `GET /jobs/{job_id}` → `{status: pending|running|completed|failed, progress, result?}`
- Le résultat final est stocké dans le session store sous `job:{job_id}` (TTL 2h).
- Seuil de déclenchement asynchrone : configurable (`ASYNC_THRESHOLD_ROWS=10000` dans `.env`).

**Fichiers à créer/modifier** :

- `src/api/routes/jobs.py` — router `/jobs`
- `src/core/config.py` — `async_threshold_rows: int = 10000`
- `src/api/main.py` — `app.include_router(jobs.router)`
- `streamlit_app.py` — polling du statut du job avec `st.status` + barre de progression
- `tests/integration/test_jobs.py` — 6 tests

**Critères de validation** :

- [ ] `POST /jobs/analyze` → HTTP 202 Accepted, `job_id` + status=pending
- [ ] `GET /jobs/{job_id}` immédiatement → status=pending ou running
- [ ] `GET /jobs/{job_id}` après completion → status=completed, `result` présent
- [ ] `GET /jobs/{unknown}` → HTTP 404
- [ ] Fichier invalide → job passe en status=failed avec `error`
- [ ] Streamlit affiche la progression et le résultat final

---

### Feature 22 — Tableau de bord analytique (`GET /stats`)

**Motivation** : en production, on veut suivre l'activité du système : combien d'analyses
lancées, quel score moyen, quels types d'issues sont les plus fréquents, tendance
dans le temps.

**Décision de design** :

- `GET /stats` → agrège les métriques depuis le session store + un compteur en mémoire
  (incrément à chaque analyse réussie, persisté en JSON comme les webhooks).
- Métriques exposées : `total_sessions`, `avg_quality_score`, `top_issue_types` (top 5),
  `sessions_by_day` (7 derniers jours), `score_distribution` (buckets 0-20/20-40/40-60/60-80/80-100).
- Pas de base time-series (pas d'InfluxDB) : compteurs JSON simple dans `./data/stats.json`.
- Réinitialisation via `DELETE /stats` (admin only quand `auth_enabled=True`).

**Fichiers à créer/modifier** :

- `src/core/stats_manager.py` — `StatsManager` avec `record_session()`, `get_stats()`, persistance JSON
- `src/api/routes/stats.py` — `GET /stats`, `DELETE /stats`
- `src/api/main.py` — `app.include_router(stats.router)`
- `src/api/routes/analyze.py` + `upload.py` + `batch.py` — appel `stats_manager.record_session()` après pipeline
- `streamlit_app.py` — onglet "📊 Stats" avec graphiques agrégés
- `tests/unit/test_stats_manager.py` — 6 tests

**Critères de validation** :

- [ ] `GET /stats` après 0 analyse → `total_sessions=0`
- [ ] `GET /stats` après 3 analyses → `total_sessions=3`, `avg_quality_score` cohérent
- [ ] `top_issue_types` liste les 5 types les plus détectés
- [ ] `score_distribution` somme à `total_sessions`
- [ ] `DELETE /stats` remet les compteurs à zéro
- [ ] Données persistées entre redémarrages (`./data/stats.json`)

---

## v0.7 — Features livrées

> Thème : **Intelligence agentique réelle** — faire entrer le LLM dans la boucle
> de décision, rendre l'orchestrateur adaptatif, activer le RAG sur les seuils,
> et fermer la boucle feedback pour que le système s'améliore à chaque correction.
>
> C'est le passage de *"pipeline déterministe avec infrastructure agentique"*
> à un *"vrai système agentique qui raisonne, s'adapte et apprend"*.

---

### Feature 23 — LLM Quality Check (Claude pour les cas ambigus)

**Motivation** : les checks heuristiques actuels (regex, pandas, IQR) ne couvrent pas
les anomalies sémantiques — une valeur texte syntaxiquement valide mais métier incohérente
(ex: `age=250`, `pays="Marteau"`, `email="directeur"` sans `@`). Seul un LLM comprend
le sens des données.

**Décision de design** :

- Nouveau check `_detect_semantic_anomalies_llm()` dans `QualityAgent`, déclenché pour
  les colonnes `object` non couvertes par les checks heuristiques ET si `ENABLE_LLM_CHECKS=true`.
- Pattern exact : sample 20 valeurs → appel Claude via function calling →
  le modèle appelle `flag_anomaly(value, reason, severity)` pour chaque anomalie détectée.
- Opt-in strict : `ENABLE_LLM_CHECKS=false` par défaut → zéro régression sur les 188 tests existants.
  En prod, `ENABLE_LLM_CHECKS=true` dans `.env` + `ANTHROPIC_API_KEY` configuré.
- Garde-fous : timeout 10s, max 1 appel LLM par colonne, max 5 colonnes par analyse,
  fallback silencieux si API indisponible (le check est simplement ignoré, pas d'exception).
- `detected_by=AgentType.QUALITY`, `confidence` = score retourné par le LLM (0.6–0.95),
  `details={"detection_method": "llm", "model": "claude-haiku-4-5"}`.
- Le modèle utilisé est `claude-haiku-4-5` (rapide, économique) sauf override `LLM_CHECK_MODEL`.

**Prompt pattern** :

```text
Tu analyses la colonne "{col_name}" (type: {dtype}, contexte: {col_context}).
Identifie les valeurs sémantiquement anormales ou métier-incohérentes.
Valeurs : {sample_values}
Pour chaque anomalie, appelle flag_anomaly() avec la valeur, la raison et la sévérité.
Ne signale que les vrais problèmes (confiance > 0.7). Ignore les valeurs normales.
```

**Fichiers à créer/modifier** :

- `src/agents/quality.py` — `_detect_semantic_anomalies_llm()`, tool `flag_anomaly`
- `src/core/config.py` — `enable_llm_checks: bool = False`, `llm_check_model: str = "claude-haiku-4-5-20251001"`
- `tests/unit/test_quality_v7.py` — 6 tests (mock de l'appel LLM via `unittest.mock`)

**Critères de validation** :

- [ ] `ENABLE_LLM_CHECKS=false` → aucun appel API, comportement identique à v0.6
- [ ] `ENABLE_LLM_CHECKS=true` + mock LLM → issues détectées avec `detected_by=QUALITY`
- [ ] API LLM indisponible (mock exception) → fallback silencieux, pipeline continue
- [ ] Timeout dépassé → fallback silencieux, pas de crash
- [ ] `confidence` retournée par le LLM propagée dans l'issue
- [ ] Max 5 colonnes traitées par analyse (protection coût)

---

### Feature 24 — Orchestrateur adaptatif (ReAct loop)

**Motivation** : l'orchestrateur actuel est un pipeline linéaire fixe :
Profiler → Quality → Corrector → Validator, toujours dans cet ordre, toujours tous les checks.
Un vrai orchestrateur agentique observe les résultats intermédiaires et adapte son plan.

**Décision de design** :

- Nouveau `run_pipeline_adaptive()` dans `OrchestratorAgent`, basé sur le pattern ReAct
  (Reason → Act → Observe → Repeat).
- **Phase 1 — Observe** : Profiler s'exécute, résultat dans le contexte.
- **Phase 2 — Reason** : l'orchestrateur analyse le profil et construit un plan :

  | Condition observée | Action adaptée |
  | --- | --- |
  | `row_count < 30` | Skip `AnomalyDetector` (Isolation Forest instable) |
  | `total_null_count == 0` | Skip check `MISSING_VALUES` |
  | `null_percentage > 50%` sur toutes colonnes | Skip checks format (inutile sur colonnes vides) |
  | `column_count > 100` | Mode sampling : top 20 colonnes par `null_count` |
  | Aucune colonne numérique | Skip outlier + drift detection |
  | Colonne temporelle détectée | Active `DriftDetector` sur ces colonnes |

- **Phase 3 — Act** : exécute uniquement les checks du plan (via `asyncio.gather` sélectif).
- **Phase 4 — Observe** : si un check lève un `CRITICAL`, re-raisonne : faut-il relancer
  un check complémentaire ?
- Le `reasoning_log` (liste de `{step, thought, action, observation}`) est stocké dans
  `context.metadata["reasoning_steps"]` et exposé dans `AnalyzeResponse` si
  `include_reasoning=true` dans la requête.
- `run_pipeline_async()` reste inchangé (rétrocompatibilité — les endpoints existants
  n'utilisent pas encore le mode adaptatif).

**Fichiers à créer/modifier** :

- `src/agents/orchestrator.py` — `run_pipeline_adaptive()`, `_build_execution_plan()`, `_observe_and_replan()`
- `src/api/schemas/requests.py` — `include_reasoning: bool = False` dans `AnalyzeRequest`
- `src/api/schemas/responses.py` — `reasoning_steps: list[dict] = []` dans `AnalyzeResponse`
- `src/api/routes/analyze.py` — passe `include_reasoning` au pipeline
- `tests/unit/test_orchestrator_v7.py` — 8 tests

**Critères de validation** :

- [ ] Dataset avec `row_count < 30` → `reasoning_steps` contient "skip AnomalyDetector"
- [ ] Dataset sans nulls → check MISSING_VALUES non lancé
- [ ] Dataset > 100 colonnes → seules les 20 colonnes les plus dégradées analysées
- [ ] Dataset avec colonne date → DriftDetector activé automatiquement
- [ ] `include_reasoning=false` → `reasoning_steps=[]` (pas de surcoût)
- [ ] Plan adaptatif ≡ résultats cohérents avec le plan fixe sur datasets standard (pas de régression)
- [ ] Un check CRITICAL → re-raisonnement loggé dans `reasoning_steps`

---

### Feature 25 — RAG actif dans la boucle de décision

**Motivation** : ChromaDB stocke des règles métier mais elles ne sont consultées qu'au
moment du `ValidatorAgent`, après que les issues ont déjà été générées. Le RAG doit
intervenir *avant* chaque check pour ajuster les seuils — une règle "email obligatoire"
doit rendre la sévérité CRITICAL dès 1% de nulls, pas MEDIUM à 10%.

**Décision de design** :

- Nouveau `ChromaStore.get_relevant_rules(col_name, col_type, sample_values, top_k=3)` :
  query par similarité sur le nom de colonne + type → retourne les règles les plus proches.
- `RuleContext` dataclass dans `src/core/models.py` :

  ```python
  @dataclass
  class RuleContext:
      rules: list[str]          # textes des règles matchées
      null_threshold_override: float | None    # None = seuils par défaut
      severity_override: Severity | None
      format_tolerance_override: float | None  # None = 5%
  ```

- Avant chaque check colonne dans `QualityAgent`, appel `_get_rule_context(col_name)` :
  parse les règles retournées et extrait les overrides via keywords :
  - "obligatoire" / "required" / "non null" → `null_threshold_override = 0.01`
  - "identifiant unique" / "clé primaire" → `severity_override = CRITICAL` pour DUPLICATE
  - "format strict" → `format_tolerance_override = 0.0`
  - "optionnel" / "nullable" → `null_threshold_override = 0.8`
- Les règles actives sont loggées dans `issue.details["applied_rules"]`.
- Le `ValidatorAgent` utilise le RAG pour détecter les `CONSTRAINT_VIOLATION` :
  query ChromaDB avec le profil de la colonne → si une règle matchée est violée → issue.

**Fichiers à créer/modifier** :

- `src/memory/chroma_store.py` — `get_relevant_rules(col_name, col_type, sample_values)`
- `src/core/models.py` — `RuleContext` dataclass
- `src/agents/quality.py` — `_get_rule_context()`, integration dans chaque check
- `src/agents/validator.py` — RAG-based constraint violation detection
- `tests/unit/test_rag_active.py` — 7 tests

**Critères de validation** :

- [ ] Règle "email obligatoire" dans ChromaDB → colonne `email` avec 5% nulls → CRITICAL (pas MEDIUM)
- [ ] Règle "identifiant unique" → doublon → CRITICAL (pas LOW/MEDIUM)
- [ ] Colonne sans règle correspondante → seuils par défaut inchangés
- [ ] `issue.details["applied_rules"]` liste les règles qui ont influencé la détection
- [ ] RAG query échoue → fallback sur seuils par défaut, pas d'exception
- [ ] Performance : query ChromaDB < 50ms (index vectoriel, pas de scan full)

---

### Feature 26 — Feedback qui améliore le comportement

**Motivation** : `POST /feedback` enregistre les retours mais ne modifie rien.
C'est une boîte noire. Un système agentique apprend : un faux positif répété
doit baisser la sensibilité du check concerné ; une correction confirmée doit
enrichir les règles pour les analyses futures.

**Décision de design** :

- `FeedbackProcessor` dans `src/core/feedback_processor.py`, appelé depuis `POST /feedback`
  après persistance du feedback dans ChromaDB.
- **`was_correct=False` (fausse alerte)** :
  1. Écrit une règle d'exception dans ChromaDB :
     `"Colonne '{col}' : la valeur '{sample}' est normale, ne pas signaler comme {issue_type}"`
  2. Incrémente `false_positive_stats[check_type]` dans `./data/feedback_stats.json`
  3. Si `false_positive_stats[check_type] > 5` → `confidence_adjustments[check_type] -= 0.05`
     (plancher à 0.5 — le check ne disparaît jamais, devient juste moins affirmatif)
- **`was_correct=True` (confirmation)** :
  1. `confidence_adjustments[check_type] += 0.02` (plafonné à 0.99)
  2. Renforce la règle si une règle ChromaDB est liée au check
- **`was_correct=None` + `correction=custom_value`** :
  1. Ajoute l'exemple comme règle positive : `"Dans colonne '{col}', la valeur correcte est '{correction}'"`
  2. Enrichit ChromaDB pour les prochaines validations RAG
- `confidence_adjustments` chargés au démarrage de `QualityAgent` depuis `./data/feedback_stats.json`
  → les seuils de confiance deviennent dynamiques entre les redémarrages.
- `GET /stats` expose `feedback_summary` : `{false_positives_corrected, confirmations, examples_added, checks_adjusted}`.

**Fichiers à créer/modifier** :

- `src/core/feedback_processor.py` — `FeedbackProcessor` avec `process()`, `_update_confidence()`, `_write_exception_rule()`
- `src/agents/quality.py` — `_load_confidence_adjustments()` au `__init__`, utilisation dans les seuils
- `src/api/routes/feedback.py` — appelle `FeedbackProcessor.process()` après persistance
- `./data/feedback_stats.json` — fichier de persistance des ajustements (créé automatiquement)
- `tests/unit/test_feedback_processor.py` — 8 tests

**Critères de validation** :

- [ ] Feedback `was_correct=False` sur `MISSING_VALUES` × 6 → `confidence_adjustments["missing_values"]` baisse de 0.30
- [ ] Feedback `was_correct=True` × 3 → `confidence_adjustments["missing_values"]` remonte de 0.06
- [ ] Règle d'exception écrite dans ChromaDB après fausse alerte
- [ ] `confidence_adjustments` persistés dans `feedback_stats.json`
- [ ] `QualityAgent` chargé après feedbacks → seuils mis à jour (pas les seuils hardcodés)
- [ ] Plancher à 0.5 et plafond à 0.99 respectés
- [ ] `GET /stats` retourne `feedback_summary` cohérent
- [ ] Feedback `correction=custom_value` → règle positive dans ChromaDB

---

### Architecture v0.7 — Vue d'ensemble de la boucle intelligente

```text
POST /upload → DataFrame
       │
       ▼
 OrchestratorAgent.run_pipeline_adaptive()
       │
       ├─ [Observe] ProfilerAgent → DataProfile
       │
       ├─ [Reason]  _build_execution_plan(profile)
       │            → skip inutiles, active LLM si ENABLE_LLM_CHECKS
       │
       ├─ [Act]     QualityAgent (checks sélectifs en parallèle)
       │            │
       │            ├─ _get_rule_context(col) → ChromaDB RAG query
       │            │   → seuils dynamiques selon règles métier
       │            │
       │            ├─ checks heuristiques (nulls, formats, doublons…)
       │            │
       │            └─ _detect_semantic_anomalies_llm() [si activé]
       │                → Claude function calling → issues sémantiques
       │
       ├─ [Observe] Issues collectées → si CRITICAL → re-plan ?
       │
       ├─ [Act]     CorrectorAgent → propositions
       │
       └─ [Act]     ValidatorAgent (RAG-based constraint check)
                    → ChromaDB query → violations détectées

POST /feedback → FeedbackProcessor
       │
       ├─ Fausse alerte → règle d'exception ChromaDB + confidence_adjustments--
       ├─ Confirmation  → confidence_adjustments++
       └─ Correction    → règle positive ChromaDB
              │
              └─ [Prochain démarrage] QualityAgent charge confidence_adjustments
                 → seuils dynamiques, système qui apprend
```

---

## v1.0 — Features livrées

### Feature 32 — Custom Domain Agent Builder ✅

**Motivation** : La pipeline est généraliste. Un dataset RH (salary, employee_id) et un
dataset e-commerce (sku, cart_value) passent par les mêmes règles sans différenciation métier.
L'utilisateur veut nommer un domaine, lui associer des types sémantiques déclencheurs, des
champs requis, des règles descriptives et des overrides de sévérité — le tout configurable
via une interface Streamlit.

**Décision de design** :

- `DomainProfile` dataclass : `name`, `trigger_types` (types sémantiques déclencheurs),
  `min_match_ratio` (seuil de détection, ex. 0.3), `required_types` (colonnes obligatoires),
  `rules` (règles textuelles), `severity_overrides` (upgrade de sévérité par type sémantique)
- `DomainManager` singleton : CRUD persisté dans `./data/domain_agents.json`
- **Détection automatique** : après F27, `detect_domain(semantic_types)` calcule pour chaque
  profil actif `ratio = len(col_types ∩ trigger_types) / len(trigger_types)`. Profil avec
  le meilleur ratio ≥ `min_match_ratio` est activé.
- `_validate_domain_rules()` dans `QualityAgent` : colonnes requises manquantes →
  `CONSTRAINT_VIOLATION CRITICAL`, severity_overrides → upgrade des issues existantes,
  règles textuelles → `context.metadata["domain_rules"]`
- `AnalyzeResponse.domain_agent: str | None` — nom du domaine activé ou None
- Orchestrateur : `detect_domain()` appelé après `_run_semantic_enrichment` dans les
  3 pipelines (sync, async, adaptive)

**Fichiers créés/modifiés** :

- `src/core/domain_manager.py` — `DomainProfile`, `DomainRule`, `DomainManager` singleton
- `src/api/routes/domain_agents.py` — CRUD : GET/POST/DELETE /domain-agents
- `src/agents/orchestrator.py` — `_detect_domain()` dans les 3 pipelines
- `src/agents/quality.py` — `_validate_domain_rules()` en fin de `execute()` et `execute_async()`
- `src/api/main.py` — inclusion du router `domain_agents`
- `src/api/schemas/responses.py` — champ `domain_agent: str | None`
- `streamlit_app.py` — badge domaine actif + onglet "🏢 Agents Métier"

**Critères de validation** :

- [x] POST /domain-agents crée un profil, GET liste, DELETE supprime
- [x] Dataset avec types sémantiques correspondants → `response.domain_agent == "RH"`
- [x] Type requis absent → CONSTRAINT_VIOLATION CRITICAL ajoutée
- [x] severity_overrides → issues upgradées en sévérité
- [x] 0 régression sur les 288 tests existants (domain_id absent → no-op immédiat)
- [x] Onglet Streamlit "Agents Métier" : créer/lister/supprimer des agents

---

## v1.1 — Features livrées

### Feature 27v2 — SemanticProfiler heuristique (sans LLM) ✅

**Motivation** : F27v1 reposait entièrement sur Claude — si `ENABLE_LLM_CHECKS=false`,
`semantic_types` restait vide, rendant F28 (validators) et F32 (domain detection)
totalement inopérants. De plus, le pipeline synchrone (`run_pipeline`) n'appelait jamais
`enrich_async`, créant une asymétrie invisible entre les deux pipelines.

**Problèmes résolus** :

1. `semantic_types` toujours vide sans LLM → F32 et F28 non-fonctionnels hors LLM
2. Pipeline sync sans enrichissement sémantique → domain detection jamais actif en sync
3. LLM seule source de vérité — aucun fallback partiel sur erreur ou timeout
4. Perte totale de la classification si LLM échoue

**Décision de design** :

- **Classifieur heuristique** (`_heuristic_classify`) : pure Python, aucune dépendance LLM
  - Regex sur les valeurs : email, phone, url, postal_code, ip_address
  - Keywords sur noms de colonnes : 24 types sémantiques mappés (`salary` → `monetary_amount`)
  - Word-boundary matching (`_is_keyword_match`) : mots isolés, évite "stage" → "age"
  - Ranges numériques (age: [0,150], percentage: [0,100], rating: [0,10])
  - Cardinalité faible → `category`, chaînes longues → `description`, fallback → `free_text`
- **Niveaux de confiance** calibrés autour du seuil F28 (0.70) :
  - 0.50 : free_text fallback
  - 0.55–0.65 : regex/nom seul → déclenche F32 uniquement (sous 0.70, évite les faux-positifs F28)
  - 0.75 : nom-keyword + range numérique ≥90% valide → déclenche F28 range validators
- **`enrich_sync()`** : point d'entrée synchrone pour le pipeline sync (heuristique only)
- **`enrich_async()` v2** : heuristique d'abord → LLM enhance si activé → `_merge_results()`
- **`_merge_results(heuristic, llm)`** : LLM gagne seulement si confidence strictement
  supérieure ; toutes les colonnes heuristiques conservées si LLM les ignore
- **Symétrie sync/async** : `execute()` (sync) appelle maintenant `_validate_semantic_types()`
  comme `execute_async()` le faisait déjà

**Fichiers modifiés** :

- `src/agents/semantic_profiler.py` — réécriture complète v2 : `_heuristic_classify`,
  `enrich_sync`, `_merge_results`, `enrich_async` v2, `_classify_columns_llm` (renommé)
- `src/agents/quality.py` — `_validate_semantic_types()` ajouté dans `execute()` sync
- `src/agents/orchestrator.py` — `_run_semantic_enrichment_sync()` + 3 appels dans pipelines

**Tests** :

- 3 tests mis à jour : `test_returns_heuristic_types_when_llm_disabled`,
  `test_fallback_on_api_error`, `test_fallback_on_timeout`
- 18 nouveaux tests : `TestHeuristicClassifier` (11), `TestEnrichSync` (3), `TestMergeResults` (4)

**Critères de validation** :

- [x] `enrich_sync()` popule `semantic_types` même sans LLM
- [x] `enrich_async()` avec `ENABLE_LLM_CHECKS=false` → tout en méthode `"heuristic"`
- [x] `enrich_async()` avec LLM → fusion heuristique + LLM (LLM gagne si confidence > heuristique)
- [x] Timeout ou erreur API → heuristique conservé, pas de crash
- [x] "stage" ne classifié pas comme "age" (word-boundary)
- [x] `sample_dirty_df.age` (8/10=80% en range) → confidence < 0.70 → 0 faux positif F28
- [x] 0 régression sur les ~288 tests existants
- [x] +18 nouveaux tests

---

## Processus de développement

### Définition of Done (DoD)

Une feature est **terminée** si :

1. Le code est écrit et ne régresse pas les tests existants
2. Des tests couvrent la nouvelle feature (unitaires + intégration)
3. `ARCHITECTURE.md` est mis à jour si le design change
4. `DEVLOG.md` documente les décisions non triviales
5. `ROADMAP.md` coche les critères de validation

---

## Changelog

| Version | Date | Changements |
|---------|------|-------------|
| v0.1.0 | — | Commit initial : pipeline complet, API, ChromaDB |
| v0.1.1 | 2026-03 | 4 bugs corrigés, 66 tests ajoutés, DEVLOG créé |
| v0.2.0 | 2026-03 | Upload CSV/Parquet, Prometheus, JWT, Redis sessions — 97/97 tests |
| v0.3.0 | 2026-03 | Parallélisation Quality (asyncio), rate limiting, webhooks, PDF, Streamlit — 120/120 tests |
| v0.4.0 | 2026-03 | Q1 doublons, Q2 pseudo-nulls, Q3 formats, Q4 score/colonne, corrections JSON, Excel, persistance webhooks, dette technique — 166/166 tests |
| v0.5.0 | 2026-03 | Apply-corrections (CSV propre), persistance DataFrame (parquet/base64), batch API (10 fichiers en parallèle), Streamlit v0.5 — 188/188 tests |
| v0.6.0 | 2026-03 | Comparison before/after (F19), CRUD /rules (F20), async jobs (F21), dashboard stats (F22), Streamlit v0.6 — 210+ tests |
| v0.7.0 | 2026-03 | LLM Quality Check opt-in (F23), orchestrateur adaptatif ReAct (F24), RAG actif seuils dynamiques (F25), feedback qui apprend (F26), Streamlit v0.7 — 275/275 tests |
| v0.8.0 | 2026-03 | SemanticProfilerAgent batch LLM (F27v1), validation sémantique QualityAgent (F28), export schéma GET /schema (F29), onglet Schéma Streamlit — 300+ tests |
| v0.9.0 | 2026-03 | Logs console colorés (ColoredFormatter + structlog opt.), setup_logging() dans lifespan — 288/288 tests |
| v1.0.0 | 2026-03 | Custom Domain Agent Builder (F32) : DomainManager, CRUD /domain-agents, _validate_domain_rules, onglet "Agents Métier" Streamlit — 288/288 tests |
| v1.1.0 | 2026-03 | SemanticProfiler v2 (F27v2) : classifieur heuristique, enrich_sync, fusion heuristic+LLM, symétrie sync/async Quality — ~306 tests |
