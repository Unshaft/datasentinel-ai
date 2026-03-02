# DataSentinel AI — Roadmap et suivi de développement

Ce fichier pilote la progression feature par feature.
Chaque section documente le **pourquoi**, le **comment** et les **critères de validation**.

---

## État actuel — v0.5.0

| Composant | État | Notes |
|-----------|------|-------|
| Pipeline Profiler → Quality → Corrector → Validator | ✅ Opérationnel | 155+ tests passent |
| Upload CSV / Parquet (`POST /upload`) | ✅ v0.2 | `pyarrow`, validation extension + taille |
| Métriques Prometheus (`GET /metrics`) | ✅ v0.2 | Auto-instrumenté via `prometheus-fastapi-instrumentator` |
| Authentification JWT (`POST /auth/token`) | ✅ v0.2 | Opt-in (`AUTH_ENABLED=true`), fallback anonymous en dev |
| Persistance sessions Redis (`GET /analyze/{id}`) | ✅ v0.2 | Fallback in-memory si Redis indisponible |
| Quality checks en parallèle (`asyncio.gather`) | ✅ v0.3 | ~40% de latence en moins sur le pipeline Quality |
| Rate limiting (`slowapi`) | ✅ v0.3 | 30/min sur `/analyze`, 10/min sur `/upload` |
| Webhooks (`POST /webhooks`) | ✅ v0.3 | Notifications async POST JSON après analyse |
| Rapport PDF (`GET /analyze/{id}/report.pdf`) | ✅ v0.3 | `reportlab`, export professionnel |
| Interface Streamlit (`streamlit_app.py`) | ✅ v0.5 | Dashboard + score/colonne + corrections + Excel + batch + apply |
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

### Dette technique résolue en v0.4

| Problème | Résolution |
| -------- | ---------- |
| `datetime.utcnow()` déprécié (Python 3.12+) | Remplacé par `datetime.now(timezone.utc)` dans tous les modèles |
| `Class Config` pydantic déprécié | Remplacé par `model_config = ConfigDict(...)` dans `responses.py` |
| Webhooks in-memory (perdus au redémarrage) | Persistance JSON dans `./data/webhooks.json` |

### Dette technique restante

| Problème | Impact | Priorité |
| -------- | ------ | -------- |
| Singleton `ChromaStore` problématique en tests parallèles | Flakiness potentielle | Moyenne |

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
