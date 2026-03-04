"""
SemanticProfilerAgent — Classification sémantique des colonnes (F27 — v2.0).

v2 apporte trois améliorations majeures par rapport à v1 :

1. Classificateur heuristique (toujours actif, sans LLM)
   - Regex sur les valeurs (email, phone, url, postal, ip)
   - Mots-clés dans le nom de colonne (salary → monetary_amount, age → age, …)
   - Validation de plage numérique (age → [0,150], percentage → [0,100])
   - Heuristiques statistiques (cardinalité basse → category, tout unique → identifier)

2. enrich_sync() — synchrone, heuristique seul
   Utilisé par le pipeline synchrone (run_pipeline) pour activer F32 et F28
   sans aucun appel API.

3. enrich_async() — heuristique + LLM (si activé)
   Lance l'heuristique en premier, puis enrichit/corrige avec le LLM via
   _merge_results() : le LLM gagne uniquement si sa confidence est plus haute.

Niveaux de confidence (par rapport au seuil F28 de 0.70) :
  0.50       — free_text / fallback
  0.55–0.65  — heuristiques statistiques / regex / mots-clés simples (domain F32 only)
  0.75       — nom de colonne + validation de plage ≥90 % (déclenche les validators F28)

Résultat stocké dans context.metadata["semantic_types"] :
    {
        "col_name": {
            "semantic_type": "email",
            "confidence": 0.65,
            "method": "heuristic" | "llm",
            "language": None,
            "pattern": None,
        },
        …
    }
"""

import asyncio
import logging
import re
import time
from typing import Any

import pandas as pd

from src.core.config import settings
from src.core.models import AgentContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes — types sémantiques reconnus
# ---------------------------------------------------------------------------

_SEMANTIC_TYPES = [
    "email", "phone", "first_name", "last_name", "full_name",
    "postal_code", "address", "city", "country",
    "identifier", "monetary_amount", "percentage", "age",
    "date_string", "url", "ip_address",
    "boolean_text", "category", "product_code",
    "employee_id", "description", "free_text",
    "quantity", "rating",
]

# ---------------------------------------------------------------------------
# Constantes — classificateur heuristique
# ---------------------------------------------------------------------------

_BOOL_VALUES: frozenset = frozenset({
    "yes", "no", "oui", "non", "true", "false",
    "vrai", "faux", "y", "n", "1", "0",
    "on", "off", "actif", "inactif",
})

# Regex appliqués sur les valeurs (≥70 % de match → type détecté, confidence 0.65)
_HEURISTIC_REGEXES: dict[str, re.Pattern] = {
    "email":       re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"),
    "phone":       re.compile(r"^(\+?33|0033|0)[1-9](\s?\d{2}){4}$"),
    "url":         re.compile(r"^https?://[^\s]{3,}$"),
    "postal_code": re.compile(r"^\d{5}$"),
    "ip_address":  re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"),
}

# Mots-clés dans le nom de colonne → semantic_type (confidence 0.62)
# Clés d'abord les plus spécifiques pour éviter les faux positifs.
# Les keywords multi-mots (avec _) utilisent la correspondance par sous-chaîne ;
# les keywords simples utilisent la délimitation par mot (split sur _/-/espace).
_HEURISTIC_NAME_KEYWORDS: dict[str, list[str]] = {
    "employee_id":     ["employee_id", "emp_id", "employe_id", "matricule", "staff_id"],
    "product_code":    ["sku", "product_code", "code_produit", "ref_produit", "article_code"],
    "ip_address":      ["ip_address", "ip_addr", "adresse_ip"],
    "email":           ["email", "mail", "courriel", "e_mail"],
    "phone":           ["phone", "tel", "telephone", "mobile", "portable", "cellulaire"],
    "postal_code":     ["postal_code", "zip_code", "code_postal", "zipcode"],
    "url":             ["url", "website", "site_web", "href"],
    "monetary_amount": [
        "salary", "salaire", "montant", "amount", "price", "prix",
        "revenue", "revenu", "cost", "cout", "wage", "budget",
        "turnover", "invoice", "facture", "chiffre_affaire",
    ],
    "percentage": ["percentage", "pct", "percent", "taux", "ratio"],
    "age":        ["age", "annee_naissance", "birth_year"],
    "date_string": [
        "date", "time", "timestamp", "created_at", "updated_at",
        "created_on", "updated_on", "birth_date", "date_naissance",
    ],
    "boolean_text": ["is_active", "is_enabled", "is_deleted", "has_", "flag_", "actif", "active", "enabled"],
    "quantity":     ["qty", "quantity", "quantite", "stock", "nb_", "qte", "count"],
    "rating":       ["rating", "stars", "grade", "notation", "evaluation"],
    "first_name":   ["prenom", "first_name", "firstname", "given_name"],
    "last_name":    ["nom_famille", "last_name", "lastname", "surname", "family_name"],
    "full_name":    ["full_name", "fullname", "nom_complet", "nom_prenom"],
    "address":      ["address", "adresse", "street", "addr"],
    "city":         ["city", "ville", "localite", "commune"],
    "country":      ["country", "pays", "nation", "country_code"],
    "category":     ["category", "categorie", "status", "statut", "class", "genre", "kind"],
    "description":  ["description", "desc", "comment", "remarks", "observations"],
}

# Plages valides pour la validation numérique des types bornés
_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age":        (0.0, 150.0),
    "percentage": (0.0, 100.0),
    "rating":     (0.0, 10.0),
}


def _is_keyword_match(col_lower: str, keyword: str) -> bool:
    """
    Vérifie la correspondance d'un keyword dans un nom de colonne.

    - Keywords multi-mots (avec _ ou -) : correspondance par sous-chaîne.
    - Keywords simples : délimitation par mots (split sur _/-/espace)
      pour éviter les faux positifs (ex. "age" dans "stage").
    """
    if "_" in keyword or "-" in keyword:
        return keyword in col_lower
    col_parts = set(re.split(r"[_\-\s]+", col_lower))
    return keyword in col_parts


# ---------------------------------------------------------------------------
# LLM tool definition (inchangé par rapport à v1)
# ---------------------------------------------------------------------------

_CLASSIFY_TOOL: dict[str, Any] = {
    "name": "classify_column",
    "description": "Classify the semantic type of a data column.",
    "input_schema": {
        "type": "object",
        "properties": {
            "column_name": {"type": "string"},
            "semantic_type": {"type": "string", "enum": _SEMANTIC_TYPES},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "language": {"type": "string"},
            "pattern": {"type": "string"},
        },
        "required": ["column_name", "semantic_type", "confidence"],
    },
}

_SYSTEM_PROMPT = (
    "You are a data engineer. Classify each column's semantic type using classify_column. "
    "Call it once per column. Use column name AND sample values. "
    "Be concise — omit language/pattern unless clearly applicable."
)


# ---------------------------------------------------------------------------
# SemanticProfilerAgent
# ---------------------------------------------------------------------------


class SemanticProfilerAgent:
    """
    Agent de classification sémantique des colonnes (F27 — v2.0).

    Deux couches :
    1. Heuristique (toujours actif) — regex + mots-clés + statistiques pandas.
    2. LLM (opt-in, ENABLE_LLM_CHECKS=True) — enrichit / corrige les résultats
       heuristiques via un seul appel batch à Claude.

    La couche heuristique permet à F32 (détection de domaine) et F28
    (validateurs sémantiques) de fonctionner sans API key.
    """

    # ------------------------------------------------------------------
    # Heuristic classifier
    # ------------------------------------------------------------------

    def _classify_one_heuristic(
        self,
        df: pd.DataFrame,
        col: str,
    ) -> tuple[str, float]:
        """
        Classifie une colonne par heuristiques.

        Confidence levels (par rapport au seuil F28 de 0.70) :
        - 0.50       : free_text / fallback
        - 0.55–0.65  : statistiques / regex / mots-clés → domain detection (F32) only
        - 0.75       : nom + plage numérique ≥90 % → déclenche validateurs F28

        Returns:
            (semantic_type, confidence)
        """
        col_lower = col.lower()
        series = df[col].dropna()
        total_rows = len(df)
        n_non_null = len(series)

        if n_non_null == 0:
            return "free_text", 0.50

        # ── 1. Regex sur les valeurs (colonnes object) ─────────────────────────
        if series.dtype == object:
            str_samples = series.astype(str).str.strip().head(50)

            for sem_type, pattern in _HEURISTIC_REGEXES.items():
                match_rate = str_samples.str.match(pattern, na=False).mean()
                if match_rate >= 0.70:
                    # Confidence capped à 0.65 : évite les doublons avec
                    # _detect_format_issues() qui couvre déjà email/phone/url/postal
                    return sem_type, 0.65

            # Détection date
            try:
                parsed = pd.to_datetime(
                    series.head(20), errors="coerce", infer_datetime_format=True
                )
                if parsed.notna().mean() >= 0.70:
                    return "date_string", 0.65
            except Exception:  # noqa: BLE001
                pass

            # Texte booléen
            bool_rate = series.astype(str).str.strip().str.lower().isin(_BOOL_VALUES).mean()
            if bool_rate >= 0.80:
                return "boolean_text", 0.65

        # ── 2. Mots-clés dans le nom de colonne ────────────────────────────────
        for sem_type, keywords in _HEURISTIC_NAME_KEYWORDS.items():
            if any(_is_keyword_match(col_lower, kw) for kw in keywords):
                # Validation numérique pour les types bornés → confidence 0.75
                if sem_type in _NUMERIC_RANGES and pd.api.types.is_numeric_dtype(series):
                    try:
                        numeric = pd.to_numeric(series, errors="coerce").dropna()
                        if len(numeric) > 0:
                            lo, hi = _NUMERIC_RANGES[sem_type]
                            if numeric.between(lo, hi).mean() >= 0.90:
                                # ≥90 % des valeurs dans la plage → confidence élevée
                                # → déclenche les validateurs F28 (range check)
                                return sem_type, 0.75
                    except Exception:  # noqa: BLE001
                        pass
                # Mot-clé seul → confidence en dessous du seuil F28
                return sem_type, 0.62

        # ── 3. Heuristiques structurelles ─────────────────────────────────────

        # Suffixe _id → identifier
        if col_lower.endswith("_id") or col_lower == "id":
            conf = 0.70 if (total_rows >= 5 and series.nunique() == total_rows) else 0.63
            return "identifier", conf

        # Toutes les valeurs uniques → identifier
        if total_rows >= 5 and series.nunique() == total_rows:
            return "identifier", 0.63

        # Colonnes object
        if series.dtype == object:
            n_unique = series.nunique()
            # Faible cardinalité → category (≤10 valeurs distinctes et ratio ≤40%)
            if n_non_null >= 5 and 2 <= n_unique <= 10 and n_unique / n_non_null <= 0.40:
                return "category", 0.60
            # Chaînes longues → description
            if series.astype(str).str.len().mean() > 50:
                return "description", 0.55

        return "free_text", 0.50

    def _heuristic_classify(
        self,
        df: pd.DataFrame,
        max_columns: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Classifie toutes les colonnes par heuristiques (sans LLM).

        Returns:
            dict {col_name: {semantic_type, confidence, method, language, pattern}}
        """
        result: dict[str, dict[str, Any]] = {}
        cols = list(df.columns[:max_columns])

        for col in cols:
            sem_type, confidence = self._classify_one_heuristic(df, col)
            result[col] = {
                "semantic_type": sem_type,
                "confidence": confidence,
                "method": "heuristic",
                "language": None,
                "pattern": None,
            }

        return result

    @staticmethod
    def _merge_results(
        heuristic: dict[str, dict],
        llm: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Fusionne résultats heuristiques et LLM.

        Règle : le LLM remplace l'heuristique uniquement si sa confidence est
        strictement supérieure. Les colonnes absentes du LLM conservent leur
        classification heuristique.
        """
        merged = dict(heuristic)
        for col, llm_info in llm.items():
            if col not in merged:
                merged[col] = {**llm_info, "method": "llm"}
            elif llm_info.get("confidence", 0.0) > merged[col].get("confidence", 0.0):
                merged[col] = {**llm_info, "method": "llm"}
            # else: heuristique conservé
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_sync(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        max_columns: int = 20,
    ) -> AgentContext:
        """
        Enrichissement sémantique synchrone — heuristiques uniquement.

        Toujours disponible sans appel API. Utilisé par le pipeline synchrone
        pour activer la détection de domaine (F32) et les validateurs
        sémantiques (F28) sans dépendance LLM.

        Pour l'enrichissement LLM, utiliser enrich_async() dans un contexte async.
        """
        heuristic = self._heuristic_classify(df, max_columns)
        context.metadata["semantic_types"] = heuristic
        logger.debug(
            "[F27v2] Heuristic classified %d/%d columns",
            len(heuristic),
            min(len(df.columns), max_columns),
        )
        return context

    async def enrich_async(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        max_columns: int = 20,
    ) -> AgentContext:
        """
        Enrichissement sémantique asynchrone.

        Étape 1 (toujours) : heuristique — popule semantic_types.
        Étape 2 (opt-in)   : LLM améliore/corrige si ENABLE_LLM_CHECKS=True.
                             Le LLM ne gagne que si sa confidence > heuristique.

        En cas de timeout ou d'erreur LLM, les résultats heuristiques sont conservés.
        """
        # Étape 1 : heuristique (toujours, sans API)
        heuristic = self._heuristic_classify(df, max_columns)
        context.metadata["semantic_types"] = heuristic
        logger.debug("[F27v2] Heuristic pre-classified %d columns", len(heuristic))

        # Étape 2 : LLM (opt-in)
        if not settings.enable_llm_checks:
            return context

        try:
            import anthropic  # noqa: F401
        except ImportError:
            logger.warning("SemanticProfilerAgent: anthropic non installé, LLM skipped.")
            return context

        try:
            return await asyncio.wait_for(
                self._classify_columns_llm(context, df, max_columns, heuristic),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "SemanticProfilerAgent: timeout LLM (30s) — heuristique conservé."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticProfilerAgent: LLM enhancement failed — %s (heuristique conservé)",
                exc,
            )

        return context

    # ------------------------------------------------------------------
    # Internal — LLM batch call
    # ------------------------------------------------------------------

    async def _classify_columns_llm(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        max_columns: int,
        heuristic: dict[str, dict],
    ) -> AgentContext:
        """
        Appel LLM batch + fusion avec l'heuristique.

        Le LLM classifie toutes les colonnes en un seul appel ;
        _merge_results() garde le résultat avec la plus haute confidence.
        """
        import anthropic

        cols = list(df.columns[:max_columns])
        if not cols:
            return context

        # Format compact : col [dtype]: v1, v2, v3, v4, v5
        lines: list[str] = []
        for col in cols:
            non_null = df[col].dropna()
            samples = (
                non_null.sample(min(5, len(non_null)), random_state=42)
                .astype(str)
                .tolist()
                if len(non_null) > 0
                else []
            )
            lines.append(f"{col} [{df[col].dtype}]: {', '.join(samples)}")

        user_message = (
            "Classify each column using classify_column (once per column):\n"
            + "\n".join(lines)
        )

        logger.info("[LLM F27v2] Enhancing %d columns...", len(cols))
        _t0 = time.time()

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.llm_check_model,
            max_tokens=max(1024, len(cols) * 100),
            system=_SYSTEM_PROMPT,
            tools=[_CLASSIFY_TOOL],  # type: ignore[list-item]
            messages=[{"role": "user", "content": user_message}],
        )

        # Parse les résultats LLM
        col_set = set(cols)
        llm_results: dict[str, dict[str, Any]] = {}

        for block in response.content:
            if block.type != "tool_use" or block.name != "classify_column":
                continue
            inp = block.input
            col_name = inp.get("column_name", "")
            if col_name not in col_set:
                continue
            llm_results[col_name] = {
                "semantic_type": inp.get("semantic_type", "free_text"),
                "confidence": float(inp.get("confidence", 0.8)),
                "method": "llm",
                "language": inp.get("language"),
                "pattern": inp.get("pattern"),
            }

        # Fusion : LLM gagne si confidence plus haute
        context.metadata["semantic_types"] = self._merge_results(heuristic, llm_results)

        logger.info(
            "LLM enhanced %d/%d cols → %d total (%dms)",
            len(llm_results),
            len(cols),
            len(context.metadata["semantic_types"]),
            int((time.time() - _t0) * 1000),
        )
        return context


def get_semantic_profiler() -> SemanticProfilerAgent:
    """Retourne une instance de SemanticProfilerAgent (stateless, pas de singleton)."""
    return SemanticProfilerAgent()
