"""
SemanticProfilerAgent — Classification sémantique des colonnes via LLM (F27 — v0.8).

Classifie la nature métier de chaque colonne à partir de son nom et de ses valeurs.
1 seul appel LLM batch pour toutes les colonnes (vs 1 appel/colonne en F23).
Opt-in via ENABLE_LLM_CHECKS=true.

Résultat stocké dans context.metadata["semantic_types"] :
    {
        "nom_colonne": {
            "semantic_type": "email",
            "confidence": 0.97,
            "language": "fr",
            "pattern": None,
            "notes": None,
        },
        ...
    }
"""

import asyncio
import logging
import time
from typing import Any

import pandas as pd

from src.core.config import settings
from src.core.models import AgentContext

logger = logging.getLogger(__name__)

# Types sémantiques reconnus — même liste que le plan F27
_SEMANTIC_TYPES = [
    "email", "phone", "first_name", "last_name", "full_name",
    "postal_code", "address", "city", "country",
    "identifier", "monetary_amount", "percentage", "age",
    "date_string", "url", "ip_address",
    "boolean_text", "category", "product_code",
    "employee_id", "description", "free_text",
    "quantity", "rating",
]

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


class SemanticProfilerAgent:
    """
    Agent de classification sémantique des colonnes via LLM (F27 — v0.8).

    Design :
    - Pas de BaseAgent (pas de LangChain/ChatAnthropic) — utilise anthropic.AsyncAnthropic
      directement comme _detect_semantic_anomalies_llm (F23) pour cohérence.
    - 1 seul appel batch → Claude appelle classify_column une fois par colonne.
    - Timeout global 30s sur tout le batch.
    - Fallback silencieux : toute exception laisse context intact.
    """

    async def enrich_async(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        max_columns: int = 20,
    ) -> AgentContext:
        """
        Enrichit context.metadata["semantic_types"] avec les types sémantiques LLM.

        Args:
            context: Contexte agent courant.
            df: DataFrame à analyser.
            max_columns: Limite de colonnes classifiées (défaut 20).

        Returns:
            context enrichi (inchangé si LLM désactivé ou erreur).
        """
        if not settings.enable_llm_checks:
            return context

        try:
            import anthropic
        except ImportError:
            logger.warning("SemanticProfilerAgent: anthropic non installé, skip.")
            return context

        try:
            return await asyncio.wait_for(
                self._classify_columns(context, df, max_columns),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "SemanticProfilerAgent: timeout (30s) — classification abandonnée."
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SemanticProfilerAgent: erreur inattendue — %s", exc)

        return context

    async def _classify_columns(
        self,
        context: AgentContext,
        df: pd.DataFrame,
        max_columns: int,
    ) -> AgentContext:
        """Appel LLM batch réel."""
        import anthropic

        cols = list(df.columns[:max_columns])
        if not cols:
            return context

        # Format compact : col [dtype]: v1, v2, v3, v4, v5  (5 samples max)
        lines: list[str] = []
        for col in cols:
            non_null = df[col].dropna()
            samples = (
                non_null.sample(min(5, len(non_null)), random_state=42)
                .astype(str)
                .tolist()
                if len(non_null) > 0 else []
            )
            lines.append(f"{col} [{df[col].dtype}]: {', '.join(samples)}")

        user_message = (
            "Classify each column using classify_column (once per column):\n"
            + "\n".join(lines)
        )

        logger.info("[LLM F27] Classifying %d columns...", len(cols))
        _t0 = time.time()
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.llm_check_model,
            max_tokens=max(1024, len(cols) * 100),
            system=_SYSTEM_PROMPT,
            tools=[_CLASSIFY_TOOL],  # type: ignore[list-item]
            messages=[{"role": "user", "content": user_message}],
        )

        # Parse les tool_use blocks
        col_set = set(cols)
        semantic_types: dict[str, dict[str, Any]] = {}

        for block in response.content:
            if block.type != "tool_use" or block.name != "classify_column":
                continue
            inp = block.input
            col_name = inp.get("column_name", "")
            if col_name not in col_set:
                # Colonne inconnue retournée par le LLM → ignorer
                continue
            semantic_types[col_name] = {
                "semantic_type": inp.get("semantic_type", "free_text"),
                "confidence": float(inp.get("confidence", 0.8)),
                "language": inp.get("language"),
                "pattern": inp.get("pattern"),
            }

        context.metadata["semantic_types"] = semantic_types
        logger.info(
            "%d/%d cols classifiées (%dms)",
            len(semantic_types),
            len(cols),
            int((time.time() - _t0) * 1000),
        )
        return context


def get_semantic_profiler() -> SemanticProfilerAgent:
    """Retourne une instance de SemanticProfilerAgent (stateless, pas de singleton)."""
    return SemanticProfilerAgent()
