"""
DataSentinel AI — Page Historique (NEW)

Historique inter-sessions d'un dataset (F30 — DatasetMemory).
Visualise la tendance de qualité, les issues récurrentes, les colonnes problématiques.
"""

import pandas as pd
import streamlit as st

from pages._helpers import api_get, score_badge, setup_sidebar

st.set_page_config(
    page_title="Historique — DataSentinel AI",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("📂 Historique des datasets")
st.caption(
    "Suivez la tendance de qualité d'un dataset au fil du temps (F30 — DatasetMemory). "
    "Le dataset est identifié par un hash de son schéma + premier contenu."
)

# =============================================================================
# SÉLECTION DU DATASET
# =============================================================================

last_result = st.session_state.get("analysis_result")
last_dataset_id = last_result.get("dataset_id", "") if last_result else ""

dataset_id_input = st.text_input(
    "Dataset ID",
    value=last_dataset_id,
    placeholder="dataset_xxxxxxxxxxxx",
    key="hist_dataset_id",
)

if last_result and last_dataset_id:
    st.caption(f"💡 Dataset de la dernière analyse : `{last_dataset_id}`")

if not dataset_id_input.strip():
    st.info("Entrez un Dataset ID ou lancez une analyse depuis la page **Analyse**.")
    st.stop()

if st.button("🔍 Charger l'historique", key="btn_load_history", type="primary"):
    resp = api_get(f"{api_url}/datasets/{dataset_id_input.strip()}/history", headers)
    if resp and resp.status_code == 200:
        st.session_state["history_data"] = resp.json()
        st.session_state["history_dataset_id"] = dataset_id_input.strip()
    elif resp and resp.status_code == 404:
        st.warning(f"Dataset `{dataset_id_input.strip()}` introuvable (pas encore enregistré).")
        st.session_state.pop("history_data", None)
    elif resp:
        st.error(f"Erreur : HTTP {resp.status_code}")

# =============================================================================
# AFFICHAGE
# =============================================================================

history = st.session_state.get("history_data")
if not history:
    # Auto-load si dataset_id correspond à la dernière analyse
    if last_dataset_id and dataset_id_input.strip() == last_dataset_id:
        resp = api_get(f"{api_url}/datasets/{last_dataset_id}/history", headers)
        if resp and resp.status_code == 200:
            history = resp.json()
            st.session_state["history_data"] = history

if not history:
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────────────────
st.divider()

trend = history.get("trend", "new")
trend_icon = {"improving": "📈", "degrading": "📉", "stable": "➡️", "new": "🆕"}.get(trend, "➡️")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sessions enregistrées", history.get("session_count", 0))
k2.metric("Score moyen", f"{history.get('avg_quality_score', 0):.1f} / 100")
k3.metric("Tendance", f"{trend_icon} {trend.upper()}")
k4.metric("Première analyse", (history.get("first_seen", "—") or "—")[:10])

# ── GRAPHIQUE TENDANCE ────────────────────────────────────────────────────────
sessions = history.get("sessions", [])
if sessions:
    st.divider()
    st.subheader("📈 Évolution du score qualité")

    df_trend = pd.DataFrame([
        {
            "Date": s.get("timestamp", "")[:16],
            "Score": s.get("quality_score", 0),
            "Issues": sum(s.get("issue_counts", {}).values()),
        }
        for s in sessions
    ])

    if len(df_trend) > 1:
        st.line_chart(df_trend.set_index("Date")["Score"])
    else:
        st.metric("Score unique (1 session)", f"{df_trend['Score'].iloc[0]:.1f}")

    st.caption(f"{len(sessions)} session(s) — 10 dernières affichées")

    with st.expander("Tableau des sessions"):
        df_display = pd.DataFrame([
            {
                "Session": s.get("session_id", "")[:18] + "…",
                "Date": (s.get("timestamp", "—") or "—")[:16],
                "Score": score_badge(s.get("quality_score", 0)),
                "Issues": sum(s.get("issue_counts", {}).values()),
                "Top colonnes": ", ".join(s.get("top_columns", [])[:3]) or "—",
            }
            for s in sessions
        ])
        st.dataframe(df_display, use_container_width=True, hide_index=True)

# ── ISSUES RÉCURRENTES ────────────────────────────────────────────────────────
recurring = history.get("recurring_issues", {})
if recurring:
    st.divider()
    st.subheader("🔁 Issues récurrentes")
    st.caption("Nombre de sessions où chaque type d'issue a été détecté.")
    df_issues = pd.DataFrame(
        [{"Type": k.replace("_", " ").title(), "Sessions": v} for k, v in sorted(recurring.items(), key=lambda x: -x[1])]
    )
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        st.bar_chart(df_issues.set_index("Type")["Sessions"])
    with col_table:
        st.dataframe(df_issues, use_container_width=True, hide_index=True)

# ── COLONNES PROBLÉMATIQUES ───────────────────────────────────────────────────
problematic = history.get("problematic_columns", {})
if problematic:
    st.divider()
    st.subheader("🚨 Colonnes les plus problématiques")
    st.caption("Nombre de sessions avec au moins une issue sur chaque colonne.")
    df_cols = pd.DataFrame(
        [{"Colonne": k, "Sessions avec issue": v}
         for k, v in sorted(problematic.items(), key=lambda x: -x[1])]
    )
    st.dataframe(df_cols, use_container_width=True, hide_index=True)

# ── SUGGESTIONS PRO-ACTIVES ───────────────────────────────────────────────────
suggested = history.get("suggested_rules", [])
if suggested:
    st.divider()
    with st.expander("💡 Suggestions de règles pro-actives (F30)"):
        st.caption(
            "Ces règles sont suggérées automatiquement basées sur les patterns récurrents. "
            "Ajoutez-les dans la page **Règles** pour renforcer la détection."
        )
        for suggestion in suggested:
            st.markdown(f"- {suggestion}")
