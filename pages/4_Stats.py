"""
DataSentinel AI — Page Stats

Tableau de bord analytique des analyses effectuées.
"""

import requests
import streamlit as st

from pages._helpers import api_get, setup_sidebar

st.set_page_config(
    page_title="Stats — DataSentinel AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("📊 Tableau de bord analytique")
st.caption("Agrégats de toutes les analyses effectuées depuis le démarrage du serveur.")

col_refresh, col_reset = st.columns([1, 1])

if col_refresh.button("🔄 Rafraîchir", key="btn_refresh_stats"):
    st.session_state.pop("stats_data", None)

if "stats_data" not in st.session_state:
    resp = api_get(f"{api_url}/stats", headers)
    if resp and resp.status_code == 200:
        st.session_state["stats_data"] = resp.json()
    elif resp:
        st.error(f"Erreur stats : HTTP {resp.status_code}")

stats_data = st.session_state.get("stats_data")

if stats_data:
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sessions totales", stats_data.get("total_sessions", 0))
    k2.metric("Score moyen", f"{stats_data.get('avg_quality_score', 0):.1f} / 100")

    dist = stats_data.get("score_distribution", {})
    poor = dist.get("0-20", 0) + dist.get("20-40", 0) + dist.get("40-60", 0)
    k3.metric("Sessions score < 60", poor)

    top = stats_data.get("top_issue_types", {})
    top_issue = max(top, key=top.get) if top else "—"
    k4.metric("Issue #1", top_issue.replace("_", " ").title() if top_issue != "—" else "—")

    st.divider()

    row1_left, row1_right = st.columns(2)

    with row1_left:
        if top:
            st.subheader("Top types d'issues")
            st.bar_chart(top)

    with row1_right:
        sessions_by_day = stats_data.get("sessions_by_day", {})
        if sessions_by_day:
            st.subheader("Sessions par jour (7 derniers jours)")
            st.bar_chart(sessions_by_day)

    if dist:
        st.subheader("Distribution des scores (buckets)")
        st.bar_chart(dist)

    st.caption(f"Mis à jour : {stats_data.get('updated_at', '—')}")
    st.divider()

    if col_reset.button("🗑️ Réinitialiser les stats", key="btn_reset_stats"):
        try:
            r = requests.delete(f"{api_url}/stats", headers=headers, timeout=10)
            if r.status_code == 200:
                st.session_state.pop("stats_data", None)
                st.success("Stats remises à zéro.")
                st.rerun()
            else:
                st.error(f"Erreur reset : HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Erreur : {e}")
else:
    st.info("Cliquez sur **Rafraîchir** ou lancez quelques analyses pour alimenter le dashboard.")
