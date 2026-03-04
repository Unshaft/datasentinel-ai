"""
DataSentinel AI — Page d'accueil

Lancement :
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

st.set_page_config(
    page_title="DataSentinel AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
from pages._helpers import setup_sidebar  # noqa: E402
api_url, headers = setup_sidebar()

# =============================================================================
# HERO
# =============================================================================

st.markdown(
    """
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px; padding: 40px 48px; margin-bottom: 32px; color: white;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
">
    <h1 style="margin:0; font-size:2.6em; color:white;">🔍 DataSentinel AI</h1>
    <p style="margin:12px 0 0; opacity:.85; font-size:1.1em;">
        Système multi-agents de qualité des données —
        <strong>ReAct</strong> · <strong>Sémantique</strong> ·
        <strong>RAG</strong> · <strong>Agents Métier</strong>
    </p>
    <p style="margin:6px 0 0; opacity:.6; font-size:.9em;">
        v1.3 · Pipeline : Observe → Reason → Act → Reflect
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# STATUT API
# =============================================================================

try:
    r = requests.get(f"{api_url}/health", timeout=4)
    if r.status_code == 200:
        hdata = r.json()
        api_ok = True
        api_status = hdata.get("status", "ok")
        api_version = hdata.get("version", "—")
    else:
        api_ok = False
        api_status = f"HTTP {r.status_code}"
        api_version = "—"
except Exception:
    api_ok = False
    api_status = "Inaccessible"
    api_version = "—"

status_col, vers_col = st.columns([1, 3])
with status_col:
    if api_ok:
        st.success(f"🟢 API {api_status.upper()}")
    else:
        st.error(f"🔴 API {api_status}")
        st.caption(f"Démarrez le serveur : `uvicorn src.api.main:app --reload`")

# =============================================================================
# NAVIGATION CARDS
# =============================================================================

st.markdown("### Navigation")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
<div style="background:#1e3a5f; border-radius:12px; padding:20px; text-align:center; border:1px solid #2980b9;">
    <div style="font-size:2em;">🔍</div>
    <h4 style="color:white; margin:8px 0 4px;">Analyse</h4>
    <p style="color:#aac4e0; font-size:.85em; margin:0;">Upload CSV/Parquet · Score · Issues · ReAct</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
<div style="background:#1e3a5f; border-radius:12px; padding:20px; text-align:center; border:1px solid #2980b9;">
    <div style="font-size:2em;">📦</div>
    <h4 style="color:white; margin:8px 0 4px;">Batch</h4>
    <p style="color:#aac4e0; font-size:.85em; margin:0;">Analyser jusqu'à 10 fichiers en parallèle</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
<div style="background:#1e3a5f; border-radius:12px; padding:20px; text-align:center; border:1px solid #2980b9;">
    <div style="font-size:2em;">📊</div>
    <h4 style="color:white; margin:8px 0 4px;">Stats</h4>
    <p style="color:#aac4e0; font-size:.85em; margin:0;">Tableau de bord · Tendances · Distribution</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
<div style="background:#1e3a5f; border-radius:12px; padding:20px; text-align:center; border:1px solid #2980b9;">
    <div style="font-size:2em;">💬</div>
    <h4 style="color:white; margin:8px 0 4px;">Feedback</h4>
    <p style="color:#aac4e0; font-size:.85em; margin:0;">Corriger · Apprendre · Améliorer</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("")

c5, c6, c7, c8 = st.columns(4)

with c5:
    st.markdown(
        """
<div style="background:#1a3323; border-radius:12px; padding:20px; text-align:center; border:1px solid #27ae60;">
    <div style="font-size:2em;">⏳</div>
    <h4 style="color:white; margin:8px 0 4px;">Jobs async</h4>
    <p style="color:#a3c9a8; font-size:.85em; margin:0;">Soumettre · Suivre · Résultats différés</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c6:
    st.markdown(
        """
<div style="background:#1a3323; border-radius:12px; padding:20px; text-align:center; border:1px solid #27ae60;">
    <div style="font-size:2em;">📋</div>
    <h4 style="color:white; margin:8px 0 4px;">Règles</h4>
    <p style="color:#a3c9a8; font-size:.85em; margin:0;">CRUD · Règles métier · Active RAG</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c7:
    st.markdown(
        """
<div style="background:#1a3323; border-radius:12px; padding:20px; text-align:center; border:1px solid #27ae60;">
    <div style="font-size:2em;">🏢</div>
    <h4 style="color:white; margin:8px 0 4px;">Agents métier</h4>
    <p style="color:#a3c9a8; font-size:.85em; margin:0;">Domaines · Trigger types · Overrides</p>
</div>
""",
        unsafe_allow_html=True,
    )

with c8:
    st.markdown(
        """
<div style="background:#1a3323; border-radius:12px; padding:20px; text-align:center; border:1px solid #27ae60;">
    <div style="font-size:2em;">📂</div>
    <h4 style="color:white; margin:8px 0 4px;">Historique</h4>
    <p style="color:#a3c9a8; font-size:.85em; margin:0;">Tendances · Sessions · Drift qualité</p>
</div>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# STATS RAPIDES
# =============================================================================

st.divider()
st.markdown("### Aperçu système")

try:
    stats_r = requests.get(f"{api_url}/stats", headers=headers, timeout=5)
    if stats_r.status_code == 200:
        sd = stats_r.json()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sessions totales", sd.get("total_sessions", 0))
        m2.metric("Score moyen", f"{sd.get('avg_quality_score', 0):.1f} / 100")
        top = sd.get("top_issue_types", {})
        top_issue = max(top, key=top.get) if top else "—"
        m3.metric("Issue la plus fréquente", top_issue.replace("_", " ").title() if top_issue != "—" else "—")
        m4.metric("Version API", api_version)
    else:
        st.info("Stats non disponibles — lancez quelques analyses d'abord.")
except Exception:
    st.info("Connectez l'API pour voir les statistiques système.")

# =============================================================================
# DERNIÈRE ANALYSE
# =============================================================================

last_result = st.session_state.get("analysis_result")
if last_result:
    st.divider()
    st.markdown("### Dernière analyse en mémoire")
    lr1, lr2, lr3, lr4 = st.columns(4)
    lr1.metric("Session", (st.session_state.get("analysis_session_id", "")[:16] + "…"))
    lr2.metric("Score", f"{last_result.get('quality_score', 0):.1f}")
    lr3.metric("Issues", len(last_result.get("issues", [])))
    lr4.metric("Statut", last_result.get("status", "—").upper())
    st.caption("Naviguez vers **Analyse** pour voir les détails complets.")
