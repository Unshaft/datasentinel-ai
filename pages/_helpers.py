"""
Helpers partagés pour toutes les pages Streamlit.

Ce module fournit la sidebar commune, les appels API et les utilitaires UI.
Il commence par _ pour être invisible dans la navigation Streamlit.
"""

import requests
import streamlit as st

API_DEFAULT = "http://localhost:8000"

_SEVERITY_BADGE = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# =============================================================================
# SIDEBAR PARTAGÉE
# =============================================================================


def setup_sidebar() -> tuple[str, dict]:
    """Configure la sidebar et retourne (api_url, headers)."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        api_url = st.text_input(
            "URL de l'API",
            value=st.session_state.get("api_url", API_DEFAULT),
            key="sidebar_api_url",
        )
        st.session_state["api_url"] = api_url

        auth_token = st.text_input(
            "Token Bearer (optionnel)",
            type="password",
            value=st.session_state.get("auth_token", ""),
            key="sidebar_auth_token",
        )
        st.session_state["auth_token"] = auth_token

        st.divider()

        col1, col2 = st.columns(2)
        if col1.button("🩺 API", key="sidebar_health", use_container_width=True):
            try:
                r = requests.get(f"{api_url}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ {data.get('status', 'ok').upper()}")
                else:
                    st.error(f"KO — HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Connexion échouée : {e}")

        if col2.button("🗑️ Reset", key="sidebar_clear", use_container_width=True,
                       help="Efface les résultats d'analyse en mémoire"):
            for key in ("analysis_result", "analysis_session_id"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.caption("**DataSentinel AI v1.3**")
        st.caption("ReAct · Sémantique · RAG · Métier")

    headers: dict = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return api_url, headers


# =============================================================================
# UTILITAIRES UI
# =============================================================================


def score_banner(score: float, summary: str = "") -> None:
    """Affiche le bandeau de score coloré."""
    color = "#27ae60" if score >= 80 else "#e67e22" if score >= 60 else "#e74c3c"
    emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
    st.markdown(
        f"""
<div style="
    background: linear-gradient(135deg, #1a252f, #2980b9);
    border-left: 6px solid {color};
    border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; color: white;
">
    <h2 style="margin:0; color:white;">
        {emoji} Score qualité :
        <span style="color:{color}; font-size:1.8em; font-weight:bold;">{score:.1f}</span>
        <span style="opacity:.7;">/100</span>
    </h2>
    {"<p style='margin:6px 0 0; opacity:.8;'>" + summary + "</p>" if summary else ""}
</div>
""",
        unsafe_allow_html=True,
    )


def score_badge(score: float) -> str:
    """Retourne un badge coloré pour un score."""
    if score >= 90:
        return f"🟢 {score:.0f}"
    if score >= 70:
        return f"🟡 {score:.0f}"
    if score >= 50:
        return f"🟠 {score:.0f}"
    return f"🔴 {score:.0f}"


def severity_badge(sev: str) -> str:
    return f"{_SEVERITY_BADGE.get(sev, '⚪')} {sev.upper()}"


def api_get(url: str, headers: dict, timeout: int = 15) -> requests.Response | None:
    """GET avec gestion d'erreur silencieuse. Retourne None si exception."""
    try:
        return requests.get(url, headers=headers, timeout=timeout)
    except Exception as e:
        st.error(f"Erreur réseau : {e}")
        return None


def api_post(url: str, headers: dict, timeout: int = 120, **kwargs) -> requests.Response | None:
    try:
        return requests.post(url, headers=headers, timeout=timeout, **kwargs)
    except requests.ConnectionError:
        st.error("Impossible de joindre l'API — vérifiez que le serveur est lancé.")
        return None
    except Exception as e:
        st.error(f"Erreur réseau : {e}")
        return None
