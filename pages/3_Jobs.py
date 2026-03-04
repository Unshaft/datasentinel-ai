"""
DataSentinel AI — Page Jobs async

Soumission et suivi des analyses asynchrones.
"""

import requests
import streamlit as st

from pages._helpers import api_get, setup_sidebar

st.set_page_config(
    page_title="Jobs — DataSentinel AI",
    page_icon="⏳",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("⏳ Jobs asynchrones")
st.caption(
    "Pour les gros fichiers, soumettez un job et revenez vérifier son statut "
    "sans attendre la fin de l'analyse. TTL : 2h."
)

col_submit, col_check = st.columns(2)

# ── Soumettre un job ─────────────────────────────────────────────────────────
with col_submit:
    st.subheader("Soumettre un job")
    job_file = st.file_uploader(
        "Fichier à analyser en tâche de fond",
        type=["csv", "parquet"],
        key="job_uploader",
    )
    if job_file and st.button("🚀 Soumettre", key="btn_submit_job", type="primary"):
        try:
            job_resp = requests.post(
                f"{api_url}/jobs/analyze",
                files={"file": (job_file.name, job_file.getvalue(), job_file.type)},
                headers=headers,
                timeout=15,
            )
            if job_resp.status_code == 202:
                jdata = job_resp.json()
                st.session_state["current_job_id"] = jdata["job_id"]
                st.success(f"✅ Job soumis ! **ID : `{jdata['job_id']}`**")
                st.caption(f"Créé à : {jdata.get('created_at', '—')}")
            else:
                try:
                    detail = job_resp.json().get("detail", job_resp.text)
                except Exception:
                    detail = job_resp.text
                st.error(f"Erreur (HTTP {job_resp.status_code}) : {detail}")
        except Exception as e:
            st.error(f"Impossible de soumettre le job : {e}")


# ── Vérifier un job ──────────────────────────────────────────────────────────
with col_check:
    st.subheader("Vérifier le statut")
    job_id_input = st.text_input(
        "ID du job",
        value=st.session_state.get("current_job_id", ""),
        placeholder="job_xxxxxxxxxxxxxxxx",
        key="job_id_input",
    )

    col_check_btn, col_auto = st.columns(2)
    check = col_check_btn.button("🔍 Vérifier", key="btn_check_job")

    if job_id_input and check:
        status_resp = api_get(f"{api_url}/jobs/{job_id_input}", headers)
        if status_resp and status_resp.status_code == 200:
            jstatus = status_resp.json()
            status_val = jstatus["status"]
            progress = jstatus.get("progress", 0)

            _badge = {
                "pending": "⏳ En attente",
                "running": "🔄 En cours",
                "completed": "✅ Terminé",
                "failed": "❌ Échoué",
            }.get(status_val, status_val)

            st.metric("Statut", _badge)
            st.progress(min(int(progress), 100))

            if status_val == "completed" and jstatus.get("result"):
                res = jstatus["result"]
                sq = res.get("quality_score", 0)
                n_issues = len(res.get("issues", []))
                st.success(f"Score : **{sq:.1f}** — {n_issues} issue(s)")
                if st.button("📋 Voir dans Analyse", key="btn_load_job_result"):
                    st.session_state["analysis_result"] = res
                    st.session_state["analysis_session_id"] = res.get("session_id", "")
                    st.info("Résultats chargés — naviguez vers la page **Analyse**.")
                with st.expander("Résultats complets (JSON)"):
                    st.json(res)
            elif status_val == "failed" and jstatus.get("error"):
                st.error(f"Erreur : {jstatus['error']}")

        elif status_resp and status_resp.status_code == 404:
            st.warning(f"Job `{job_id_input}` introuvable ou expiré (TTL 2h).")
        elif status_resp:
            st.error(f"Erreur : HTTP {status_resp.status_code}")

# ── Guide ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ Comment ça marche ?"):
    st.markdown("""
1. **Soumettez** un fichier CSV/Parquet — l'API répond immédiatement avec un `job_id` (HTTP 202).
2. L'analyse tourne **en arrière-plan** (dans un thread séparé).
3. **Vérifiez** périodiquement le statut avec le `job_id`.
4. Quand `status = completed`, récupérez les résultats — identiques à une analyse `/upload` classique.

**TTL** : les jobs expirent après **2 heures**. Après expiration, le `job_id` retourne HTTP 404.
""")
