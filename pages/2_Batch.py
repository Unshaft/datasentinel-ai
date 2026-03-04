"""
DataSentinel AI — Page Batch

Analyse de plusieurs fichiers en parallèle.
"""

import pandas as pd
import requests
import streamlit as st

from pages._helpers import score_badge, setup_sidebar

st.set_page_config(
    page_title="Batch — DataSentinel AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("📦 Analyse Batch")
st.caption("Analysez jusqu'à 10 fichiers CSV/Parquet simultanément en parallèle.")

batch_files = st.file_uploader(
    "Sélectionnez plusieurs fichiers",
    type=["csv", "parquet"],
    accept_multiple_files=True,
    key="batch_uploader",
    help="Max 10 fichiers, 100 MB chacun",
)

if batch_files:
    st.info(f"**{len(batch_files)}** fichier(s) sélectionné(s).")
    if len(batch_files) > 10:
        st.error("Maximum 10 fichiers par batch.")
        st.stop()

    if st.button("🚀 Lancer le batch", type="primary", key="btn_batch"):
        with st.spinner(f"Analyse de {len(batch_files)} fichier(s) en parallèle…"):
            try:
                resp = requests.post(
                    f"{api_url}/batch",
                    files=[("files", (f.name, f.getvalue(), f.type)) for f in batch_files],
                    headers=headers,
                    timeout=300,
                )
            except requests.ConnectionError:
                st.error("Impossible de joindre l'API.")
                st.stop()
            except Exception as e:
                st.error(f"Erreur réseau : {e}")
                st.stop()

        if resp.status_code == 200:
            bdata = resp.json()
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Total", bdata["total"])
            bc2.metric("✅ Réussies", bdata["succeeded"])
            bc3.metric("❌ Erreurs", bdata["failed"])

            rows = []
            for item in bdata["results"]:
                sq = item.get("quality_score")
                rows.append({
                    "Fichier": item["filename"],
                    "Statut": "✅ OK" if item["status"] == "success" else "❌ Erreur",
                    "Score": score_badge(sq) if sq is not None else "—",
                    "Issues": item["issues_count"],
                    "Session ID": item.get("session_id") or item.get("error", "—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Graphique des scores
            scores = {
                item["filename"]: item["quality_score"]
                for item in bdata["results"]
                if item.get("quality_score") is not None
            }
            if len(scores) > 1:
                st.divider()
                st.subheader("Scores comparatifs")
                st.bar_chart(pd.DataFrame({"Score": scores}))
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.error(f"Erreur batch (HTTP {resp.status_code}) : {detail}")
else:
    st.info("Sélectionnez au moins un fichier pour lancer le batch.")
