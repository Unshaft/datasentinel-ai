"""
DataSentinel AI — Interface Streamlit

Dashboard visuel pour uploader un fichier CSV/Parquet et visualiser
les résultats de l'analyse de qualité.

Lancement :
    streamlit run streamlit_app.py

L'API DataSentinel doit être lancée séparément :
    uvicorn src.api.main:app --reload
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="DataSentinel AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_DEFAULT = "http://localhost:8000"

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    api_url = st.text_input("URL de l'API", value=API_DEFAULT)
    auth_token = st.text_input("Token Bearer (optionnel)", type="password")

    st.divider()
    st.caption("DataSentinel AI v0.5")
    st.caption("Pipeline : Profiler → Quality (parallèle) → Score")

    if st.button("🩺 Vérifier l'API"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success(f"API opérationnelle — {data.get('status', 'ok')}")
            else:
                st.error(f"API KO — HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Connexion échouée : {e}")

    if st.button("🗑️ Effacer les résultats"):
        for key in ("analysis_result", "analysis_session_id"):
            st.session_state.pop(key, None)
        st.rerun()


# =============================================================================
# HEADER
# =============================================================================

st.title("🔍 DataSentinel AI")
st.markdown(
    "Système multi-agents de **qualité des données**. "
    "Uploadez un fichier CSV ou Parquet pour détecter les problèmes, "
    "obtenir un score de qualité et télécharger un rapport."
)
st.divider()


# =============================================================================
# UPLOAD
# =============================================================================

uploaded = st.file_uploader(
    "Choisissez un fichier",
    type=["csv", "parquet"],
    help="Formats acceptés : .csv, .parquet — Taille max : 100 MB",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 Analyser", type="primary", disabled=uploaded is None)

headers = {}
if auth_token:
    headers["Authorization"] = f"Bearer {auth_token}"

# Lance l'analyse et stocke le résultat en session_state
if uploaded and analyze_btn:
    with st.spinner("Analyse en cours (quality checks en parallèle)…"):
        try:
            resp = requests.post(
                f"{api_url}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                headers=headers,
                timeout=120,
            )
        except requests.ConnectionError:
            st.error(
                "Impossible de joindre l'API. "
                "Vérifiez que le serveur est lancé : `uvicorn src.api.main:app --reload`"
            )
            st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"Erreur API (HTTP {resp.status_code}) : {detail}")
        st.stop()

    # Stocker les résultats en session_state pour survivre aux reruns Streamlit
    st.session_state["analysis_result"] = resp.json()
    st.session_state["analysis_session_id"] = resp.json().get("session_id", "")


# =============================================================================
# RÉSULTATS (chargés depuis session_state → survivent aux clics d'onglets)
# =============================================================================

result = st.session_state.get("analysis_result")
session_id = st.session_state.get("analysis_session_id", "")

if result is None:
    st.info("Uploadez un fichier et cliquez sur **Analyser** pour démarrer.")
    st.stop()

# =========================================================================
# MÉTRIQUES GLOBALES
# =========================================================================

score = result.get("quality_score", 0)
issues_count = len(result.get("issues", []))
proc_ms = result.get("processing_time_ms", 0)
status_val = result.get("status", "completed")

score_color = (
    "#27ae60" if score >= 80
    else "#e67e22" if score >= 60
    else "#e74c3c"
)
score_emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #2c3e50, #3498db);
    border-radius: 12px; padding: 24px; margin-bottom: 16px; color: white;
">
    <h2 style="margin:0; color:white;">
        {score_emoji} Score de qualité :
        <span style="color:{score_color}; font-size: 2em;">{score:.1f}</span>
        <span style="font-size: 1em;">/100</span>
    </h2>
    <p style="margin:4px 0 0; opacity:0.8;">{result.get('summary', '')}</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Session", session_id[:18] + "…" if len(session_id) > 18 else session_id)
m2.metric("Problèmes détectés", issues_count)
m3.metric("Temps de traitement", f"{proc_ms} ms")
m4.metric("Statut", status_val.upper())

if result.get("needs_human_review"):
    st.warning(
        "⚠️ **Revue humaine recommandée** — "
        + ", ".join(result.get("escalation_reasons", []))
    )

# =========================================================================
# TABS
# =========================================================================

tab_issues, tab_scores, tab_corrections, tab_profile, tab_exports, tab_batch, tab_json = st.tabs(
    ["🐛 Problèmes", "📈 Score / colonne", "🔧 Corrections", "📊 Profil", "📥 Exports", "📦 Batch", "🔩 JSON brut"]
)

# ---- PROBLÈMES ----
with tab_issues:
    issues = result.get("issues", [])
    by_sev = result.get("issues_by_severity", {})

    if by_sev:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("🔴 Critical", by_sev.get("critical", 0))
        s2.metric("🟠 High", by_sev.get("high", 0))
        s3.metric("🟡 Medium", by_sev.get("medium", 0))
        s4.metric("🔵 Low", by_sev.get("low", 0))

    if not issues:
        st.success("✅ Aucun problème de qualité détecté — données propres !")
    else:
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        rows = []
        for iss in sorted(issues, key=lambda x: sev_order.get(x["severity"], 9)):
            sev = iss["severity"]
            badge = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(
                sev, "⚪"
            )
            rows.append({
                "Sévérité": f"{badge} {sev.upper()}",
                "Type": iss["issue_type"].replace("_", " ").title(),
                "Colonne": iss.get("column") or "—",
                "Description": iss["description"],
                "Affecté": f"{iss['affected_percentage']:.1f}%",
                "Confiance": f"{iss['confidence']:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---- SCORE PAR COLONNE ----
with tab_scores:
    col_scores = result.get("column_scores", {})
    if not col_scores:
        st.info("Scores par colonne non disponibles.")
    else:
        st.markdown("Score de qualité individuel par colonne (0–100).")

        def _score_level(s: float) -> str:
            if s >= 90:
                return "🟢 Excellent"
            if s >= 70:
                return "🟡 Bon"
            if s >= 50:
                return "🟠 Moyen"
            return "🔴 Mauvais"

        rows_scores = [
            {"Colonne": col, "Score": score_v, "Niveau": _score_level(score_v)}
            for col, score_v in sorted(col_scores.items(), key=lambda x: x[1])
        ]
        st.dataframe(pd.DataFrame(rows_scores), use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"Score": col_scores}).sort_values("Score"))

# ---- CORRECTIONS ----
with tab_corrections:
    corr_url = f"{api_url}/analyze/{session_id}/corrections"
    try:
        corr_resp = requests.get(corr_url, headers=headers, timeout=15)
        if corr_resp.status_code == 200:
            plan = corr_resp.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Score actuel", f"{plan['quality_score']:.1f}")
            c2.metric("Score estimé après auto", f"{plan['estimated_score_after_auto']:.1f}")
            c3.metric("Total issues", plan["total_issues"])

            auto = plan.get("auto_corrections", [])
            manual = plan.get("manual_reviews", [])

            if auto:
                st.subheader(f"⚙️ Corrections automatiques ({len(auto)})")
                st.dataframe(pd.DataFrame([{
                    "Type": e["issue_type"].replace("_", " ").title(),
                    "Sévérité": e["severity"].upper(),
                    "Colonne": e.get("column") or "—",
                    "Action recommandée": e["recommended_action"],
                    "Affecté": f"{e['affected_percentage']:.1f}%",
                } for e in auto]), use_container_width=True, hide_index=True)

            if manual:
                st.subheader(f"👁️ Revue manuelle requise ({len(manual)})")
                st.dataframe(pd.DataFrame([{
                    "Type": e["issue_type"].replace("_", " ").title(),
                    "Sévérité": e["severity"].upper(),
                    "Colonne": e.get("column") or "—",
                    "Action recommandée": e["recommended_action"],
                    "Affecté": f"{e['affected_percentage']:.1f}%",
                } for e in manual]), use_container_width=True, hide_index=True)

            if not auto and not manual:
                st.success("✅ Aucune correction nécessaire.")

            # ── Bouton apply-corrections ───────────────────────────────────────
            if auto:
                st.divider()
                st.markdown("### ⚡ Appliquer les corrections automatiques")
                st.caption(
                    f"{len(auto)} correction(s) applicable(s) automatiquement. "
                    "Le fichier corrigé sera téléchargeable en CSV."
                )
                if st.button("🔧 Appliquer et télécharger le CSV corrigé", key="btn_apply"):
                    with st.spinner("Application des corrections…"):
                        try:
                            apply_resp = requests.post(
                                f"{api_url}/analyze/{session_id}/apply-corrections",
                                headers=headers,
                                timeout=60,
                            )
                            if apply_resp.status_code == 200:
                                rows_before = apply_resp.headers.get("X-Rows-Before", "?")
                                rows_after = apply_resp.headers.get("X-Rows-After", "?")
                                corr_count = apply_resp.headers.get("X-Corrections-Count", "?")
                                st.success(
                                    f"✅ {corr_count} correction(s) appliquée(s) — "
                                    f"Lignes : {rows_before} → {rows_after}"
                                )
                                st.download_button(
                                    label="📥 Télécharger le CSV corrigé",
                                    data=apply_resp.content,
                                    file_name=f"corrected_{session_id}.csv",
                                    mime="text/csv",
                                )
                            elif apply_resp.status_code == 404:
                                st.warning("Session expirée. Relancez l'analyse.")
                            elif apply_resp.status_code == 422:
                                st.warning(apply_resp.json().get("detail", "Données non disponibles."))
                            else:
                                st.error(f"Erreur : HTTP {apply_resp.status_code}")
                        except Exception as e:
                            st.error(f"Impossible d'appliquer les corrections : {e}")

        elif corr_resp.status_code == 404:
            st.warning("Session expirée ou serveur redémarré. Relancez l'analyse.")
        else:
            st.error(f"Erreur récupération plan : HTTP {corr_resp.status_code}")
    except Exception as e:
        st.error(f"Impossible de récupérer le plan de corrections : {e}")

# ---- PROFIL ----
with tab_profile:
    profile = result.get("profile")
    if not profile:
        st.info("Profil non disponible.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Lignes", f"{profile['row_count']:,}")
        p2.metric("Colonnes", profile["column_count"])
        p3.metric("Valeurs manquantes", f"{profile['total_null_count']:,}")
        p4.metric("Mémoire", f"{profile['memory_mb']:.2f} MB")

        cols_data = []
        for col in profile.get("columns", []):
            cols_data.append({
                "Colonne": col["name"],
                "Type": col["dtype"],
                "Type inféré": col["inferred_type"],
                "Nulls": f"{col['null_count']} ({col['null_percentage']:.1f}%)",
                "Uniques": f"{col['unique_count']} ({col['unique_percentage']:.1f}%)",
                "Min": col.get("min"),
                "Max": col.get("max"),
                "Moyenne": f"{col['mean']:.2f}" if col.get("mean") is not None else "—",
            })
        st.dataframe(pd.DataFrame(cols_data), use_container_width=True, hide_index=True)

# ---- EXPORTS (boutons lazy-fetch) ----
with tab_exports:
    ex1, ex2 = st.columns(2)

    # PDF
    with ex1:
        st.markdown("### Rapport PDF")
        st.caption("Rapport complet avec issues et profil.")
        if st.button("⬇️ Générer le PDF", key="btn_pdf"):
            with st.spinner("Génération du PDF…"):
                try:
                    pdf_resp = requests.get(
                        f"{api_url}/analyze/{session_id}/report.pdf",
                        headers=headers, timeout=30,
                    )
                    if pdf_resp.status_code == 200:
                        st.download_button(
                            label="📥 Télécharger le rapport PDF",
                            data=pdf_resp.content,
                            file_name=f"rapport_{session_id}.pdf",
                            mime="application/pdf",
                        )
                        st.caption(f"Taille : {len(pdf_resp.content) / 1024:.1f} KB")
                    elif pdf_resp.status_code == 404:
                        st.warning("Session expirée. Relancez l'analyse pour regénérer le rapport.")
                    else:
                        st.error(f"Erreur PDF : HTTP {pdf_resp.status_code}")
                except Exception as e:
                    st.error(f"Impossible de générer le PDF : {e}")

    # Excel
    with ex2:
        st.markdown("### Rapport Excel")
        st.caption("4 onglets : Résumé, Issues, Profil, Score par colonne.")
        if st.button("⬇️ Générer l'Excel", key="btn_xlsx"):
            with st.spinner("Génération de l'Excel…"):
                try:
                    xlsx_resp = requests.get(
                        f"{api_url}/analyze/{session_id}/report.xlsx",
                        headers=headers, timeout=30,
                    )
                    if xlsx_resp.status_code == 200:
                        st.download_button(
                            label="📊 Télécharger le rapport Excel",
                            data=xlsx_resp.content,
                            file_name=f"rapport_{session_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                        st.caption(f"Taille : {len(xlsx_resp.content) / 1024:.1f} KB")
                    elif xlsx_resp.status_code == 404:
                        st.warning("Session expirée. Relancez l'analyse pour regénérer le rapport.")
                    else:
                        st.error(f"Erreur Excel : HTTP {xlsx_resp.status_code}")
                except Exception as e:
                    st.error(f"Impossible de générer le fichier Excel : {e}")

# ---- BATCH ----
with tab_batch:
    st.markdown("### Analyse en lot (Batch)")
    st.caption(
        "Analysez jusqu'à 10 fichiers CSV/Parquet simultanément. "
        "Les résultats sont indépendants — chaque fichier reçoit son propre `session_id`."
    )

    batch_files = st.file_uploader(
        "Sélectionnez plusieurs fichiers",
        type=["csv", "parquet"],
        accept_multiple_files=True,
        key="batch_uploader",
        help="Max 10 fichiers, 100 MB chacun",
    )

    if batch_files:
        st.info(f"{len(batch_files)} fichier(s) sélectionné(s).")
        if len(batch_files) > 10:
            st.error("Maximum 10 fichiers par batch.")
        elif st.button("🚀 Lancer le batch", key="btn_batch", type="primary"):
            with st.spinner(f"Analyse de {len(batch_files)} fichier(s) en parallèle…"):
                try:
                    batch_resp = requests.post(
                        f"{api_url}/batch",
                        files=[
                            ("files", (f.name, f.getvalue(), f.type))
                            for f in batch_files
                        ],
                        headers=headers,
                        timeout=300,
                    )
                except requests.ConnectionError:
                    st.error("Impossible de joindre l'API.")
                    st.stop()

            if batch_resp.status_code == 200:
                bdata = batch_resp.json()
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Total", bdata["total"])
                bc2.metric("✅ Réussies", bdata["succeeded"])
                bc3.metric("❌ Erreurs", bdata["failed"])

                rows = []
                for item in bdata["results"]:
                    rows.append({
                        "Fichier": item["filename"],
                        "Statut": "✅" if item["status"] == "success" else "❌",
                        "Score": f"{item['quality_score']:.1f}" if item.get("quality_score") is not None else "—",
                        "Issues": item["issues_count"],
                        "Session ID": item.get("session_id") or item.get("error", "—"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                try:
                    detail = batch_resp.json().get("detail", batch_resp.text)
                except Exception:
                    detail = batch_resp.text
                st.error(f"Erreur batch (HTTP {batch_resp.status_code}) : {detail}")
    else:
        st.info("Sélectionnez au moins un fichier pour lancer le batch.")


# ---- JSON brut ----
with tab_json:
    st.json(result)
