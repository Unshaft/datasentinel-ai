"""
DataSentinel AI — Page Analyse

Upload un fichier et visualise les résultats d'analyse qualité.
"""

import json

import pandas as pd
import requests
import streamlit as st

from pages._helpers import (
    _SEVERITY_ORDER,
    api_get,
    score_badge,
    score_banner,
    setup_sidebar,
    severity_badge,
)

st.set_page_config(
    page_title="Analyse — DataSentinel AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("🔍 Analyse de qualité")
st.caption("Upload un fichier CSV ou Parquet pour détecter les problèmes de qualité.")

# =============================================================================
# UPLOAD
# =============================================================================

uploaded = st.file_uploader(
    "Choisissez un fichier",
    type=["csv", "parquet"],
    help="Formats acceptés : .csv, .parquet — Taille max : 100 MB",
)

col_btn, col_opts = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("🚀 Analyser", type="primary", disabled=uploaded is None)

with col_opts:
    use_react = st.checkbox(
        "Mode ReAct (include_reasoning)",
        value=False,
        help="Active le pipeline adaptatif ReAct avec raisonnement détaillé (F24/F31). Plus lent.",
    )

if uploaded and analyze_btn:
    endpoint = f"{api_url}/upload"
    with st.spinner("Analyse en cours…"):
        try:
            resp = requests.post(
                endpoint,
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                headers=headers,
                timeout=120,
            )
        except requests.ConnectionError:
            st.error("Impossible de joindre l'API — vérifiez que le serveur est lancé.")
            st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"Erreur API (HTTP {resp.status_code}) : {detail}")
        st.stop()

    st.session_state["analysis_result"] = resp.json()
    st.session_state["analysis_session_id"] = resp.json().get("session_id", "")

# =============================================================================
# RÉSULTATS
# =============================================================================

result = st.session_state.get("analysis_result")
session_id = st.session_state.get("analysis_session_id", "")

if result is None:
    st.info("Uploadez un fichier et cliquez sur **Analyser** pour démarrer.")
    st.stop()

# ── Métriques globales ────────────────────────────────────────────────────────

score = result.get("quality_score", 0)
issues_count = len(result.get("issues", []))
proc_ms = result.get("processing_time_ms", 0)

score_banner(score, result.get("summary", ""))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Session", session_id[:20] + "…" if len(session_id) > 20 else session_id)
m2.metric("Problèmes détectés", issues_count)
m3.metric("Temps de traitement", f"{proc_ms} ms")
m4.metric("Statut", result.get("status", "completed").upper())

if result.get("needs_human_review"):
    st.warning(
        "⚠️ **Revue humaine recommandée** — "
        + ", ".join(result.get("escalation_reasons", []))
    )

if result.get("domain_agent"):
    st.info(f"🏢 Agent métier activé : **{result['domain_agent']}**")

# Reflect flags
reflect_flags = result.get("reflect_flags", [])
if reflect_flags:
    flag_labels = {
        "score_vs_critical": "Score colonnes élevé mais issues CRITICAL détectées",
        "plan_blind_spot": "Plan adaptatif a ignoré la détection d'anomalies malgré des issues HIGH/CRITICAL",
    }
    msgs = [flag_labels.get(f, f) for f in reflect_flags]
    st.warning("🔮 **ReAct Reflect** — Incohérences détectées :\n" + "\n".join(f"• {m}" for m in msgs))

dataset_memory = result.get("dataset_memory")
if dataset_memory and dataset_memory.get("is_known"):
    trend_icon = {"improving": "📈", "degrading": "📉", "stable": "➡️", "new": "🆕"}.get(
        dataset_memory.get("trend", "new"), "➡️"
    )
    st.info(
        f"📂 Dataset connu — {dataset_memory['session_count']} analyse(s), "
        f"score moyen : **{dataset_memory['avg_quality_score']:.1f}%**, "
        f"tendance : {trend_icon} **{dataset_memory.get('trend', '?')}**"
    )
    if dataset_memory.get("suggested_rules"):
        with st.expander("💡 Suggestions pro-actives"):
            for r in dataset_memory["suggested_rules"]:
                st.markdown(f"- {r}")

# Copy session ID
with st.expander("🔗 Session ID complet"):
    st.code(session_id)

st.divider()

# =============================================================================
# ONGLETS RÉSULTATS
# =============================================================================

tabs = st.tabs([
    "🐛 Problèmes", "📈 Scores / colonne", "🔧 Corrections",
    "📊 Profil", "🧠 Schéma", "🔮 ReAct",
    "📥 Exports", "🔄 Comparaison", "🔩 JSON brut",
])

tab_issues, tab_scores, tab_corr, tab_profile, tab_schema, tab_react, tab_exports, tab_comp, tab_json = tabs


# ── PROBLÈMES ────────────────────────────────────────────────────────────────
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
        rows = []
        for iss in sorted(issues, key=lambda x: _SEVERITY_ORDER.get(x["severity"], 9)):
            rows.append({
                "Sévérité": severity_badge(iss["severity"]),
                "Type": iss["issue_type"].replace("_", " ").title(),
                "Colonne": iss.get("column") or "—",
                "Description": iss["description"],
                "Affecté": f"{iss['affected_percentage']:.1f}%",
                "Confiance": f"{iss['confidence']:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Détail par issue dans des expanders
        st.divider()
        st.caption("Détails des issues :")
        for iss in sorted(issues, key=lambda x: _SEVERITY_ORDER.get(x["severity"], 9)):
            with st.expander(
                f"{severity_badge(iss['severity'])} — {iss['issue_type'].replace('_', ' ').title()}"
                + (f" ({iss['column']})" if iss.get("column") else "")
            ):
                st.markdown(f"**Description** : {iss['description']}")
                st.markdown(f"**Affecté** : {iss['affected_percentage']:.1f}% — **Confiance** : {iss['confidence']:.0%}")
                if iss.get("details"):
                    st.json(iss["details"])


# ── SCORES PAR COLONNE ───────────────────────────────────────────────────────
with tab_scores:
    col_scores = result.get("column_scores", {})
    if not col_scores:
        st.info("Scores par colonne non disponibles.")
    else:
        rows_scores = sorted(col_scores.items(), key=lambda x: x[1])
        df_scores = pd.DataFrame(
            [{"Colonne": c, "Score": score_badge(s), "Valeur": s} for c, s in rows_scores]
        )

        # Barres de progression visuelles
        for col, s in rows_scores:
            color = "#27ae60" if s >= 80 else "#e67e22" if s >= 60 else "#e74c3c"
            pct = int(s)
            st.markdown(
                f"""
<div style="margin-bottom:8px;">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="width:130px; font-size:.9em; color:#ccc;">{col}</span>
    <div style="flex:1; background:#2c3e50; border-radius:4px; height:20px; overflow:hidden;">
      <div style="width:{pct}%; background:{color}; height:100%; border-radius:4px;
                  display:flex; align-items:center; justify-content:flex-end; padding-right:6px;">
        <span style="font-size:.75em; color:white; font-weight:bold;">{s:.0f}</span>
      </div>
    </div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )


# ── CORRECTIONS ──────────────────────────────────────────────────────────────
with tab_corr:
    corr_resp = api_get(f"{api_url}/analyze/{session_id}/corrections", headers)
    if corr_resp is None:
        pass
    elif corr_resp.status_code == 200:
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
                "Action": e["recommended_action"],
                "Affecté": f"{e['affected_percentage']:.1f}%",
            } for e in auto]), use_container_width=True, hide_index=True)

            st.divider()
            if st.button("🔧 Appliquer et télécharger le CSV corrigé", key="btn_apply"):
                with st.spinner("Application des corrections…"):
                    apply_resp = requests.post(
                        f"{api_url}/analyze/{session_id}/apply-corrections",
                        headers=headers, timeout=60,
                    )
                    if apply_resp.status_code == 200:
                        st.success(
                            f"✅ {apply_resp.headers.get('X-Corrections-Count', '?')} correction(s) — "
                            f"Lignes : {apply_resp.headers.get('X-Rows-Before', '?')} → "
                            f"{apply_resp.headers.get('X-Rows-After', '?')}"
                        )
                        st.download_button(
                            "📥 Télécharger le CSV corrigé",
                            data=apply_resp.content,
                            file_name=f"corrected_{session_id}.csv",
                            mime="text/csv",
                        )
                    elif apply_resp.status_code == 404:
                        st.warning("Session expirée. Relancez l'analyse.")
                    else:
                        st.error(f"Erreur : HTTP {apply_resp.status_code}")

        if manual:
            st.subheader(f"👁️ Revue manuelle requise ({len(manual)})")
            st.dataframe(pd.DataFrame([{
                "Type": e["issue_type"].replace("_", " ").title(),
                "Sévérité": e["severity"].upper(),
                "Colonne": e.get("column") or "—",
                "Action": e["recommended_action"],
            } for e in manual]), use_container_width=True, hide_index=True)

        if not auto and not manual:
            st.success("✅ Aucune correction nécessaire.")
    elif corr_resp.status_code == 404:
        st.warning("Session expirée ou serveur redémarré. Relancez l'analyse.")
    else:
        st.error(f"Erreur récupération plan : HTTP {corr_resp.status_code}")


# ── PROFIL ───────────────────────────────────────────────────────────────────
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


# ── SCHÉMA SÉMANTIQUE ────────────────────────────────────────────────────────
with tab_schema:
    st.markdown("### Schéma sémantique (F27v2)")
    semantic_types = result.get("semantic_types")

    if not semantic_types:
        st.info(
            "Types sémantiques non disponibles.\n\n"
            "Pour activer la classification LLM : `ENABLE_LLM_CHECKS=true` dans `.env`\n\n"
            "*(La classification heuristique fonctionne toujours — elle n'est pas retournée ici mais utilisée en interne.)*"
        )
    else:
        st.caption(f"{len(semantic_types)} colonne(s) classifiée(s)")
        rows = []
        for col_name, sem_info in semantic_types.items():
            conf = sem_info.get("confidence", 0.0)
            badge = "🟢" if conf >= 0.9 else "🟡" if conf >= 0.7 else "🔴"
            rows.append({
                "Colonne": col_name,
                "Type sémantique": sem_info.get("semantic_type", "—"),
                "Confiance": f"{badge} {conf:.0%}",
                "Méthode": sem_info.get("method", "—"),
                "Langue": sem_info.get("language") or "—",
                "Pattern": sem_info.get("pattern") or "—",
                "Notes": (sem_info.get("notes") or "")[:60],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if session_id:
            st.divider()
            if st.button("📥 Charger le schéma complet via /schema"):
                schema_resp = api_get(f"{api_url}/analyze/{session_id}/schema", headers)
                if schema_resp and schema_resp.status_code == 200:
                    schema_data = schema_resp.json()
                    st.metric("Couverture sémantique", f"{schema_data.get('semantic_coverage', 0):.1f}%")
                    st.json(schema_data)
                    st.download_button(
                        "⬇ schema.json",
                        data=json.dumps(schema_data, indent=2, ensure_ascii=False),
                        file_name=f"schema_{session_id}.json",
                        mime="application/json",
                    )
                elif schema_resp:
                    st.error(f"Erreur : HTTP {schema_resp.status_code}")


# ── REACT (F24 / F31) ────────────────────────────────────────────────────────
with tab_react:
    st.markdown("### Raisonnement ReAct (F24 / F31)")

    reasoning = result.get("reasoning_steps", [])
    reflect_flags = result.get("reflect_flags", [])
    exec_plan = None

    if not reasoning:
        st.info(
            "Raisonnement non disponible pour cette analyse.\n\n"
            "Pour l'activer, cochez **Mode ReAct** dans les options ci-dessus et relancez."
        )
    else:
        # Étapes du raisonnement
        _phase_icons = {
            "observe": "👁️",
            "reason": "🧠",
            "act": "⚡",
            "reflect": "🔮",
        }
        for step in reasoning:
            phase = step.get("phase", "?")
            icon = _phase_icons.get(phase, "•")
            with st.expander(
                f"{icon} **Étape {step.get('step', '?')} — {phase.upper()}** : {step.get('thought', '')[:80]}",
                expanded=(phase == "reflect"),
            ):
                st.markdown(f"**Pensée** : {step.get('thought', '—')}")
                st.markdown(f"**Action** : `{step.get('action', '—')}`")
                st.markdown(f"**Observation** : {step.get('observation', '—')}")

            # Capturer le plan d'exécution depuis l'étape "reason"
            if phase == "reason" and not exec_plan:
                obs = step.get("observation", "")
                if "Plan:" in obs:
                    exec_plan = obs.split("Plan:")[-1].strip()

    if reflect_flags:
        st.divider()
        st.subheader("🔮 Flags d'incohérence (Reflect)")
        _flag_explain = {
            "score_vs_critical": (
                "**score_vs_critical** — Le score moyen des colonnes est élevé (≥80) "
                "mais des issues CRITICAL ont été détectées (ex: règles domaine ajoutées après le scoring)."
            ),
            "plan_blind_spot": (
                "**plan_blind_spot** — Le plan adaptatif a exclu `detect_anomalies` "
                "mais des issues HIGH/CRITICAL ont tout de même été trouvées."
            ),
        }
        for flag in reflect_flags:
            st.warning(_flag_explain.get(flag, f"⚠️ {flag}"))
    elif reasoning:
        st.success("✅ Aucune incohérence détectée par la phase Reflect.")

    if exec_plan:
        st.divider()
        st.subheader("📋 Plan d'exécution adaptatif")
        st.code(exec_plan)


# ── EXPORTS ──────────────────────────────────────────────────────────────────
with tab_exports:
    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("### 📄 Rapport PDF")
        if st.button("⬇️ Générer le PDF", key="btn_pdf"):
            with st.spinner("Génération…"):
                pdf_resp = api_get(f"{api_url}/analyze/{session_id}/report.pdf", headers, timeout=30)
                if pdf_resp and pdf_resp.status_code == 200:
                    st.download_button(
                        "📥 Télécharger le PDF",
                        data=pdf_resp.content,
                        file_name=f"rapport_{session_id}.pdf",
                        mime="application/pdf",
                    )
                elif pdf_resp and pdf_resp.status_code == 404:
                    st.warning("Session expirée.")
                elif pdf_resp:
                    st.error(f"Erreur : HTTP {pdf_resp.status_code}")

    with ex2:
        st.markdown("### 📊 Rapport Excel")
        if st.button("⬇️ Générer l'Excel", key="btn_xlsx"):
            with st.spinner("Génération…"):
                xlsx_resp = api_get(f"{api_url}/analyze/{session_id}/report.xlsx", headers, timeout=30)
                if xlsx_resp and xlsx_resp.status_code == 200:
                    st.download_button(
                        "📊 Télécharger l'Excel",
                        data=xlsx_resp.content,
                        file_name=f"rapport_{session_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                elif xlsx_resp and xlsx_resp.status_code == 404:
                    st.warning("Session expirée.")
                elif xlsx_resp:
                    st.error(f"Erreur : HTTP {xlsx_resp.status_code}")


# ── COMPARAISON ──────────────────────────────────────────────────────────────
with tab_comp:
    st.markdown("### 🔄 Comparaison avant/après corrections")
    st.caption("Applique les corrections en mémoire et compare les scores (stateless).")
    if st.button("Calculer", key="btn_comparison"):
        with st.spinner("Comparaison…"):
            comp_resp = api_get(f"{api_url}/analyze/{session_id}/comparison", headers, timeout=60)
        if comp_resp and comp_resp.status_code == 200:
            cdata = comp_resp.json()
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Score avant", f"{cdata['score_before']:.1f}")
            cc2.metric("Score après", f"{cdata['score_after']:.1f}")
            delta = cdata["delta"]
            cc3.metric("Delta", f"{delta:+.1f}", delta_color="normal")

            col_a, col_b = st.columns(2)
            with col_a:
                removed = cdata.get("issues_removed", [])
                st.subheader(f"✅ Issues supprimées ({len(removed)})")
                for t in removed:
                    st.write(f"• {t.replace('_', ' ').title()}")
                if not removed:
                    st.info("Aucune issue supprimée.")
            with col_b:
                remaining = cdata.get("issues_remaining", [])
                st.subheader(f"⚠️ Issues restantes ({len(remaining)})")
                for t in remaining:
                    st.write(f"• {t.replace('_', ' ').title()}")
                if not remaining:
                    st.success("Toutes corrigées.")
        elif comp_resp and comp_resp.status_code == 404:
            st.warning("Session expirée.")
        elif comp_resp:
            st.error(f"Erreur : HTTP {comp_resp.status_code}")


# ── JSON BRUT ─────────────────────────────────────────────────────────────────
with tab_json:
    st.json(result)
