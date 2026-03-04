"""
DataSentinel AI — Page Feedback (NEW)

Formulaire de feedback pour corriger les décisions du système et alimenter
le mécanisme d'apprentissage continu (F26).
"""

import requests
import streamlit as st

from pages._helpers import _SEVERITY_ORDER, setup_sidebar

st.set_page_config(
    page_title="Feedback — DataSentinel AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("💬 Feedback")
st.caption(
    "Corrigez les décisions du système pour améliorer les analyses futures. "
    "Chaque feedback alimente le mécanisme d'apprentissage continu (F26)."
)

# =============================================================================
# CONTEXTE : résultats de la dernière analyse
# =============================================================================

result = st.session_state.get("analysis_result")
session_id = st.session_state.get("analysis_session_id", "")

if result is None:
    st.info(
        "Aucune analyse en mémoire. Naviguez vers **Analyse**, uploadez un fichier "
        "et revenez ici pour donner votre feedback."
    )
    st.divider()
    st.markdown("### Feedback manuel (session ID)")
    _manual_session_id = st.text_input("Session ID", placeholder="session_xxxxxxxxxxxx", key="fb_manual_session")
    if _manual_session_id:
        st.session_state["fb_target_session"] = _manual_session_id
else:
    st.success(
        f"Session en mémoire : `{session_id}` — "
        f"Score : **{result.get('quality_score', 0):.1f}** — "
        f"**{len(result.get('issues', []))}** issue(s)"
    )

target_session_id = session_id or st.session_state.get("fb_target_session", "")

if not target_session_id:
    st.stop()

# =============================================================================
# ONGLETS : Issue / Proposal / Décision libre
# =============================================================================

tab_issue, tab_proposal, tab_free = st.tabs(
    ["🐛 Feedback sur une issue", "📝 Feedback sur une proposition", "💡 Feedback libre"]
)


# ── FEEDBACK SUR UNE ISSUE ───────────────────────────────────────────────────
with tab_issue:
    st.subheader("Corriger une détection d'issue")
    st.caption(
        "Utilisez ce formulaire si le système a **mal détecté** un problème "
        "(faux positif ou manqué)."
    )

    issues = result.get("issues", []) if result else []

    if not issues:
        st.info("Aucune issue dans la session courante.")
    else:
        # Sélecteur d'issue
        issue_options = {
            f"{i['severity'].upper()} — {i['issue_type'].replace('_', ' ').title()}"
            + (f" ({i['column']})" if i.get("column") else "")
            + f" [ID: {i['issue_id'][:8]}…]": i["issue_id"]
            for i in sorted(issues, key=lambda x: _SEVERITY_ORDER.get(x["severity"], 9))
        }
        selected_label = st.selectbox("Sélectionnez l'issue", list(issue_options.keys()), key="fb_issue_select")
        selected_issue_id = issue_options[selected_label]

        # Détail de l'issue sélectionnée
        selected_issue = next((i for i in issues if i["issue_id"] == selected_issue_id), None)
        if selected_issue:
            with st.expander("Détails de l'issue", expanded=False):
                st.json(selected_issue)

        fb_correct = st.radio(
            "Cette détection est-elle correcte ?",
            ["✅ Oui, c'est correct", "❌ Non, c'est un faux positif", "⚠️ Partiellement correct"],
            key="fb_issue_correct",
        )
        is_correct = fb_correct.startswith("✅")

        fb_correction = st.text_area(
            "Correction / explication (optionnel)",
            placeholder="Ex: Cette colonne ne contient pas d'emails, c'est un identifiant interne.",
            key="fb_issue_correction",
        )
        fb_comments = st.text_area(
            "Commentaires supplémentaires (optionnel)",
            key="fb_issue_comments",
        )

        if st.button("📤 Envoyer le feedback", key="btn_fb_issue", type="primary"):
            payload = {
                "session_id": target_session_id,
                "target_id": selected_issue_id,
                "target_type": "issue",
                "is_correct": is_correct,
                "user_correction": fb_correction.strip() or None,
                "comments": fb_comments.strip() or None,
            }
            try:
                resp = requests.post(
                    f"{api_url}/feedback",
                    json=payload, headers=headers, timeout=15,
                )
                if resp.status_code == 200:
                    rd = resp.json()
                    st.success(f"✅ Feedback enregistré — ID : `{rd.get('feedback_id', '?')}`")
                    st.info(f"💡 Impact : {rd.get('impact', '—')}")
                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    st.error(f"Erreur (HTTP {resp.status_code}) : {detail}")
            except Exception as e:
                st.error(f"Impossible d'envoyer le feedback : {e}")


# ── FEEDBACK SUR UNE PROPOSITION ─────────────────────────────────────────────
with tab_proposal:
    st.subheader("Évaluer une proposition de correction")
    st.caption("Feedback sur une correction proposée par le système (via /analyze/{id}/corrections).")

    proposal_id = st.text_input(
        "ID de la proposition",
        placeholder="proposal_xxxxxxxxxxxx",
        key="fb_proposal_id",
    )
    fb_prop_correct = st.radio(
        "La proposition est-elle appropriée ?",
        ["✅ Oui, appropriée", "❌ Non, incorrecte", "⚠️ Partiellement bonne"],
        key="fb_prop_correct",
    )
    is_prop_correct = fb_prop_correct.startswith("✅")
    fb_prop_correction = st.text_area(
        "Meilleure correction (optionnel)",
        placeholder="Ex: Au lieu de supprimer la ligne, il faut imputer la médiane.",
        key="fb_prop_correction",
    )
    fb_prop_comments = st.text_area("Commentaires", key="fb_prop_comments")

    if st.button("📤 Envoyer", key="btn_fb_proposal", type="primary", disabled=not proposal_id.strip()):
        payload = {
            "session_id": target_session_id,
            "target_id": proposal_id.strip(),
            "target_type": "proposal",
            "is_correct": is_prop_correct,
            "user_correction": fb_prop_correction.strip() or None,
            "comments": fb_prop_comments.strip() or None,
        }
        try:
            resp = requests.post(
                f"{api_url}/feedback",
                json=payload, headers=headers, timeout=15,
            )
            if resp.status_code == 200:
                rd = resp.json()
                st.success(f"✅ Feedback enregistré — ID : `{rd.get('feedback_id', '?')}`")
                st.info(f"💡 Impact : {rd.get('impact', '—')}")
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"Erreur (HTTP {resp.status_code}) : {detail}")
        except Exception as e:
            st.error(f"Erreur : {e}")


# ── FEEDBACK LIBRE ────────────────────────────────────────────────────────────
with tab_free:
    st.subheader("Feedback sur une décision")
    st.caption("Feedback libre sur n'importe quelle décision du système.")

    fb_target_id = st.text_input(
        "ID de la cible (issue_id, proposal_id, ou libre)",
        placeholder="Ex: issue_abc123 ou 'decision_globale'",
        key="fb_free_target_id",
    )
    fb_free_correct = st.radio(
        "La décision est-elle correcte ?",
        ["✅ Oui", "❌ Non"],
        key="fb_free_correct",
    )
    is_free_correct = fb_free_correct.startswith("✅")
    fb_free_correction = st.text_area("Correction proposée (optionnel)", key="fb_free_correction")
    fb_free_comments = st.text_area("Commentaires", key="fb_free_comments")

    if st.button("📤 Envoyer", key="btn_fb_free", type="primary", disabled=not fb_target_id.strip()):
        payload = {
            "session_id": target_session_id,
            "target_id": fb_target_id.strip(),
            "target_type": "decision",
            "is_correct": is_free_correct,
            "user_correction": fb_free_correction.strip() or None,
            "comments": fb_free_comments.strip() or None,
        }
        try:
            resp = requests.post(
                f"{api_url}/feedback",
                json=payload, headers=headers, timeout=15,
            )
            if resp.status_code == 200:
                rd = resp.json()
                st.success(f"✅ Enregistré — ID : `{rd.get('feedback_id', '?')}`")
                st.info(f"💡 Impact : {rd.get('impact', '—')}")
            else:
                st.error(f"Erreur : HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"Erreur : {e}")

# =============================================================================
# STATS FEEDBACK
# =============================================================================

st.divider()
with st.expander("📊 Statistiques des feedbacks"):
    if st.button("🔄 Charger les stats feedback", key="btn_fb_stats"):
        try:
            resp = requests.get(f"{api_url}/feedback/stats", headers=headers, timeout=10)
            if resp.status_code == 200:
                fstats = resp.json().get("stats", {})
                f1, f2, f3 = st.columns(3)
                f1.metric("Total feedbacks", fstats.get("total_feedback", 0))
                f2.metric("Taux d'exactitude", f"{fstats.get('accuracy_rate', 0):.0%}")
                f3.metric("Ajustements de confiance", len(fstats.get("confidence_adjustments", {})))
                if fstats.get("confidence_adjustments"):
                    st.subheader("Ajustements par type d'issue")
                    for issue_type, adj in fstats["confidence_adjustments"].items():
                        st.markdown(f"- **{issue_type}** : `{adj:.3f}`")
            else:
                st.error(f"Erreur : HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"Erreur : {e}")

# Guide
st.divider()
with st.expander("ℹ️ Comment fonctionne le feedback ?"):
    st.markdown("""
Le feedback déclenche le mécanisme d'**apprentissage continu** (F26) :

- **Faux positif** : crée une règle d'exception dans ChromaDB pour que le système
  ignore ce type de détection sur ce contexte à l'avenir.
- **Confirmation** (feedback correct=True) : augmente le score de confiance pour ce type d'issue
  (plafonné à 0.99).
- **Correction** : crée un exemple de correction dans ChromaDB pour guider
  les propositions futures.

Les ajustements de confiance impactent les futures analyses via l'**Active RAG** (F25).
""")
