"""
DataSentinel AI — Page Agents Métier

Création et gestion des profils de domaine (F32).
"""

import requests
import streamlit as st

from pages._helpers import api_get, setup_sidebar

st.set_page_config(
    page_title="Agents Métier — DataSentinel AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("🏢 Agents Métier")
st.caption(
    "Créez des profils de validation spécialisés par domaine (RH, Finance, E-Commerce…). "
    "Quand un dataset est analysé, le profil dont les types sémantiques correspondent "
    "le mieux est activé automatiquement (F32)."
)

_SEMANTIC_TYPES_UI = [
    "email", "phone", "first_name", "last_name", "full_name",
    "postal_code", "address", "city", "country",
    "identifier", "monetary_amount", "percentage", "age",
    "date_string", "url", "ip_address",
    "boolean_text", "category", "product_code",
    "employee_id", "description", "free_text",
    "quantity", "rating",
]

# =============================================================================
# CRÉER UN AGENT
# =============================================================================

col_form, col_guide = st.columns([3, 1])

with col_form:
    with st.expander("➕ Créer un agent métier", expanded=True):
        da_name = st.text_input(
            "Nom du domaine *",
            placeholder="Ex: RH, Finance, E-Commerce",
            key="da_name",
        )
        da_desc = st.text_area(
            "Description (optionnelle)",
            placeholder="Ex: Ressources Humaines — salary, hire_date, employee_id",
            key="da_desc", height=68,
        )
        da_trigger = st.multiselect(
            "Types sémantiques déclencheurs *",
            _SEMANTIC_TYPES_UI,
            help="Le profil sera activé si suffisamment de colonnes ont ces types.",
            key="da_trigger",
        )
        da_required = st.multiselect(
            "Types sémantiques requis (absence → CRITICAL)",
            _SEMANTIC_TYPES_UI, key="da_required",
        )
        da_ratio = st.slider(
            "Seuil de détection (%)",
            min_value=10, max_value=100, value=30, step=5, key="da_ratio",
            help="% des types déclencheurs qui doivent être présents pour activer le profil.",
        )
        da_rules_raw = st.text_area(
            "Règles descriptives (1 par ligne)",
            placeholder="Le salaire doit être positif\nL'ID employé est obligatoire et unique",
            key="da_rules_raw", height=100,
        )
        st.caption("Override de sévérité (optionnel) :")
        ov1, ov2 = st.columns(2)
        da_ov_type = ov1.selectbox("Type sémantique", ["(aucun)"] + _SEMANTIC_TYPES_UI, key="da_ov_type")
        da_ov_sev = ov2.selectbox("Sévérité", ["low", "medium", "high", "critical"], index=2, key="da_ov_sev")
        da_overrides: dict = {}
        if da_ov_type != "(aucun)":
            da_overrides[da_ov_type] = da_ov_sev

        if st.button(
            "➕ Créer l'agent",
            key="btn_create_da",
            type="primary",
            disabled=not da_name.strip() or not da_trigger,
        ):
            rules_payload = [
                {"text": line.strip()}
                for line in da_rules_raw.strip().splitlines()
                if line.strip()
            ]
            payload = {
                "name": da_name.strip(),
                "description": da_desc.strip(),
                "trigger_types": da_trigger,
                "min_match_ratio": da_ratio / 100,
                "required_types": da_required,
                "rules": rules_payload,
                "severity_overrides": da_overrides,
            }
            try:
                resp = requests.post(
                    f"{api_url}/domain-agents",
                    json=payload, headers=headers, timeout=10,
                )
                if resp.status_code == 201:
                    created = resp.json()["profile"]
                    st.success(f"✅ Agent **{created['name']}** créé (`{created['domain_id'][:8]}…`)")
                    st.session_state.pop("domain_agents_data", None)
                    st.rerun()
                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    st.error(f"Erreur (HTTP {resp.status_code}) : {detail}")
            except Exception as e:
                st.error(f"Impossible de créer l'agent : {e}")

with col_guide:
    st.markdown("### Guide")
    st.info(
        "**Types déclencheurs** : si leur ratio (colonnes présentes / total décl.) ≥ seuil → agent activé.\n\n"
        "**Types requis** : si absents du dataset → issue `CONSTRAINT_VIOLATION CRITICAL`.\n\n"
        "**Overrides** : force la sévérité des issues sur les colonnes avec ce type sémantique.\n\n"
        "**Règles descriptives** : affichées dans les détails de l'analyse (non exécutées)."
    )

# =============================================================================
# LISTE DES AGENTS
# =============================================================================

st.divider()
da_list_col, da_reload_col = st.columns([4, 1])
da_list_col.subheader("Agents actifs")
if da_reload_col.button("🔄 Rafraîchir", key="btn_reload_da"):
    st.session_state.pop("domain_agents_data", None)

if "domain_agents_data" not in st.session_state:
    resp = api_get(f"{api_url}/domain-agents", headers)
    if resp and resp.status_code == 200:
        st.session_state["domain_agents_data"] = resp.json()
    elif resp:
        st.error(f"Erreur : HTTP {resp.status_code}")

da_data = st.session_state.get("domain_agents_data")
if da_data:
    count = da_data.get("count", 0)
    st.caption(f"**{count}** agent(s) actif(s)")

    if count == 0:
        st.info("Aucun agent métier. Créez-en un ci-dessus.")
    else:
        for da in da_data.get("profiles", []):
            triggers_str = ", ".join(da.get("trigger_types", [])) or "(aucun)"
            with st.expander(f"🏢 **{da['name']}** — déclencheurs : {triggers_str}"):
                info_col, del_col = st.columns([4, 1])
                with info_col:
                    st.markdown(f"**ID** : `{da['domain_id']}`")
                    if da.get("description"):
                        st.markdown(f"**Description** : {da['description']}")
                    st.markdown(f"**Seuil** : {da['min_match_ratio'] * 100:.0f}%")
                    if da.get("required_types"):
                        st.markdown(f"**Types requis** : {', '.join(da['required_types'])}")
                    if da.get("severity_overrides"):
                        overrides_str = ", ".join(f"{t} → {s}" for t, s in da["severity_overrides"].items())
                        st.markdown(f"**Overrides sévérité** : {overrides_str}")
                    if da.get("rules"):
                        st.markdown("**Règles :**")
                        for r in da["rules"]:
                            st.write(f"  • {r['text']}")
                with del_col:
                    if st.button("🗑️ Supprimer", key=f"del_da_{da['domain_id']}"):
                        try:
                            del_r = requests.delete(
                                f"{api_url}/domain-agents/{da['domain_id']}",
                                headers=headers, timeout=10,
                            )
                            if del_r.status_code == 200:
                                st.success(f"Agent `{da['name']}` supprimé.")
                                st.session_state.pop("domain_agents_data", None)
                                st.rerun()
                            else:
                                st.error(f"Erreur : HTTP {del_r.status_code}")
                        except Exception as e:
                            st.error(f"Erreur : {e}")
else:
    st.info("Cliquez sur **Rafraîchir** pour charger les agents.")
