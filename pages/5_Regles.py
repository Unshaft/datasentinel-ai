"""
DataSentinel AI — Page Règles métier

CRUD des règles utilisées par l'Active RAG (F20/F25).
"""

import requests
import streamlit as st

from pages._helpers import api_get, setup_sidebar

st.set_page_config(
    page_title="Règles — DataSentinel AI",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_url, headers = setup_sidebar()

st.title("📋 Règles métier")
st.caption(
    "Les règles métier sont utilisées par le QualityAgent (Active RAG — F25) pour ajuster "
    "les seuils de détection. Elles sont stockées dans ChromaDB."
)

# =============================================================================
# AJOUTER UNE RÈGLE
# =============================================================================

with st.expander("➕ Ajouter une règle", expanded=True):
    new_rule_text = st.text_area(
        "Texte de la règle (min. 10 caractères)",
        placeholder="Ex : La colonne 'email' est obligatoire et doit être unique",
        key="new_rule_text",
    )
    r1, r2, r3 = st.columns(3)
    rule_type = r1.selectbox("Type", ["constraint", "validation", "format", "consistency"], key="rule_type")
    rule_severity = r2.selectbox("Sévérité", ["low", "medium", "high", "critical"], index=1, key="rule_severity")
    rule_category = r3.text_input("Catégorie", value="general", key="rule_category")

    disabled_add = not (new_rule_text.strip()) or len(new_rule_text.strip()) < 10
    if st.button("➕ Ajouter", key="btn_add_rule", type="primary", disabled=disabled_add):
        try:
            add_resp = requests.post(
                f"{api_url}/rules",
                json={
                    "rule_text": new_rule_text.strip(),
                    "rule_type": rule_type,
                    "severity": rule_severity,
                    "category": rule_category,
                },
                headers=headers,
                timeout=10,
            )
            if add_resp.status_code == 201:
                st.success(f"Règle ajoutée : `{add_resp.json()['rule']['rule_id']}`")
                st.session_state.pop("rules_data", None)
                st.rerun()
            else:
                try:
                    detail = add_resp.json().get("detail", add_resp.text)
                except Exception:
                    detail = add_resp.text
                st.error(f"Erreur (HTTP {add_resp.status_code}) : {detail}")
        except Exception as e:
            st.error(f"Impossible d'ajouter la règle : {e}")

# =============================================================================
# FILTRES + LISTE
# =============================================================================

st.divider()
f1, f2, f3 = st.columns([2, 1, 1])
with f1:
    st.subheader("Règles actives")
with f2:
    filter_type = st.selectbox(
        "Filtrer par type",
        ["Tous", "constraint", "validation", "format", "consistency", "exception", "example"],
        key="filter_rule_type",
    )
with f3:
    if st.button("🔄 Recharger", key="btn_reload_rules"):
        st.session_state.pop("rules_data", None)

if "rules_data" not in st.session_state:
    params = {} if filter_type == "Tous" else {"rule_type": filter_type}
    resp = api_get(f"{api_url}/rules", headers)
    if resp and resp.status_code == 200:
        rdata = resp.json()
        # Apply local filter (rules endpoint may not support query param)
        if filter_type != "Tous":
            rdata["rules"] = [r for r in rdata.get("rules", []) if r.get("rule_type") == filter_type]
            rdata["count"] = len(rdata["rules"])
        st.session_state["rules_data"] = rdata
    elif resp:
        st.error(f"Erreur chargement règles : HTTP {resp.status_code}")

rules_data = st.session_state.get("rules_data")
if rules_data:
    count = rules_data.get("count", 0)
    st.caption(f"**{count}** règle(s) active(s)" + (f" — filtrées par type : {filter_type}" if filter_type != "Tous" else ""))

    if count == 0:
        st.info("Aucune règle active. Ajoutez-en une ci-dessus.")
    else:
        for rule in rules_data.get("rules", []):
            rule_label = f"[{rule.get('rule_type', '?').upper()}] {rule.get('text', '')[:80]}"
            with st.expander(rule_label):
                info_col, del_col = st.columns([4, 1])
                with info_col:
                    st.markdown(f"**ID** : `{rule['rule_id']}`")
                    st.markdown(f"**Texte** : {rule['text']}")
                    st.markdown(
                        f"**Type** : {rule.get('rule_type')}  |  "
                        f"**Sévérité** : {rule.get('severity')}  |  "
                        f"**Catégorie** : {rule.get('category')}"
                    )
                with del_col:
                    if st.button("🗑️ Désactiver", key=f"del_rule_{rule['rule_id']}"):
                        try:
                            del_resp = requests.delete(
                                f"{api_url}/rules/{rule['rule_id']}",
                                headers=headers, timeout=10,
                            )
                            if del_resp.status_code == 200:
                                st.success(f"Règle `{rule['rule_id']}` désactivée.")
                                st.session_state.pop("rules_data", None)
                                st.rerun()
                            else:
                                st.error(f"Erreur : HTTP {del_resp.status_code}")
                        except Exception as e:
                            st.error(f"Erreur : {e}")
else:
    st.info("Cliquez sur **Recharger** pour charger les règles.")

# =============================================================================
# GUIDE
# =============================================================================

st.divider()
with st.expander("ℹ️ À quoi servent les règles ?"):
    st.markdown("""
Les règles métier alimentent le mécanisme **Active RAG** (F25) :

1. Avant chaque analyse, le QualityAgent interroge ChromaDB pour trouver les règles **les plus proches sémantiquement** du dataset en cours.
2. Les règles trouvées peuvent **ajuster les seuils** (ex: `null_threshold`, `severity`) ou **ajouter du contexte** à l'explication.
3. Les règles de type `exception` créées via le **feedback** (F26) permettent de dire au système "ne pas signaler X comme erreur".

**Types de règles** :
- `constraint` : Contrainte sur les valeurs (ex: "L'âge doit être entre 0 et 150")
- `validation` : Validation logique (ex: "La date de fin doit être après la date de début")
- `format` : Format attendu (ex: "Le téléphone doit être au format +33XXXXXXXXX")
- `consistency` : Cohérence entre colonnes (ex: "Si statut=actif, date_sortie doit être null")
""")
