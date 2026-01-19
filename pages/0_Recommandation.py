"""
Recommandation - Use case based phone recommendations.
"""

import streamlit as st
from pathlib import Path
import sys
import json

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Recommandation",
    page_icon="phone",
    layout="wide"
)

USE_CASES = ["Gaming", "Photography", "Vlogging", "Business", "EverydayUse", "MinimalistUse"]


@st.cache_resource
def load_recommendation_system():
    """Load the recommendation model (cached)."""
    import torch
    from pykeen.triples import TriplesFactory

    model_dir = ROOT_DIR / "output" / "models" / "link_prediction"
    data_dir = ROOT_DIR / "data"

    model = torch.load(model_dir / "trained_model.pkl", weights_only=False)
    tf = TriplesFactory.from_path_binary(model_dir / "training_triples")

    # Load phone configs with specs (keyed by full config_id)
    configs_file = data_dir / "preprocessed" / "phones_configuration.json"
    with open(configs_file, "r") as f:
        configs = json.load(f)
    phone_specs = {p["phone_id"]: p for p in configs}

    return model, tf, phone_specs


def get_recommendations(interests: list[str], top_k: int = 10) -> list[dict]:
    """Get phone recommendations with specs for given interests."""
    import torch

    model, tf, phone_specs = load_recommendation_system()
    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    if "suitableFor" not in relation_to_id:
        return []

    valid_interests = [i for i in interests if i in entity_to_id]
    if not valid_interests:
        return []

    all_config_ids = [e for e in entity_to_id.keys() if e.endswith("gb")]
    if not all_config_ids:
        return []

    suitable_rel_idx = relation_to_id["suitableFor"]
    aggregate_scores: dict[str, float] = {cid: 0.0 for cid in all_config_ids}
    config_indices = torch.tensor([entity_to_id[cid] for cid in all_config_ids], dtype=torch.long)

    for interest in valid_interests:
        interest_idx = entity_to_id[interest]
        h = config_indices
        r = torch.full((len(config_indices),), suitable_rel_idx, dtype=torch.long)
        t = torch.full((len(config_indices),), interest_idx, dtype=torch.long)
        hrt_batch = torch.stack([h, r, t], dim=1)
        scores = model.score_hrt(hrt_batch).squeeze()

        for config_id, score in zip(all_config_ids, scores):
            aggregate_scores[config_id] += score.item()

    sorted_configs = sorted(aggregate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for config_id, score in sorted_configs:
        specs = phone_specs.get(config_id, {})
        phone_name = specs.get("phone_name", config_id)
        storage = specs.get("storage_gb", "")
        ram = specs.get("ram_gb", "")

        display_name = f"{phone_name} ({storage}GB/{ram}GB)" if storage and ram else phone_name

        results.append({
            "Phone": display_name,
            "Brand": specs.get("brand", ""),
            "Year": specs.get("year") or "-",
            "Screen": specs.get("screen_size_inches") or "-",
            "Battery": specs.get("battery_mah") or "-",
            "Camera": specs.get("main_camera_mp") or "-",
            "Selfie": specs.get("selfie_camera_mp") or "-",
            "Refresh": specs.get("refresh_rate_hz") or "-",
            "Chipset": specs.get("chipset") or "-",
            "Display": specs.get("display_type") or "-",
            "5G": "Yes" if specs.get("supports_5g") else "No",
            "NFC": "Yes" if specs.get("nfc") else "No",
            "Score": round(score, 2),
        })

    return results


# Initialize session state
if "selected_use_case" not in st.session_state:
    st.session_state.selected_use_case = None

# Main content
st.title("Recommandation")

st.subheader("Select your use case")

cols = st.columns(3)
for i, use_case in enumerate(USE_CASES):
    with cols[i % 3]:
        is_selected = st.session_state.selected_use_case == use_case

        with st.container(border=True):
            st.markdown(f"### {use_case}")

            if st.button(
                "Selected" if is_selected else "Select",
                key=f"btn_{use_case}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_use_case = use_case
                st.rerun()

# Show recommendations table if a use case is selected
if st.session_state.selected_use_case:
    st.divider()
    st.subheader(f"Top phones for {st.session_state.selected_use_case}")

    with st.spinner("Loading recommendations..."):
        recommendations = get_recommendations([st.session_state.selected_use_case], top_k=10)

    if recommendations:
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not load recommendations. Check if the model is trained.")
