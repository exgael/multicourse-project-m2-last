"""
Embeddings RAG - Q&A using KG embeddings.
"""

import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Embeddings RAG",
    page_icon="phone",
    layout="wide"
)

st.title("Embeddings RAG")


@st.cache_resource
def load_rag_system():
    """Load the embeddings RAG system (cached)."""
    from src.graphrag.embeddings_rag import KGEmbeddingsRAG
    rag = KGEmbeddingsRAG()
    rag.load()
    return rag


if "emb_messages" not in st.session_state:
    st.session_state.emb_messages = []

try:
    with st.spinner("Loading embeddings model..."):
        rag = load_rag_system()

    st.success(f"Model loaded: {len(rag.phone_embeddings_map):,} phone embeddings")

except FileNotFoundError:
    st.error("Embeddings model not found. Train it first:")
    st.code("python -m src.recommandation.train_phones")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

with st.expander("Example questions"):
    examples = [
        "I need a phone for professional mobile gaming",
        "Best Samsung phones for vlogging",
        "A phone that photographers would love",
        "Good for both gaming and taking photos",
        "What would a content creator choose?",
        "Best flagship for business use",
        "Something reliable for everyday use with good battery",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"ex_{i}"):
            st.session_state.emb_messages.append({"role": "user", "content": example})
            st.rerun()

for message in st.session_state.emb_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "phones" in message:
            with st.expander("Matching phones"):
                for phone in message["phones"]:
                    specs = []
                    if phone.battery:
                        specs.append(f"{phone.battery}mAh")
                    if phone.camera:
                        specs.append(f"{phone.camera}MP")
                    if phone.refresh_rate:
                        specs.append(f"{phone.refresh_rate}Hz")
                    if phone.supports_5g:
                        specs.append("5G")
                    st.write(f"- **{phone.name}** ({phone.brand}) - {' | '.join(specs)}")

if prompt := st.chat_input("Ask about smartphones..."):
    st.session_state.emb_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = rag.query(prompt)

                st.markdown(result.answer)

                if result.relevant_phones:
                    with st.expander(f"Top {len(result.relevant_phones)} matching phones"):
                        for phone in result.relevant_phones:
                            specs = []
                            if phone.battery:
                                specs.append(f"{phone.battery}mAh")
                            if phone.camera:
                                specs.append(f"{phone.camera}MP")
                            if phone.refresh_rate:
                                specs.append(f"{phone.refresh_rate}Hz")
                            if phone.supports_5g:
                                specs.append("5G")
                            st.write(f"- **{phone.name}** ({phone.brand}) - {' | '.join(specs)}")

                st.session_state.emb_messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "phones": result.relevant_phones
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.emb_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

if st.session_state.emb_messages:
    if st.button("Clear chat"):
        st.session_state.emb_messages = []
        st.rerun()
