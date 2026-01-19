"""
GraphRAG: Embeddings-Based Q&A
Uses KG embeddings (RotatE) to find phones via semantic similarity.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="Embeddings RAG",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Embeddings RAG")
st.markdown("*Ask questions about smartphones using AI-learned embeddings*")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This page uses **Knowledge Graph Embeddings** (RotatE model) to answer questions.

    **How it works:**
    1. Your question is analyzed for intent
    2. The system finds phones similar to your use-case in embedding space
    3. Results are ranked by cosine similarity
    4. An LLM generates a natural response

    **Best for:**
    - Semantic queries ("phones photographers love")
    - Use-case matching (gaming, vlogging)
    - Finding similar phones
    """)

    st.divider()
    st.caption("Requires Ollama running locally")


@st.cache_resource
def load_rag_system():
    """Load the embeddings RAG system (cached)."""
    from src.graphrag.embeddings_rag import KGEmbeddingsRAG
    rag = KGEmbeddingsRAG()
    rag.load()
    return rag


# Initialize chat history
if "emb_messages" not in st.session_state:
    st.session_state.emb_messages = []

# Try to load the RAG system
try:
    with st.spinner("Loading embeddings model..."):
        rag = load_rag_system()

    st.success(f"Model loaded: {len(rag.phone_embeddings_map):,} phone embeddings")

except FileNotFoundError as e:
    st.error("Embeddings model not found. Train it first with:")
    st.code("python -m src.recommandation.train_phones")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Example questions
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

# Display chat history
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

# Chat input
if prompt := st.chat_input("Ask about smartphones..."):
    # Add user message
    st.session_state.emb_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing with embeddings..."):
            try:
                result = rag.query(prompt)

                st.markdown(result.answer)

                # Show matching phones
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

                # Save to history
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

# Clear chat button
if st.session_state.emb_messages:
    if st.button("Clear chat"):
        st.session_state.emb_messages = []
        st.rerun()
