"""
GraphRAG: Natural Language to SPARQL
Transforms natural language questions into SPARQL queries using LLM.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(
    page_title="NL to SPARQL",
    page_icon="💬",
    layout="wide"
)

st.title("💬 NL to SPARQL")
st.markdown("*Ask questions in natural language, see the generated SPARQL query*")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This page converts **natural language** questions into **SPARQL queries**.

    **How it works:**
    1. Your question is sent to an LLM (Ollama)
    2. The LLM generates a SPARQL query
    3. The query runs against the Knowledge Graph
    4. Results are displayed

    **Best for:**
    - Precise queries ("phones with 5G and 120Hz")
    - Filtering by specs (battery > 5000mAh)
    - Comparing specific phones
    - Understanding the data structure
    """)

    st.divider()
    st.caption("Requires Ollama running locally")


@st.cache_resource
def load_nl2sparql():
    """Load the NL to SPARQL system (cached)."""
    from src.graphrag.nl_to_sparql import NLToSPARQL
    return NLToSPARQL()


# Initialize chat history
if "sparql_messages" not in st.session_state:
    st.session_state.sparql_messages = []

# Try to load the system
try:
    with st.spinner("Connecting to Ollama..."):
        nl2sparql = load_nl2sparql()

    st.success("Connected to Ollama")

except ConnectionError as e:
    st.error(f"Cannot connect to Ollama: {e}")
    st.info("Make sure Ollama is running: `ollama serve`")
    st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Example questions
with st.expander("Example questions"):
    examples = [
        "What Samsung phones have battery over 5000mAh?",
        "Find phones good for gaming (high refresh rate)",
        "What are the top 5 phones with the best camera?",
        "Which brands have 5G phones?",
        "Find phones released in 2023 with AMOLED display",
        "List flagship phones with price over 900 EUR",
        "Show phones with 256GB storage under 500 EUR",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"sparql_ex_{i}"):
            st.session_state.sparql_messages.append({"role": "user", "content": example})
            st.rerun()

# Display chat history
for message in st.session_state.sparql_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sparql" in message:
            with st.expander("Generated SPARQL"):
                st.code(message["sparql"], language="sparql")
        if "results" in message and message["results"]:
            with st.expander(f"Results ({len(message['results'])} rows)"):
                st.dataframe(message["results"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask about smartphones..."):
    # Add user message
    st.session_state.sparql_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating SPARQL query..."):
            try:
                result = nl2sparql.query(prompt)

                if result.success:
                    # Show summary
                    if result.results:
                        st.markdown(f"Found **{len(result.results)}** results")
                    else:
                        st.markdown("No results found for this query.")

                    # Show SPARQL
                    with st.expander("Generated SPARQL", expanded=True):
                        st.code(result.sparql_query, language="sparql")

                    # Show results
                    if result.results:
                        with st.expander("Results", expanded=True):
                            st.dataframe(result.results, use_container_width=True)

                    # Save to history
                    st.session_state.sparql_messages.append({
                        "role": "assistant",
                        "content": f"Found **{len(result.results)}** results" if result.results else "No results found.",
                        "sparql": result.sparql_query,
                        "results": result.results
                    })
                else:
                    error_msg = f"Query failed: {result.error}"
                    st.error(error_msg)

                    if result.sparql_query:
                        with st.expander("Generated SPARQL (failed)"):
                            st.code(result.sparql_query, language="sparql")

                    st.session_state.sparql_messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sparql": result.sparql_query,
                        "results": []
                    })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.sparql_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Clear chat button
if st.session_state.sparql_messages:
    if st.button("Clear chat"):
        st.session_state.sparql_messages = []
        st.rerun()
