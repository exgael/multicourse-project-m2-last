import streamlit as st
import torch
import json
import re
from pathlib import Path
from rdflib import Graph
import sys
import os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
TTL_FILE = OUTPUT_DIR / "final_knowledge_graph.ttl"

# --- 1. Custom Agent Wrapper ---
try:
    from agents import DefaultAgent
except ImportError:
    # Fallback mock for UI testing
    class DefaultAgent:
        def action(self, sys_prompt, user_prompt):
            # If asking for JSON map
            if "JSON list" in sys_prompt:
                return '["vocab/Gaming", "vocab/price_budget"]'
            # If asking for explanation
            return "Based on our analysis, this phone offers the best trade-off between battery life and price."

# --- 2. The Engine (Same logic, adapted for Streamlit) ---
class KGRecommender:
    def __init__(self):
        # 1. Load Maps
        with open(OUTPUT_DIR / "entity_to_id.json") as f:
            self.ent2id = json.load(f)
        with open(OUTPUT_DIR / "relation_to_id.json") as f:
            self.rel2id = json.load(f)
            
        # 2. Load Model (CPU for safety)
        self.model = torch.load(OUTPUT_DIR / "model.pkl", map_location="cpu", weights_only=False)
        self.model.eval()

        # 3. Load Names
        self.names = self._load_names()

    def _load_names(self):
        if not TTL_FILE.exists(): return {}
        g = Graph()
        g.parse(TTL_FILE, format="turtle")
        names = {}
        q = """
        PREFIX sp: <http://example.org/smartphone#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?uri ?label WHERE {
            { ?uri sp:phoneName ?label } UNION { ?uri rdfs:label ?label }
        }
        """
        for row in g.query(q):
            uri_str = str(row.uri)
            label = str(row.label)
            names[uri_str] = label
            names[uri_str.split("/")[-1]] = label 
            names["/".join(uri_str.split("/")[-2:])] = label
        return names

    def get_valid_concepts(self):
        return [k for k in self.ent2id.keys() if "vocab/" in k or "price_" in k]

    def recommend(self, concepts, top_k=3):
        c_ids = [self.ent2id[c] for c in concepts if c in self.ent2id]
        if not c_ids: return []

        phones = [k for k in self.ent2id.keys() if "instance/" in k and ("phone/" in k or "config/" in k)]
        if not phones: return []
        
        p_ids = torch.tensor([self.ent2id[p] for p in phones], dtype=torch.long)
        r_idx = self.rel2id.get("suitableFor")
        
        if r_idx is None: return []

        total_scores = torch.zeros(len(phones))
        for cid in c_ids:
            h = p_ids
            r = torch.full((len(phones),), r_idx, dtype=torch.long)
            t = torch.full((len(phones),), cid, dtype=torch.long)
            
            with torch.no_grad():
                scores = self.model.score_hrt(torch.stack([h, r, t], dim=1)).squeeze()
            total_scores += scores

        best_indices = total_scores.argsort(descending=True)[:top_k]
        results = []
        for idx in best_indices:
            uri = phones[idx]
            name = self.names.get(uri, uri)
            results.append({
                "name": name, 
                "score": total_scores[idx].item(),
                "uri": uri
            })
        return results

# --- 3. Streamlit Caching & Helpers ---

@st.cache_resource
def get_engine():
    """Initializes the model once and caches it in memory."""
    return KGRecommender()

@st.cache_resource
def get_agent():
    return DefaultAgent()

def extract_concepts(agent, query, valid_vocab):
    vocab_list = "\n".join(valid_vocab)
    sys_prompt = f"Map user intent to JSON list of these concepts:\n{vocab_list}\nReturn JSON only."
    
    response = agent.action(sys_prompt, query)
    try:
        clean = re.sub(r"```json|```", "", response).strip()
        concepts = json.loads(clean)
        return [c for c in concepts if c in valid_vocab]
    except:
        return []

def generate_pitch(agent, query, top_phone, concepts):
    sys = "You are a helpful assistant. Explain why this phone is recommended."
    user = f"User: {query}\nRec: {top_phone['name']}\nMatched Concepts: {concepts}"
    return agent.action(sys, user)

# --- 4. The UI ---

st.set_page_config(page_title="GraphRAG Recommender", page_icon="📱")

st.title("📱 Smartphone Knowledge Graph")
st.markdown("Combines **LLM Translation** with **Graph Link Prediction**.")

# Initialize
with st.spinner("Loading AI Models..."):
    engine = get_engine()
    agent = get_agent()

# Sidebar for Vocabulary
with st.sidebar:
    st.header("Graph Vocabulary")
    valid_concepts = engine.get_valid_concepts()
    st.caption("The graph understands these concepts:")
    st.dataframe([c.replace("vocab/", "") for c in valid_concepts], height=300, hide_index=True)

# Main Chat Interface
query = st.text_input("Describe your ideal phone:", placeholder="I need a cheap phone for gaming and photography...")

if st.button("Recommend", type="primary"):
    if not query:
        st.warning("Please enter a preference.")
    else:
        # Step 1: Translation
        with st.status("Thinking...", expanded=True) as status:
            st.write("🧠 Analyzing intent with LLM...")
            concepts = extract_concepts(agent, query, valid_concepts)
            
            if not concepts:
                status.update(label="Failed", state="error")
                st.error("Could not map your request to any known concepts in the graph.")
            else:
                st.write(f"🔗 Mapped to Graph Nodes: `{concepts}`")
                
                # Step 2: Prediction
                st.write("📊 Calculating Link Prediction scores...")
                results = engine.recommend(concepts)
                status.update(label="Complete!", state="complete")

                if results:
                    top = results[0]
                    
                    # Step 3: Explanation
                    with st.spinner("Writing explanation..."):
                        pitch = generate_pitch(agent, query, top, concepts)

                    # Display Top Result
                    st.divider()
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader(f"🏆 Best Match: {top['name']}")
                        st.info(pitch)
                    
                    with col2:
                        st.metric("Graph Score", f"{top['score']:.2f}")
                        st.caption(f"ID: {top['uri']}")

                    # Display Alternatives
                    st.divider()
                    st.subheader("Other Candidates")
                    for res in results[1:]:
                        st.text(f"• {res['name']} (Score: {res['score']:.2f})")
                
                else:
                    st.error("No suitable phones found in the graph.")