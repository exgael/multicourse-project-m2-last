import json
import torch
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

# --- Mock Agent (Replace with your Import) ---
try:
    from agents import DefaultAgent
except ImportError:
    class DefaultAgent:
        def action(self, sys, user):
            # Fallback mock response for testing
            return '["vocab/Gaming", "vocab/price_budget"]'

# --- The Engine ---
class KGRecommender:
    def __init__(self):
        print("Loading engine...")
        
        # 1. Load Maps
        with open(OUTPUT_DIR / "entity_to_id.json") as f:
            self.ent2id = json.load(f)
        with open(OUTPUT_DIR / "relation_to_id.json") as f:
            self.rel2id = json.load(f)
            
        # 2. Load Model
        # --- FIX IS HERE: Added weights_only=False ---
        self.model = torch.load(OUTPUT_DIR / "model.pkl", map_location="cpu", weights_only=False)
        self.model.eval()

        # 3. Load Metadata (Names) directly from TTL
        self.names = self._load_names()

    def _load_names(self):
        print("Parsing names from TTL...")
        if not TTL_FILE.exists():
            print("Warning: TTL file not found. Names will be raw IDs.")
            return {}

        g = Graph()
        g.parse(TTL_FILE, format="turtle")
        names = {}
        # Get names for BasePhones and Configs
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
            # Store full URI, suffix, and relative path to ensure matching
            names[uri_str] = label
            names[uri_str.split("/")[-1]] = label 
            names["/".join(uri_str.split("/")[-2:])] = label
        return names

    def get_valid_concepts(self):
        return [k for k in self.ent2id.keys() if "vocab/" in k or "price_" in k]

    def recommend(self, concepts, top_k=3):
        # Map concepts to IDs
        c_ids = [self.ent2id[c] for c in concepts if c in self.ent2id]
        if not c_ids: return []

        # Find phones
        phones = [k for k in self.ent2id.keys() if "instance/" in k and ("phone/" in k or "config/" in k)]
        if not phones: return []
        
        p_ids = torch.tensor([self.ent2id[p] for p in phones], dtype=torch.long)
        r_idx = self.rel2id.get("suitableFor")
        
        if r_idx is None: 
            print("Error: 'suitableFor' relation missing")
            return []

        # Scoring
        total_scores = torch.zeros(len(phones))
        for cid in c_ids:
            # Batch: (All Phones, suitableFor, One Concept)
            h = p_ids
            r = torch.full((len(phones),), r_idx, dtype=torch.long)
            t = torch.full((len(phones),), cid, dtype=torch.long)
            
            with torch.no_grad():
                scores = self.model.score_hrt(torch.stack([h, r, t], dim=1)).squeeze()
            total_scores += scores

        # Rank
        best_indices = total_scores.argsort(descending=True)[:top_k]
        results = []
        for idx in best_indices:
            uri = phones[idx]
            name = self.names.get(uri, uri) # Fallback to URI if name not found
            results.append({
                "name": name, 
                "score": total_scores[idx].item(),
                "concepts": concepts
            })
        return results

# --- App Logic ---
def run_app(query: str):
    agent = DefaultAgent()
    rec = KGRecommender()
    
    # 1. Map NL -> Concepts
    valid_vocab = rec.get_valid_concepts()
    vocab_list = "\n".join(valid_vocab)
    
    sys_prompt = f"Map user intent to JSON list of these concepts:\n{vocab_list}\nReturn JSON only."
    mapped_str = agent.action(sys_prompt, query)
    
    try:
        # Clean markdown
        clean_json = re.sub(r"```json|```", "", mapped_str).strip()
        concepts = json.loads(clean_json)
        concepts = [c for c in concepts if c in valid_vocab]
    except:
        print("LLM failed to output JSON")
        concepts = []

    print(f"\nUser Query: {query}")
    print(f"Mapped to: {concepts}")

    if concepts:
        results = rec.recommend(concepts)
        if results:
            top = results[0]
            
            # 2. Explain
            sys_prompt_2 = "Explain why this phone is recommended based on the data provided."
            user_prompt_2 = f"User wants: {query}\nRec: {top['name']} (Score {top['score']:.2f})\nMatched: {concepts}"
            
            explanation = agent.action(sys_prompt_2, user_prompt_2)
            print("-" * 40)
            print(f"Recommendation: {top['name']}")
            print(explanation)
            print("-" * 40)
        else:
            print("No results found.")
    else:
        print("No matching concepts found.")

if __name__ == "__main__":
    # Test Run
    run_app("I want a cheap phone good for gaming")