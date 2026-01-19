"""
GraphRAG Embeddings Module

This module implements a RAG (Retrieval-Augmented Generation) approach using
Knowledge Graph embeddings learned by PyKEEN (TransE model).

The approach:
1. Use KG embeddings to represent entities (phones, users, use-cases)
2. For a natural language question, extract key entities and intents
3. Use embedding similarity to find relevant phones
4. Generate a natural language response with Ollama

This complements the NL→SPARQL approach by using vector similarity
instead of structured queries.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
import requests
from rdflib import Graph

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models" / "link_prediction"
FINAL_KG_TTL = DATA_DIR / "rdf" / "knowledge_graph_full.ttl"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"


@dataclass
class Phone:
    """Phone entity with its properties."""
    phone_id: str
    name: str
    brand: str
    battery: int | None = None
    camera: int | None = None
    refresh_rate: int | None = None
    supports_5g: bool = False
    display_type: str | None = None
    release_year: int | None = None


@dataclass
class RAGResult:
    """Result of a GraphRAG query."""
    question: str
    intent: str
    relevant_phones: list[Phone]
    answer: str
    method: str = "embeddings"


class KGEmbeddingsRAG:
    """
    GraphRAG using Knowledge Graph Embeddings.
    
    Uses TransE embeddings from PyKEEN to find similar entities
    and generates natural language responses with Ollama.
    """
    
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        kg_path: Path = FINAL_KG_TTL,
        ollama_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL
    ):
        self.model_dir = model_dir
        self.kg_path = kg_path
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        self.model = None
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_to_id: dict[str, int] = {}
        self.id_to_entity: dict[int, str] = {}
        self.relation_to_id: dict[str, int] = {}
        
        self.phones: dict[str, Phone] = {}
        self.use_cases: list[str] = []
        self.graph: Graph | None = None
        
    def load(self) -> None:
        """Load embeddings and knowledge graph data."""
        print("Loading KG embeddings model...")
        
        # Load PyKEEN model
        self.model = torch.load(self.model_dir / "trained_model.pkl", weights_only=False)
        
        # Load entity/relation mappings from triples factory
        from pykeen.triples import TriplesFactory
        tf = TriplesFactory.from_path_binary(self.model_dir / "training_triples")
        
        self.entity_to_id = tf.entity_to_id
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.relation_to_id = tf.relation_to_id
        
        # Extract embeddings - handle complex embeddings (RotatE) by taking real part
        raw_embeddings = self.model.entity_representations[0](indices=None).detach().cpu()
        if raw_embeddings.is_complex():
            # RotatE uses complex embeddings - convert to real by concatenating real and imaginary parts
            self.entity_embeddings = torch.cat([raw_embeddings.real, raw_embeddings.imag], dim=-1).numpy()
        else:
            self.entity_embeddings = raw_embeddings.numpy()
        
        raw_rel_embeddings = self.model.relation_representations[0](indices=None).detach().cpu()
        if raw_rel_embeddings.is_complex():
            self.relation_embeddings = torch.cat([raw_rel_embeddings.real, raw_rel_embeddings.imag], dim=-1).numpy()
        else:
            self.relation_embeddings = raw_rel_embeddings.numpy()
        
        print(f"  Loaded {len(self.entity_to_id)} entity embeddings (dim={self.entity_embeddings.shape[1]})")
        print(f"  Loaded {len(self.relation_to_id)} relation embeddings")
        
        # Load phone data from JSON
        self._load_phones_from_json()
        
        # Load phone specs from KG
        self._load_phone_specs_from_kg()
        
        # Extract use-cases from entities (filter out non-string entities like nan)
        self.use_cases = [e for e in self.entity_to_id.keys() 
                         if isinstance(e, str) and not e.startswith("instance/") 
                         and "_" not in e[:10] and e not in ["true", "false"]]
        
        print(f"  Found {len(self.phones)} phones")
        print(f"  Found {len(self.use_cases)} use-cases/features")
        
    def _load_phones_from_json(self) -> None:
        """Load phone info from JSON including specs."""
        phones_file = DATA_DIR / "raw_pretty" / "phones.json"
        with open(phones_file, "r", encoding="utf-8") as f:
            phones_data = json.load(f)
        
        for p in phones_data:
            phone_id = p["phone_id"]
            self.phones[phone_id] = Phone(
                phone_id=phone_id,
                name=p.get("phone_name", phone_id),
                brand=p.get("brand", "Unknown"),
                battery=p.get("battery_mah"),
                camera=p.get("main_camera_mp"),
                refresh_rate=p.get("refresh_rate_hz"),
                supports_5g=p.get("supports_5g", False),
                display_type=p.get("display_type"),
                release_year=p.get("year")
            )
    
    def _load_phone_specs_from_kg(self) -> None:
        """Load phone specifications from the knowledge graph."""
        print("Loading phone specs from KG...")
        
        self.graph = Graph()
        self.graph.parse(self.kg_path, format="turtle")
        
        query = """
        PREFIX sp: <http://example.org/smartphone#>
        
        SELECT ?phone_id ?battery ?camera ?refresh ?has5g ?display ?year WHERE {
            ?phone a sp:Smartphone .
            BIND(STRAFTER(STR(?phone), "phone/") AS ?phone_id)
            
            OPTIONAL { ?phone sp:batteryCapacityMah ?battery }
            OPTIONAL { ?phone sp:mainCameraMP ?camera }
            OPTIONAL { ?phone sp:refreshRateHz ?refresh }
            OPTIONAL { ?phone sp:supports5G ?has5g }
            OPTIONAL { ?phone sp:displayType ?display }
            OPTIONAL { ?phone sp:releaseYear ?year }
        }
        """
        
        for row in self.graph.query(query):
            phone_id = str(row.phone_id)
            if phone_id in self.phones:
                phone = self.phones[phone_id]
                if row.battery:
                    phone.battery = int(row.battery)
                if row.camera:
                    phone.camera = int(row.camera)
                if row.refresh:
                    phone.refresh_rate = int(row.refresh)
                if row.has5g:
                    phone.supports_5g = str(row.has5g).lower() == "true"
                if row.display:
                    phone.display_type = str(row.display)
                if row.year:
                    phone.release_year = int(row.year)
    
    def get_entity_embedding(self, entity: str) -> np.ndarray | None:
        """Get embedding for an entity."""
        if entity in self.entity_to_id:
            idx = self.entity_to_id[entity]
            return self.entity_embeddings[idx]
        return None
    
    def find_similar_entities(
        self, 
        query_embedding: np.ndarray, 
        entity_filter: str | None = None,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find entities most similar to a query embedding."""
        # Compute cosine similarity
        norms = np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        normalized = self.entity_embeddings / (norms + 1e-10)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        similarities = normalized @ query_norm
        
        # Filter by entity type if specified
        if entity_filter:
            mask = np.array([entity_filter in self.id_to_entity[i] 
                           for i in range(len(similarities))])
            similarities = np.where(mask, similarities, -np.inf)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                entity = self.id_to_entity[idx]
                results.append((entity, float(similarities[idx])))
        
        return results
    
    def extract_intent(self, question: str) -> dict:
        """
        Extract intent and key features from a natural language question.
        Uses Ollama to parse the question.
        """
        prompt = f"""Analyze this smartphone question and extract the intent.

Question: {question}

Return a JSON object with:
- "intent": one of ["find_phones", "compare", "recommend", "info"]
- "features": list of desired features like ["gaming", "photography", "5g", "big_battery", "high_camera", "amoled"]
- "brand": brand name if mentioned, else null
- "budget": "flagship", "midrange", "budget", or null
- "use_case": main use case if clear (gaming, photography, business, etc.), else null

Return ONLY the JSON, no explanation."""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()["response"].strip()
        
        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            return json.loads(result)
        except json.JSONDecodeError:
            return {
                "intent": "find_phones",
                "features": [],
                "brand": None,
                "budget": None,
                "use_case": None
            }
    
    def find_phones_by_features(self, intent: dict, top_k: int = 10) -> list[Phone]:
        """
        Find phones matching the extracted intent using embeddings.
        """
        # Map features to discretized entity names (from train.py)
        feature_mapping = {
            "gaming": ["refresh_144hzplus", "refresh_120to144hz", "battery_6000to8000", "battery_8000plus"],
            "photography": ["camera_100to200", "camera_200plus", "camera_48to100"],
            "big_battery": ["battery_6000to8000", "battery_8000plus"],
            "high_camera": ["camera_100to200", "camera_200plus"],
            "5g": ["true"],  # supports5G = true
            "amoled": [],  # Not in embeddings, filter later
            "selfie": ["selfie_48plus", "selfie_32to48"],
        }
        
        # Collect feature embeddings
        feature_embeddings = []
        
        for feature in intent.get("features", []):
            mapped = feature_mapping.get(feature.lower(), [])
            for entity_name in mapped:
                emb = self.get_entity_embedding(entity_name)
                if emb is not None:
                    feature_embeddings.append(emb)
        
        # Use-case based search
        use_case = intent.get("use_case")
        if use_case:
            uc_mapping = {
                "gaming": "ProGaming",
                "photography": "ProPhotography", 
                "business": "Business",
                "everyday": "EverydayUse",
                "vlogging": "Vlogging"
            }
            uc_entity = uc_mapping.get(use_case.lower())
            if uc_entity:
                emb = self.get_entity_embedding(uc_entity)
                if emb is not None:
                    feature_embeddings.append(emb * 2)  # Weight use-case more
        
        if not feature_embeddings:
            # No specific features, return popular phones
            return list(self.phones.values())[:top_k]
        
        # Average feature embeddings
        query_embedding = np.mean(feature_embeddings, axis=0)
        
        # Find similar phones using TransE
        # In TransE: h + r ≈ t, so we look for phones that "have" these features
        phone_scores: dict[str, float] = {}
        
        for phone_id, phone in self.phones.items():
            # Entity format in training: instance/phone/{phone_id}
            entity_key = f"instance/phone/{phone_id}"
            emb = self.get_entity_embedding(entity_key)
            if emb is not None:
                # Cosine similarity from embeddings
                sim = np.dot(emb, query_embedding) / (
                    np.linalg.norm(emb) * np.linalg.norm(query_embedding) + 1e-10
                )
                
                # Boost score based on actual specs matching the intent
                spec_boost = 0.0
                features = [f.lower() for f in intent.get("features", [])]
                use_case = (intent.get("use_case") or "").lower()
                
                # Gaming boost: high refresh rate + big battery
                if "gaming" in features or use_case == "gaming":
                    if phone.refresh_rate and phone.refresh_rate >= 120:
                        spec_boost += 0.3
                    if phone.battery and phone.battery >= 5000:
                        spec_boost += 0.2
                
                # Photography boost: high camera MP
                if "photography" in features or use_case == "photography":
                    if phone.camera and phone.camera >= 100:
                        spec_boost += 0.4
                    elif phone.camera and phone.camera >= 50:
                        spec_boost += 0.2
                
                # Big battery boost
                if "big_battery" in features:
                    if phone.battery and phone.battery >= 6000:
                        spec_boost += 0.3
                    elif phone.battery and phone.battery >= 5000:
                        spec_boost += 0.15
                
                # 5G boost
                if "5g" in features:
                    if phone.supports_5g:
                        spec_boost += 0.2
                
                # Modern phone boost (recent release year)
                if phone.release_year and phone.release_year >= 2023:
                    spec_boost += 0.15
                elif phone.release_year and phone.release_year >= 2021:
                    spec_boost += 0.05
                
                phone_scores[phone_id] = sim + spec_boost
        
        # Sort by score
        sorted_phones = sorted(phone_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by brand if specified
        brand = intent.get("brand")
        if brand:
            brand_lower = brand.lower()
            sorted_phones = [(pid, s) for pid, s in sorted_phones 
                           if self.phones[pid].brand.lower() == brand_lower]
        
        # Filter by display type for AMOLED
        if "amoled" in [f.lower() for f in intent.get("features", [])]:
            sorted_phones = [(pid, s) for pid, s in sorted_phones
                           if self.phones[pid].display_type and 
                              "amoled" in self.phones[pid].display_type.lower()]
        
        # If no phones found via embeddings, fallback to spec-based search
        if not sorted_phones:
            sorted_phones = self._fallback_spec_search(intent)
        
        # Return top phones
        result = []
        for phone_id, score in sorted_phones[:top_k]:
            result.append(self.phones[phone_id])
        
        return result
    
    def _fallback_spec_search(self, intent: dict) -> list[tuple[str, float]]:
        """Fallback search using phone specs when embeddings don't find matches."""
        phone_scores: dict[str, float] = {}
        features = [f.lower() for f in intent.get("features", [])]
        use_case = (intent.get("use_case") or "").lower()
        brand = intent.get("brand")
        
        for phone_id, phone in self.phones.items():
            # Filter by brand first if specified
            if brand and phone.brand.lower() != brand.lower():
                continue
            
            score = 0.0
            
            # Gaming: high refresh rate + big battery
            if "gaming" in features or use_case in ["gaming", "mobile gaming"]:
                if phone.refresh_rate and phone.refresh_rate >= 144:
                    score += 0.5
                elif phone.refresh_rate and phone.refresh_rate >= 120:
                    score += 0.3
                if phone.battery and phone.battery >= 5000:
                    score += 0.3
            
            # Photography: high camera MP
            if "photography" in features or use_case == "photography":
                if phone.camera and phone.camera >= 200:
                    score += 0.6
                elif phone.camera and phone.camera >= 100:
                    score += 0.4
                elif phone.camera and phone.camera >= 50:
                    score += 0.2
            
            # Big battery
            if "big_battery" in features:
                if phone.battery and phone.battery >= 6000:
                    score += 0.4
                elif phone.battery and phone.battery >= 5000:
                    score += 0.2
            
            # 5G
            if "5g" in features:
                if phone.supports_5g:
                    score += 0.3
            
            # AMOLED display
            if "amoled" in features:
                if phone.display_type and "amoled" in phone.display_type.lower():
                    score += 0.3
            
            # Vlogging: good selfie cam + stabilization (assume high main cam)
            if "vlogging" in features or use_case == "vlogging":
                if phone.camera and phone.camera >= 50:
                    score += 0.3
                if phone.refresh_rate and phone.refresh_rate >= 60:
                    score += 0.1
            
            # Modern phone boost
            if phone.release_year and phone.release_year >= 2024:
                score += 0.25
            elif phone.release_year and phone.release_year >= 2023:
                score += 0.15
            elif phone.release_year and phone.release_year >= 2021:
                score += 0.05
            
            if score > 0:
                phone_scores[phone_id] = score
        
        return sorted(phone_scores.items(), key=lambda x: x[1], reverse=True)
    
    def generate_response(self, question: str, phones: list[Phone], intent: dict) -> str:
        """Generate a natural language response using Ollama."""
        
        # Format phone info
        phones_info = []
        for i, phone in enumerate(phones[:5], 1):
            info = f"{i}. {phone.name} ({phone.brand})"
            specs = []
            if phone.battery:
                specs.append(f"battery: {phone.battery}mAh")
            if phone.camera:
                specs.append(f"camera: {phone.camera}MP")
            if phone.refresh_rate:
                specs.append(f"refresh: {phone.refresh_rate}Hz")
            if phone.supports_5g:
                specs.append("5G")
            if phone.display_type:
                specs.append(phone.display_type)
            if specs:
                info += f" - {', '.join(specs)}"
            phones_info.append(info)
        
        phones_text = "\n".join(phones_info)
        
        prompt = f"""Based on the user's question and the relevant phones found, provide a helpful response.

User Question: {question}

Detected Intent: {intent.get('intent', 'find_phones')}
Features wanted: {', '.join(intent.get('features', [])) or 'not specified'}

Relevant Phones Found:
{phones_text}

Provide a concise, helpful response that:
1. Directly answers the question
2. Highlights 2-3 best matches with their key specs
3. Explains why they match the user's needs

Keep it under 150 words."""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=120
        )
        response.raise_for_status()
        
        return response.json()["response"].strip()
    
    def query(self, question: str) -> RAGResult:
        """
        Answer a natural language question using KG embeddings.
        """
        if self.model is None:
            self.load()
        
        # Extract intent
        print(f"Analyzing question...")
        intent = self.extract_intent(question)
        print(f"  Intent: {intent}")
        
        # Find relevant phones
        print("Finding relevant phones using embeddings...")
        phones = self.find_phones_by_features(intent)
        print(f"  Found {len(phones)} matching phones")
        
        # Generate response
        print("Generating response...")
        answer = self.generate_response(question, phones, intent)
        
        return RAGResult(
            question=question,
            intent=str(intent),
            relevant_phones=phones[:10],
            answer=answer
        )
    
    def format_result(self, result: RAGResult) -> str:
        """Format a RAG result for display."""
        output = []
        output.append(f"Question: {result.question}")
        output.append(f"\nIntent Analysis: {result.intent}")
        output.append(f"\n{'='*60}")
        output.append(f"\nAnswer:\n{result.answer}")
        output.append(f"\n{'='*60}")
        output.append(f"\nTop Matching Phones:")
        
        for i, phone in enumerate(result.relevant_phones[:5], 1):
            specs = []
            if phone.battery:
                specs.append(f"{phone.battery}mAh")
            if phone.camera:
                specs.append(f"{phone.camera}MP cam")
            if phone.refresh_rate:
                specs.append(f"{phone.refresh_rate}Hz")
            if phone.supports_5g:
                specs.append("5G")
            spec_str = " | ".join(specs) if specs else "N/A"
            output.append(f"  {i}. {phone.name} - {spec_str}")
        
        return "\n".join(output)


def demo():
    """Run demo with sample questions."""
    questions = [
        "I need a phone for mobile gaming with good battery life",
        "What's the best Samsung phone for photography?",
        "Recommend a phone with 5G and AMOLED display",
        "I want a phone with at least 120Hz refresh rate",
        "What phone should I get for vlogging?",
    ]
    
    print("=" * 60)
    print("GraphRAG Demo: Embeddings-based Q&A")
    print("=" * 60)
    
    try:
        rag = KGEmbeddingsRAG()
        rag.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you've trained the model first:")
        print("  python -m src.recommandation.train")
        return
    
    for question in questions:
        print("\n" + "=" * 60)
        result = rag.query(question)
        print(rag.format_result(result))


def interactive_mode():
    """Run interactive Q&A session."""
    print("=" * 60)
    print("GraphRAG: Embeddings-based Q&A (Interactive)")
    print("=" * 60)
    print("\nAsk questions about smartphones in natural language.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    try:
        rag = KGEmbeddingsRAG()
        rag.load()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            result = rag.query(question)
            print("\n" + rag.format_result(result))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG: Embeddings-based Q&A")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample questions")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    elif args.question:
        try:
            rag = KGEmbeddingsRAG()
            rag.load()
            result = rag.query(args.question)
            print(rag.format_result(result))
        except Exception as e:
            print(f"Error: {e}")
    else:
        interactive_mode()
