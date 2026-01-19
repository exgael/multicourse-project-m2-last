"""
GraphRAG Embeddings Module - Using Phone Embeddings

The model contains:
- Real phones (Samsung Galaxy S24, iPhone 16, etc.) with embeddings
- Users with their preferences
- 6 use-cases (Gaming, Photography, Vlogging, Business, EverydayUse, MinimalistUse)
- 3 price segments (Flagship, MidRange, Budget)
- Relations (likes, suitableFor, hasBrand, hasPriceSegment, etc.)

How it works:
1. Parse user question to extract intent (use-case, features, brand)
2. Find use-case embedding (e.g., Gaming, Photography)
3. Use embeddings to find phones that are "suitableFor" that use-case
4. Rank by cosine similarity in embedding space
5. Generate natural language response with Ollama

KEY ADVANTAGE OVER SQL:
- Embeddings capture latent relationships learned from user behavior
- "Phones similar to what gamers like" - uses collaborative filtering via embeddings
- "Premium and futuristic" - semantic meaning, not exact filters
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
# Use the NEW model with real phone embeddings
MODEL_DIR = OUTPUT_DIR / "models" / "phone_embeddings"
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
    GraphRAG using Real Phone Knowledge Graph Embeddings (RotatE).
    
    This model has embeddings for 4787 REAL phones (Samsung, Apple, etc.)
    learned from:
    - User preferences (user -> likes -> phone)
    - Use-case suitability (phone -> suitableFor -> usecase)
    - Phone specs (phone -> hasBattery/hasCamera/etc -> value)
    
    RotatE models relations as rotations in complex space, capturing:
    - User-phone affinity patterns (collaborative filtering)
    - Phone-usecase compatibility
    - Spec-based similarity
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
        self.phone_embeddings_map: dict[str, np.ndarray] = {}
        self.usecase_embeddings_map: dict[str, np.ndarray] = {}
        self.graph: Graph | None = None
        
    def load(self) -> None:
        """Load embeddings and knowledge graph data."""
        print("Loading KG embeddings model (real phones)...")
        
        # Load PyKEEN model
        self.model = torch.load(self.model_dir / "trained_model.pkl", weights_only=False)
        
        # Load entity/relation mappings from triples factory
        from pykeen.triples import TriplesFactory
        tf = TriplesFactory.from_path_binary(self.model_dir / "training_triples")
        
        self.entity_to_id = tf.entity_to_id
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.relation_to_id = tf.relation_to_id
        
        # Extract embeddings - handle complex embeddings (RotatE)
        raw_embeddings = self.model.entity_representations[0](indices=None).detach().cpu()
        if raw_embeddings.is_complex():
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
        print(f"  Relations: {list(self.relation_to_id.keys())}")
        
        # Build phone and usecase embedding maps
        self._build_embedding_maps()
        
        # Load phone data from JSON
        self._load_phones_from_json()
        
        # Load additional phone specs from KG
        self._load_phone_specs_from_kg()
        
        print(f"  Phones with embeddings: {len(self.phone_embeddings_map)}")
        print(f"  Phones in catalog: {len(self.phones)}")
        print(f"  Use-cases: {list(self.usecase_embeddings_map.keys())}")
    
    def _build_embedding_maps(self) -> None:
        """Build maps from entity names to embeddings.
        
        The model trained from train_phones.py uses entity names:
        - Phones: "samsung_samsung_galaxy_s24_ultra_512gb_12gb" (phone_id from config)
        - Use-cases: "usecase/ProGaming", "usecase/ProPhotography", etc. (with prefix)
        - Brands: "Apple", "Samsung", "Xiaomi", etc.
        - Users: "user/user_0000", etc. (with prefix)
        - Specs: "battery_high", "camera_high", "refresh_high", etc.
        """
        # Known brands from KG
        known_brands = {
            "Apple", "Samsung", "Xiaomi", "OnePlus", "Google", "Motorola",
            "Huawei", "Oppo", "Vivo", "Realme", "Honor", "Asus", "Nothing",
            "Sony", "Nokia", "ZTE", "Infinix", "Tecno", "Poco", "Redmi", "vivo"
        }
        
        # Spec value patterns
        spec_patterns = ["battery_", "camera_", "selfie_", "refresh_", "ram_", "storage_", "5G_", "NFC_"]
        
        for entity, idx in self.entity_to_id.items():
            # Skip non-string entities (e.g., NaN)
            if not isinstance(entity, str):
                continue
                
            emb = self.entity_embeddings[idx]
            
            # Check if it's a use-case (prefixed with "usecase/")
            if entity.startswith("usecase/"):
                usecase_name = entity[8:]  # Remove "usecase/" prefix
                self.usecase_embeddings_map[usecase_name] = emb
            # Check if it's a user (skip)
            elif entity.startswith("user/"):
                continue
            # Check if it's a brand (skip, we don't need brand embeddings for search)
            elif entity in known_brands:
                continue
            # Check if it's a spec value (skip)
            elif any(entity.startswith(p) for p in spec_patterns):
                continue
            # Skip boolean values
            elif entity in ("true", "false"):
                continue
            # Otherwise, assume it's a phone_id
            else:
                # Phone IDs look like: "samsung_samsung_galaxy_s24_ultra_512gb_12gb"
                self.phone_embeddings_map[entity] = emb
    
    def _load_phones_from_json(self) -> None:
        """Load phone info from JSON."""
        phones_file = DATA_DIR / "preprocessed" / "phones_configuration.json"
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
        """Load additional phone specs from KG."""
        print("Loading phone specs from KG...")
        
        self.graph = Graph()
        self.graph.parse(self.kg_path, format="turtle")
        
        query = """
        PREFIX sp: <http://example.org/smartphone#>
        
        SELECT ?phone_id ?battery ?camera ?refresh ?has5g ?display ?year WHERE {
            ?phone a sp:Smartphone .
            BIND(STRAFTER(STR(?phone), "instance/phone/") AS ?phone_id)
            
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
    
    def get_phone_embedding(self, phone_id: str) -> np.ndarray | None:
        """Get embedding for a phone."""
        return self.phone_embeddings_map.get(phone_id)
    
    def get_usecase_embedding(self, usecase: str) -> np.ndarray | None:
        """Get embedding for a use-case."""
        return self.usecase_embeddings_map.get(usecase)
    
    def extract_intent(self, question: str) -> dict:
        """Extract intent and key features from a natural language question."""
        prompt = f"""Analyze this smartphone question and extract the intent.

Question: {question}

Return a JSON object with:
- "intent": one of ["find_phones", "compare", "recommend", "info"]
- "features": list of desired features like ["gaming", "photography", "5g", "big_battery", "high_camera", "amoled", "vlogging", "nfc"]
- "brand": brand name if mentioned (Samsung, Apple, Xiaomi, Google, OnePlus, etc.), else null
- "budget": "flagship", "midrange", "budget", or null
- "use_case": main use case if clear, one of: "gaming", "photography", "vlogging", "business", "everyday", "minimalist", else null

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
        
        try:
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
    
    def find_phones_by_embeddings(self, intent: dict, top_k: int = 20) -> list[tuple[Phone, float]]:
        """
        Find phones using REAL embedding similarity.
        
        This is the core of the GraphRAG approach:
        1. Map intent to use-case embedding
        2. Find phones similar to that use-case in embedding space
        3. Phones that are "suitableFor" the use-case will be closer
        """
        
        # Map intent to use-case (use actual use-case names from SKOS vocabulary)
        # Available in spv: Gaming, Photography, Vlogging, Business, EverydayUse, MinimalistUse
        # Price segments: Flagship, MidRange, Budget
        usecase_mapping = {
            # Gaming
            "gaming": ["Gaming"],
            "pro gaming": ["Gaming"],
            "mobile gaming": ["Gaming"],
            "gamer": ["Gaming"],
            # Photography
            "photography": ["Photography"],
            "photo": ["Photography"],
            "camera": ["Photography"],
            # Vlogging
            "vlogging": ["Vlogging"],
            "video": ["Vlogging"],
            "youtube": ["Vlogging"],
            "content creator": ["Vlogging"],
            "selfie": ["Vlogging"],
            # Business
            "business": ["Business"],
            "work": ["Business"],
            "professional": ["Business"],
            "work phone": ["Business"],
            # Everyday Use
            "everyday": ["EverydayUse"],
            "daily": ["EverydayUse"],
            "daily driver": ["EverydayUse"],
            "general": ["EverydayUse"],
            "general use": ["EverydayUse"],
            # Minimalist / Budget use
            "casual": ["EverydayUse", "MinimalistUse"],
            "simple": ["MinimalistUse"],
            "minimalist": ["MinimalistUse"],
            "basic": ["MinimalistUse"],
            "entry level": ["MinimalistUse"],
            "affordable": ["MinimalistUse"],
        }
        
        query_embeddings = []
        
        # Get use-case embedding
        use_case = intent.get("use_case", "").lower() if intent.get("use_case") else ""
        if use_case:
            uc_names = usecase_mapping.get(use_case, [])
            for uc_name in uc_names:
                uc_emb = self.get_usecase_embedding(uc_name)
                if uc_emb is not None:
                    query_embeddings.append(uc_emb * 2.0)  # Weight use-case strongly
        
        # Also check features for use-cases
        for feature in intent.get("features", []):
            feature_lower = feature.lower()
            uc_names = usecase_mapping.get(feature_lower, [])
            for uc_name in uc_names:
                uc_emb = self.get_usecase_embedding(uc_name)
                if uc_emb is not None:
                    query_embeddings.append(uc_emb)
        
        # If no use-case found, use all use-cases weighted
        if not query_embeddings:
            for uc_name, uc_emb in self.usecase_embeddings_map.items():
                query_embeddings.append(uc_emb * 0.5)
        
        # Average query embeddings
        query_embedding = np.mean(query_embeddings, axis=0)
        
        # Find similar phones using embedding distance
        phone_scores: list[tuple[str, float]] = []
        
        for phone_id, phone_emb in self.phone_embeddings_map.items():
            # Cosine similarity from embeddings
            emb_sim = np.dot(phone_emb, query_embedding) / (
                np.linalg.norm(phone_emb) * np.linalg.norm(query_embedding) + 1e-10
            )
            
            # Apply spec-based bonus for specific use-cases
            spec_bonus = self._compute_spec_bonus(phone_id, intent)
            
            # Hybrid score: embedding similarity + spec bonus
            final_score = emb_sim + spec_bonus
            
            phone_scores.append((phone_id, float(final_score)))
        
        # Sort by combined score
        phone_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply filters
        brand = intent.get("brand")
        budget = intent.get("budget")
        
        filtered_phones = []
        for phone_id, score in phone_scores:
            if phone_id not in self.phones:
                continue
            
            phone = self.phones[phone_id]
            
            # Brand filter
            if brand and brand.lower() not in ["none", "null", ""]:
                if phone.brand.lower() != brand.lower():
                    continue
            
            # Budget filter
            if budget and budget.lower() not in ["none", "null", ""]:
                if not self._matches_budget_tier(phone, budget.lower()):
                    continue
            
            filtered_phones.append((phone, score))
            
            if len(filtered_phones) >= top_k:
                break
        
        return filtered_phones
    
    def _compute_spec_bonus(self, phone_id: str, intent: dict) -> float:
        """
        Compute a spec-based bonus to improve ranking for specific use-cases.
        
        This hybrid approach uses phone specs to boost the embedding-based score,
        ensuring phones with relevant specs rank higher for specific use-cases.
        """
        if phone_id not in self.phones:
            return 0.0
        
        phone = self.phones[phone_id]
        name_lower = phone.name.lower() if phone.name else ""
        bonus = 0.0
        
        use_case = intent.get("use_case", "").lower() if intent.get("use_case") else ""
        features = [f.lower() for f in intent.get("features", [])]
        
        # Gaming bonus: high refresh rate, gaming phones
        if use_case in ["gaming", "pro gaming"] or "gaming" in features:
            # ROG, gaming phones
            if "rog" in name_lower or "gaming" in name_lower or "redmagic" in name_lower:
                bonus += 0.5
            # High refresh rate (120Hz+)
            if phone.refresh_rate and phone.refresh_rate >= 120:
                bonus += 0.2
            # High performance (flagship-level)
            if "ultra" in name_lower or "pro" in name_lower:
                bonus += 0.1
            # Big RAM (inferred from phone name if available)
            if "16gb" in phone_id or "24gb" in phone_id or "12gb" in phone_id:
                bonus += 0.15
        
        # Photography bonus: high camera MP
        if use_case in ["photography", "photo"] or "photography" in features or "camera" in features:
            if phone.camera and phone.camera >= 200:
                bonus += 0.4
            elif phone.camera and phone.camera >= 108:
                bonus += 0.3
            elif phone.camera and phone.camera >= 64:
                bonus += 0.15
            # Camera-focused phones
            if "camera" in name_lower or "photo" in name_lower:
                bonus += 0.2
        
        # Vlogging bonus: good selfie + video features
        if use_case in ["vlogging", "video"] or "vlogging" in features or "video" in features:
            if phone.camera and phone.camera >= 50:
                bonus += 0.15
            # Flip phones good for vlogging
            if "flip" in name_lower:
                bonus += 0.3
            # Good front camera devices
            if "v60" in name_lower or "v50" in name_lower or "vivo v" in name_lower:
                bonus += 0.15
        
        # Big battery bonus
        if "battery" in features or "big_battery" in features:
            if phone.battery and phone.battery >= 6000:
                bonus += 0.3
            elif phone.battery and phone.battery >= 5000:
                bonus += 0.15
        
        # 5G bonus
        if "5g" in features:
            if phone.supports_5g:
                bonus += 0.2
        
        return bonus

    def _matches_budget_tier(self, phone: Phone, budget: str) -> bool:
        """Check if phone matches budget tier based on name and specs."""
        name_lower = phone.name.lower()
        
        flagship_patterns = [
            "ultra", "pro max", "pro+", " pro", "fold", "flip",
            "galaxy s2", "galaxy s3",
            "iphone 16", "iphone 15", "iphone 14", "iphone 13",
            "pixel 9", "pixel 8", "pixel 7",
            "find x", "find n", "mate ", "magic",
            "rog phone", "z fold", "z flip",
        ]
        
        budget_patterns = [
            "a0", "a1", "a2", "a3",
            "galaxy m", "galaxy f",
            "moto g", "moto e",
            "redmi", "poco m", "poco c",
            "realme c", "realme narzo",
            "nord n", "nord ce",
            "spark", "pop", "hot",
        ]
        
        if budget == "flagship":
            has_flagship = any(p in name_lower for p in flagship_patterns)
            has_budget = any(p in name_lower for p in budget_patterns)
            has_high_specs = (
                (phone.camera and phone.camera >= 100) or
                (phone.refresh_rate and phone.refresh_rate >= 120)
            )
            return has_flagship or (has_high_specs and not has_budget)
        
        elif budget == "budget":
            has_budget = any(p in name_lower for p in budget_patterns)
            has_flagship = any(p in name_lower for p in flagship_patterns)
            return has_budget or (not has_flagship and phone.camera and phone.camera < 50)
        
        elif budget == "midrange":
            has_flagship = any(p in name_lower for p in flagship_patterns)
            has_budget = any(p in name_lower for p in budget_patterns)
            return not has_flagship and not has_budget
        
        return True
    
    def generate_response(self, question: str, phones: list[tuple[Phone, float]], intent: dict) -> str:
        """Generate a natural language response using Ollama."""
        
        if not phones:
            return "I couldn't find phones matching your criteria. Try being more specific about features like gaming, photography, or battery life."
        
        # Format phone info with similarity scores
        phones_info = []
        for i, (phone, score) in enumerate(phones[:5], 1):
            info = f"{i}. {phone.name} ({phone.brand}) [similarity: {score:.2f}]"
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
                specs.append(phone.display_type[:30])
            if specs:
                info += f" - {', '.join(specs)}"
            phones_info.append(info)
        
        phones_text = "\n".join(phones_info)
        
        prompt = f"""Based on the user's question and the phones found via embedding similarity, provide a helpful response.

IMPORTANT: Only recommend phones from the list below. These were found using AI-learned patterns from user preferences.

User Question: {question}

Detected Intent: {intent.get('intent', 'find_phones')}
Use-case: {intent.get('use_case') or 'general'}
Features wanted: {', '.join(intent.get('features', [])) or 'not specified'}

PHONES FOUND (ranked by embedding similarity to user intent):
{phones_text}

Provide a concise response that:
1. Recommends 2-3 best matches from the list
2. Explains why they match based on specs shown
3. Notes the similarity score indicates how well they match the use-case

Keep it under 150 words. Use ONLY phones from the list above."""

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
        """Answer a natural language question using KG embeddings."""
        if self.model is None:
            self.load()
        
        print(f"Analyzing question...")
        intent = self.extract_intent(question)
        print(f"  Intent: {intent}")
        
        print("Finding phones using embedding similarity...")
        phones_with_scores = self.find_phones_by_embeddings(intent)
        print(f"  Found {len(phones_with_scores)} matching phones")
        
        print("Generating response...")
        phones_only = [p for p, _ in phones_with_scores]
        answer = self.generate_response(question, phones_with_scores, intent)
        
        return RAGResult(
            question=question,
            intent=str(intent),
            relevant_phones=phones_only[:10],
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
        output.append(f"\nTop Matching Phones (by embedding similarity):")
        
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
    """Run demo with semantic questions that showcase embedding-based search."""
    questions = [
        # Use-case based - embeddings find phones "suitableFor" these use-cases
        "I need a phone for professional mobile gaming",
        
        # Brand + use-case - embedding similarity within brand
        "Best Samsung phones for vlogging",
        
        # Vague/semantic - embeddings understand user intent
        "A phone that photographers would love",
        
        # Multi-criteria - embedding space captures combinations
        "Good for both gaming and taking photos",
        
        # Persona-based - embeddings learned from user preferences
        "What would a content creator choose?",
        
        # Budget + use-case - filtered embedding search
        "Best flagship for business use",
        
        # Casual description - semantic understanding
        "Something reliable for everyday use with good battery",
    ]
    
    try:
        rag = KGEmbeddingsRAG()
        rag.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you've trained the phone model first:")
        print("  python -m src.recommandation.train_phones")
        return
    
    for question in questions:
        print("\n" + "=" * 60)
        result = rag.query(question)
        print(rag.format_result(result))


def interactive_mode():
    """Run interactive Q&A session."""
    print("=" * 60)
    print("GraphRAG: Real Phone Embeddings Q&A (Interactive)")
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
    
    parser = argparse.ArgumentParser(description="GraphRAG: Real Phone Embeddings Q&A")
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