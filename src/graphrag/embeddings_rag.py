"""
GraphRAG Embeddings Module - Using Phone Embeddings

The model contains:
- Real phones (Samsung Galaxy S24, iPhone 16, etc.) with embeddings
- Users with their preferences
- 6 use-cases (Gaming, Photography, Vlogging, Business, EverydayUse, MinimalistUse)
- 3 price segments (Flagship, MidRange, Budget)
- Relations (likes, suitableFor, hasBrand, hasPriceSegment, etc.)

Extended data from OWL schema:
- Selfie camera specs (selfieCameraMP) - key for vlogging recommendations
- Processor info (processorName) - for gaming/performance queries
- NFC support (supportsNFC) - for business/payment use cases
- Storage/RAM configurations (storageGB, ramGB)
- Tag sentiment from reviews (TagSentiment) - user feedback on Camera, Battery, etc.
- Actual prices (PriceOffering) - real EUR prices for budget filtering
- Store availability (Store) - where to buy

How it works:
1. Parse user question to extract intent (use-case, features, brand, price range)
2. Find use-case embedding (e.g., Gaming, Photography)
3. Use embeddings to find phones that are "suitableFor" that use-case
4. Apply spec bonuses (processor, RAM, selfie camera, NFC) based on intent
5. Incorporate review sentiment scores for relevant tags
6. Filter by actual price range (if specified)
7. Rank by hybrid score: embedding similarity + specs + sentiment
8. Generate natural language response with Ollama

KEY ADVANTAGES:
- Embeddings capture latent relationships learned from user behavior
- Real prices enable accurate budget filtering ("under 500 EUR")
- Review sentiment surfaces user-validated quality signals
- NFC, selfie camera, processor data enable specialized queries
- Store data shows where to purchase
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
    selfie_camera: int | None = None
    refresh_rate: int | None = None
    supports_5g: bool = False
    supports_nfc: bool = False
    display_type: str | None = None
    processor: str | None = None
    release_year: int | None = None
    storage_gb: int | None = None
    ram_gb: int | None = None
    min_price: float | None = None
    stores: list[str] | None = None
    # Tag sentiment scores (positive - negative counts)
    sentiment: dict[str, tuple[int, int]] | None = None  # tag -> (positive, negative)


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
                selfie_camera=p.get("selfie_camera_mp"),
                refresh_rate=p.get("refresh_rate_hz"),
                supports_5g=p.get("supports_5g", False),
                supports_nfc=p.get("supports_nfc", False),
                display_type=p.get("display_type"),
                processor=p.get("processor"),
                release_year=p.get("year"),
                storage_gb=p.get("storage_gb"),
                ram_gb=p.get("ram_gb"),
            )
    
    def _load_phone_specs_from_kg(self) -> None:
        """Load additional phone specs from KG."""
        print("Loading phone specs from KG...")

        self.graph = Graph()
        self.graph.parse(self.kg_path, format="turtle")

        # Query base phone specs
        query = """
        PREFIX sp: <http://example.org/smartphone#>

        SELECT ?phone_id ?battery ?camera ?selfie ?refresh ?has5g ?hasNfc ?display ?processor ?year WHERE {
            ?phone a sp:Smartphone .
            BIND(STRAFTER(STR(?phone), "instance/phone/") AS ?phone_id)

            OPTIONAL { ?phone sp:batteryCapacityMah ?battery }
            OPTIONAL { ?phone sp:mainCameraMP ?camera }
            OPTIONAL { ?phone sp:selfieCameraMP ?selfie }
            OPTIONAL { ?phone sp:refreshRateHz ?refresh }
            OPTIONAL { ?phone sp:supports5G ?has5g }
            OPTIONAL { ?phone sp:supportsNFC ?hasNfc }
            OPTIONAL { ?phone sp:displayType ?display }
            OPTIONAL { ?phone sp:processorName ?processor }
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
                if row.selfie:
                    phone.selfie_camera = int(row.selfie)
                if row.refresh:
                    phone.refresh_rate = int(row.refresh)
                if row.has5g:
                    phone.supports_5g = str(row.has5g).lower() == "true"
                if row.hasNfc:
                    phone.supports_nfc = str(row.hasNfc).lower() == "true"
                if row.display:
                    phone.display_type = str(row.display)
                if row.processor:
                    phone.processor = str(row.processor)
                if row.year:
                    phone.release_year = int(row.year)

        # Load configuration data (storage, RAM)
        self._load_phone_configurations()
        # Load tag sentiment from reviews
        self._load_tag_sentiments()
        # Load price offerings
        self._load_price_offerings()

    def _load_phone_configurations(self) -> None:
        """Load phone configuration data (storage, RAM) from KG."""
        print("Loading phone configurations...")

        query = """
        PREFIX sp: <http://example.org/smartphone#>

        SELECT ?phone_id ?storage ?ram WHERE {
            ?config a sp:PhoneConfiguration ;
                    sp:hasBasePhone ?phone .
            BIND(STRAFTER(STR(?phone), "instance/phone/") AS ?phone_id)

            OPTIONAL { ?config sp:storageGB ?storage }
            OPTIONAL { ?config sp:ramGB ?ram }
        }
        """

        for row in self.graph.query(query):
            phone_id = str(row.phone_id)
            if phone_id in self.phones:
                phone = self.phones[phone_id]
                # Take the highest storage/RAM values (flagship config)
                if row.storage:
                    storage = int(row.storage)
                    if phone.storage_gb is None or storage > phone.storage_gb:
                        phone.storage_gb = storage
                if row.ram:
                    ram = int(row.ram)
                    if phone.ram_gb is None or ram > phone.ram_gb:
                        phone.ram_gb = ram

    def _load_tag_sentiments(self) -> None:
        """Load tag sentiment data from KG reviews."""
        print("Loading tag sentiments...")

        query = """
        PREFIX sp: <http://example.org/smartphone#>

        SELECT ?phone_id ?tag ?positive ?negative WHERE {
            ?phone a sp:Smartphone ;
                   sp:hasSentiment ?sentiment .
            BIND(STRAFTER(STR(?phone), "instance/phone/") AS ?phone_id)

            ?sentiment sp:forTag ?tag ;
                       sp:positiveCount ?positive ;
                       sp:negativeCount ?negative .
        }
        """

        sentiment_count = 0
        for row in self.graph.query(query):
            phone_id = str(row.phone_id)
            if phone_id in self.phones:
                phone = self.phones[phone_id]
                if phone.sentiment is None:
                    phone.sentiment = {}
                tag = str(row.tag)
                positive = int(row.positive)
                negative = int(row.negative)
                phone.sentiment[tag] = (positive, negative)
                sentiment_count += 1

        print(f"  Loaded {sentiment_count} tag sentiment records")

    def _load_price_offerings(self) -> None:
        """Load price offerings and store data from KG."""
        print("Loading price offerings...")

        query = """
        PREFIX sp: <http://example.org/smartphone#>

        SELECT ?phone_id ?price ?storeName WHERE {
            ?offering a sp:PriceOffering ;
                      sp:priceValue ?price ;
                      sp:forConfiguration ?config ;
                      sp:offeredBy ?store .

            ?config sp:hasBasePhone ?phone .
            BIND(STRAFTER(STR(?phone), "instance/phone/") AS ?phone_id)

            ?store sp:storeName ?storeName .
        }
        ORDER BY ?phone_id ?price
        """

        price_count = 0
        for row in self.graph.query(query):
            phone_id = str(row.phone_id)
            if phone_id in self.phones:
                phone = self.phones[phone_id]
                price = float(row.price)
                store = str(row.storeName)

                # Track minimum price
                if phone.min_price is None or price < phone.min_price:
                    phone.min_price = price

                # Track stores
                if phone.stores is None:
                    phone.stores = []
                if store not in phone.stores:
                    phone.stores.append(store)

                price_count += 1

        phones_with_prices = sum(1 for p in self.phones.values() if p.min_price is not None)
        print(f"  Loaded {price_count} price offerings for {phones_with_prices} phones")
    
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
- "features": list of desired features like ["gaming", "photography", "5g", "big_battery", "high_camera", "amoled", "vlogging", "nfc", "good_selfie", "fast_processor", "high_ram", "large_storage"]
- "brand": brand name if mentioned (Samsung, Apple, Xiaomi, Google, OnePlus, etc.), else null
- "budget": "flagship", "midrange", "budget", or null
- "use_case": main use case if clear, one of: "gaming", "photography", "vlogging", "business", "everyday", "minimalist", else null
- "max_price": maximum price in EUR if mentioned (e.g., "under 500" -> 500, "around 800" -> 900), else null
- "min_price": minimum price in EUR if mentioned (e.g., "over 1000" -> 1000), else null

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

            parsed = json.loads(result)
            # Ensure price fields exist
            if "max_price" not in parsed:
                parsed["max_price"] = None
            if "min_price" not in parsed:
                parsed["min_price"] = None
            return parsed
        except json.JSONDecodeError:
            return {
                "intent": "find_phones",
                "features": [],
                "brand": None,
                "budget": None,
                "use_case": None,
                "max_price": None,
                "min_price": None,
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
        max_price = intent.get("max_price")
        min_price = intent.get("min_price")

        filtered_phones = []
        for phone_id, score in phone_scores:
            if phone_id not in self.phones:
                continue

            phone = self.phones[phone_id]

            # Brand filter
            if brand and brand.lower() not in ["none", "null", ""]:
                if phone.brand.lower() != brand.lower():
                    continue

            # Budget tier filter
            if budget and budget.lower() not in ["none", "null", ""]:
                if not self._matches_budget_tier(phone, budget.lower()):
                    continue

            # Price range filter (actual EUR prices)
            if max_price is not None or min_price is not None:
                if not self._matches_price_range(phone, min_price, max_price):
                    continue

            filtered_phones.append((phone, score))

            if len(filtered_phones) >= top_k:
                break

        return filtered_phones
    
    def _compute_spec_bonus(self, phone_id: str, intent: dict) -> float:
        """
        Compute a spec-based bonus to improve ranking for specific use-cases.

        This hybrid approach uses phone specs and review sentiment to boost
        the embedding-based score, ensuring phones with relevant specs and
        positive user feedback rank higher for specific use-cases.
        """
        if phone_id not in self.phones:
            return 0.0

        phone = self.phones[phone_id]
        name_lower = phone.name.lower() if phone.name else ""
        bonus = 0.0

        use_case = intent.get("use_case", "").lower() if intent.get("use_case") else ""
        features = [f.lower() for f in intent.get("features", [])]

        # Gaming bonus: high refresh rate, gaming phones, processor, RAM
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
            # Big RAM (use actual RAM data)
            if phone.ram_gb and phone.ram_gb >= 12:
                bonus += 0.2
            elif phone.ram_gb and phone.ram_gb >= 8:
                bonus += 0.1
            # High-end processor
            if phone.processor:
                proc_lower = phone.processor.lower()
                if any(p in proc_lower for p in ["snapdragon 8 gen", "a17", "a18", "dimensity 9"]):
                    bonus += 0.25
            # Sentiment bonus for gaming/performance tags
            bonus += self._get_sentiment_bonus(phone, ["Performance", "Gaming", "Speed"])

        # Photography bonus: high camera MP + sentiment
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
            # Sentiment bonus for camera tags
            bonus += self._get_sentiment_bonus(phone, ["Camera", "Photo", "Photography"])

        # Vlogging bonus: good SELFIE camera + video features
        if use_case in ["vlogging", "video"] or "vlogging" in features or "video" in features:
            # Selfie camera is key for vlogging
            if phone.selfie_camera and phone.selfie_camera >= 32:
                bonus += 0.3
            elif phone.selfie_camera and phone.selfie_camera >= 16:
                bonus += 0.15
            # Main camera still matters
            if phone.camera and phone.camera >= 50:
                bonus += 0.1
            # Flip phones good for vlogging (use main cam for selfies)
            if "flip" in name_lower:
                bonus += 0.3
            # Sentiment bonus for video/selfie tags
            bonus += self._get_sentiment_bonus(phone, ["Video", "Selfie", "Camera"])

        # Good selfie feature
        if "good_selfie" in features or "selfie" in features:
            if phone.selfie_camera and phone.selfie_camera >= 32:
                bonus += 0.3
            elif phone.selfie_camera and phone.selfie_camera >= 16:
                bonus += 0.15

        # Business bonus: NFC for payments, good battery, professional look
        if use_case in ["business", "work", "professional"] or "business" in features:
            if phone.supports_nfc:
                bonus += 0.25
            if phone.battery and phone.battery >= 4500:
                bonus += 0.1
            # Sentiment bonus for business-related tags
            bonus += self._get_sentiment_bonus(phone, ["Battery", "Value", "Build"])

        # NFC bonus
        if "nfc" in features:
            if phone.supports_nfc:
                bonus += 0.3

        # Big battery bonus
        if "battery" in features or "big_battery" in features:
            if phone.battery and phone.battery >= 6000:
                bonus += 0.3
            elif phone.battery and phone.battery >= 5000:
                bonus += 0.15
            # Sentiment bonus for battery tag
            bonus += self._get_sentiment_bonus(phone, ["Battery"])

        # 5G bonus
        if "5g" in features:
            if phone.supports_5g:
                bonus += 0.2

        # High RAM bonus
        if "high_ram" in features:
            if phone.ram_gb and phone.ram_gb >= 12:
                bonus += 0.25
            elif phone.ram_gb and phone.ram_gb >= 8:
                bonus += 0.1

        # Large storage bonus
        if "large_storage" in features:
            if phone.storage_gb and phone.storage_gb >= 512:
                bonus += 0.25
            elif phone.storage_gb and phone.storage_gb >= 256:
                bonus += 0.1

        # Fast processor bonus
        if "fast_processor" in features:
            if phone.processor:
                proc_lower = phone.processor.lower()
                if any(p in proc_lower for p in ["snapdragon 8 gen", "a17", "a18", "dimensity 9"]):
                    bonus += 0.3
                elif any(p in proc_lower for p in ["snapdragon 7", "dimensity 8", "a16"]):
                    bonus += 0.15

        return bonus

    def _get_sentiment_bonus(self, phone: Phone, tags: list[str]) -> float:
        """
        Calculate a bonus based on review sentiment for specific tags.

        Returns a bonus between -0.2 and +0.2 based on positive/negative ratio.
        """
        if not phone.sentiment:
            return 0.0

        total_positive = 0
        total_negative = 0

        for tag in tags:
            if tag in phone.sentiment:
                pos, neg = phone.sentiment[tag]
                total_positive += pos
                total_negative += neg

        if total_positive + total_negative == 0:
            return 0.0

        # Calculate sentiment ratio (-1 to 1)
        ratio = (total_positive - total_negative) / (total_positive + total_negative)

        # Scale to bonus (-0.2 to +0.2)
        return ratio * 0.2

    def _matches_budget_tier(self, phone: Phone, budget: str) -> bool:
        """
        Check if phone matches budget tier using ACTUAL PRICES when available.

        Price tiers (EUR):
        - Flagship: >= 800 EUR
        - Midrange: 300-799 EUR
        - Budget: < 300 EUR

        Falls back to name-pattern heuristics if no price data available.
        """
        # Use actual price if available
        if phone.min_price is not None:
            if budget == "flagship":
                return phone.min_price >= 800
            elif budget == "midrange":
                return 300 <= phone.min_price < 800
            elif budget == "budget":
                return phone.min_price < 300

        # Fallback to name-pattern heuristics
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

    def _matches_price_range(self, phone: Phone, min_price: float | None, max_price: float | None) -> bool:
        """Check if phone price falls within the specified range."""
        if phone.min_price is None:
            # If no price data, allow through (don't filter out)
            return True

        if min_price is not None and phone.min_price < min_price:
            return False
        if max_price is not None and phone.min_price > max_price:
            return False

        return True
    
    def generate_response(self, question: str, phones: list[tuple[Phone, float]], intent: dict) -> str:
        """Generate a natural language response using Ollama."""

        if not phones:
            return "I couldn't find phones matching your criteria. Try being more specific about features like gaming, photography, or battery life."

        # Format phone info with similarity scores, prices, and sentiment
        phones_info = []
        for i, (phone, score) in enumerate(phones[:5], 1):
            info = f"{i}. {phone.name} ({phone.brand}) [similarity: {score:.2f}]"

            # Price info
            if phone.min_price:
                info += f" - FROM {phone.min_price:.0f} EUR"
                if phone.stores:
                    info += f" at {', '.join(phone.stores[:2])}"

            # Core specs
            specs = []
            if phone.battery:
                specs.append(f"battery: {phone.battery}mAh")
            if phone.camera:
                specs.append(f"main cam: {phone.camera}MP")
            if phone.selfie_camera:
                specs.append(f"selfie: {phone.selfie_camera}MP")
            if phone.refresh_rate:
                specs.append(f"refresh: {phone.refresh_rate}Hz")
            if phone.processor:
                specs.append(f"chip: {phone.processor[:25]}")
            if phone.ram_gb:
                specs.append(f"RAM: {phone.ram_gb}GB")
            if phone.storage_gb:
                specs.append(f"storage: {phone.storage_gb}GB")
            if phone.supports_5g:
                specs.append("5G")
            if phone.supports_nfc:
                specs.append("NFC")

            if specs:
                info += f"\n   Specs: {', '.join(specs)}"

            # Sentiment summary (if available)
            if phone.sentiment:
                pos_tags = []
                neg_tags = []
                for tag, (pos, neg) in phone.sentiment.items():
                    if pos > neg and pos >= 3:
                        pos_tags.append(f"{tag}(+{pos})")
                    elif neg > pos and neg >= 3:
                        neg_tags.append(f"{tag}(-{neg})")
                if pos_tags:
                    info += f"\n   User reviews: Positive on {', '.join(pos_tags[:3])}"
                if neg_tags:
                    info += f" | Negative on {', '.join(neg_tags[:2])}"

            phones_info.append(info)

        phones_text = "\n".join(phones_info)

        # Build price context
        price_context = ""
        if intent.get("max_price"):
            price_context += f"\nMax budget: {intent['max_price']} EUR"
        if intent.get("min_price"):
            price_context += f"\nMin budget: {intent['min_price']} EUR"

        prompt = f"""Based on the user's question and the phones found via embedding similarity, provide a helpful response.

IMPORTANT: Only recommend phones from the list below. These were found using AI-learned patterns from user preferences.

User Question: {question}

Detected Intent: {intent.get('intent', 'find_phones')}
Use-case: {intent.get('use_case') or 'general'}
Features wanted: {', '.join(intent.get('features', [])) or 'not specified'}{price_context}

PHONES FOUND (ranked by embedding similarity + specs + user reviews):
{phones_text}

Provide a concise response that:
1. Recommends 2-3 best matches from the list
2. Mentions the price and key specs that match their needs

Keep it under 200 words. Use ONLY phones from the list above."""

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
        output.append(f"\nTop Matching Phones (by embedding similarity + specs + reviews):")

        for i, phone in enumerate(result.relevant_phones[:5], 1):
            # Price line
            price_str = f"{phone.min_price:.0f} EUR" if phone.min_price else "Price N/A"

            # Core specs
            specs = []
            if phone.battery:
                specs.append(f"{phone.battery}mAh")
            if phone.camera:
                specs.append(f"{phone.camera}MP")
            if phone.selfie_camera:
                specs.append(f"selfie:{phone.selfie_camera}MP")
            if phone.refresh_rate:
                specs.append(f"{phone.refresh_rate}Hz")
            if phone.ram_gb:
                specs.append(f"{phone.ram_gb}GB RAM")
            if phone.supports_5g:
                specs.append("5G")
            if phone.supports_nfc:
                specs.append("NFC")
            spec_str = " | ".join(specs) if specs else "N/A"

            output.append(f"  {i}. {phone.name} ({phone.brand}) - {price_str}")
            output.append(f"     {spec_str}")

            # Processor
            if phone.processor:
                output.append(f"     Processor: {phone.processor}")

            # Sentiment
            if phone.sentiment:
                pos_tags = [t for t, (p, n) in phone.sentiment.items() if p > n]
                if pos_tags:
                    output.append(f"     Positive reviews: {', '.join(pos_tags[:3])}")

            # Stores
            if phone.stores:
                output.append(f"     Available at: {', '.join(phone.stores[:3])}")

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

        # NEW: Price range filtering (actual EUR prices)
        "Good phone for gaming under 600 euros",

        # NEW: NFC for business/payments
        "I need a phone with NFC for contactless payments",

        # NEW: Selfie camera focused
        "Best phone for selfies and video calls",

        # NEW: High-end processor query
        "Phone with the fastest processor for mobile gaming",
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