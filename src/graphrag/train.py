"""
Train KG Embeddings with REAL phones from the Knowledge Graph.

This module extracts triples from the actual KG structure:
- Phones: instance/phone/{phone_id} with specs (battery, camera, refresh_rate...)
- Users: instance/user/{user_id} with interests and likes
- Use-cases: vocab/{UseCase} (ProGaming, Photography, Vlogging...)

Relations:
- user interestedIn usecase
- user likes phone  
- phone hasSpec spec_value (discretized)
- phone suitableFor usecase (derived from user preferences)

This creates embeddings for REAL phones (Samsung Galaxy S24, iPhone 16, etc.)
instead of abstract config IDs.
"""

from pathlib import Path
from rdflib import Graph, Namespace
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
FINAL_KG_TTL = DATA_DIR / "rdf" / "knowledge_graph_full.ttl"
MODEL_DIR = OUTPUT_DIR / "models" / "phone_embeddings"

SP = Namespace("http://example.org/smartphone#")
SPV = Namespace("http://example.org/smartphone/vocab/")
SPINST = Namespace("http://example.org/smartphone/instance/")


def load_triples_from_kg() -> tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    """Load triples from the knowledge graph using actual phone URIs."""
    print(f"Loading knowledge graph from {FINAL_KG_TTL}...")
    g = Graph()
    g.parse(FINAL_KG_TTL, format="turtle")
    print(f"Loaded {len(g)} RDF triples")

    all_triples: list[tuple[str, str, str]] = []

    # =========================================================================
    # 1. Extract PhoneConfiguration specs as triples
    # The KG uses PhoneConfiguration (instance/config/) linked to BasePhone
    # =========================================================================
    phone_spec_query = """
    PREFIX sp: <http://example.org/smartphone#>

    SELECT ?config_id ?property ?value WHERE {
        ?config a sp:PhoneConfiguration .
        BIND(STRAFTER(STR(?config), "instance/config/") AS ?config_id)
        
        # Get specs from the linked BasePhone
        ?config sp:hasBasePhone ?basephone .
        
        {
            ?basephone sp:batteryCapacityMah ?val .
            BIND("hasBattery" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?basephone sp:refreshRateHz ?val .
            BIND("hasRefreshRate" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?basephone sp:mainCameraMP ?val .
            BIND("hasCamera" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?basephone sp:selfieCameraMP ?val .
            BIND("hasSelfieCamera" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?basephone sp:supports5G ?val .
            BIND("supports5G" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?basephone sp:supportsNFC ?val .
            BIND("supportsNFC" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?basephone sp:hasBrand ?brand .
            BIND("hasBrand" AS ?property)
            BIND(STRAFTER(STR(?brand), "instance/brand/") AS ?value)
        } UNION {
            # RAM from config itself
            ?config sp:ramGB ?val .
            BIND("hasRAM" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            # Storage from config itself
            ?config sp:storageGB ?val .
            BIND("hasStorage" AS ?property)
            BIND(STR(?val) AS ?value)
        }
    }
    """

    phone_spec_count = 0
    phone_ids = set()
    for row in g.query(phone_spec_query):
        config_id = str(row.config_id)
        prop = str(row.property)
        value = str(row.value)
        
        if not config_id:
            continue
        
        phone_ids.add(config_id)
        
        # Discretize numeric values
        if prop == "hasBattery":
            value = discretize_battery(value)
        elif prop == "hasRefreshRate":
            value = discretize_refresh_rate(value)
        elif prop == "hasCamera":
            value = discretize_camera(value)
        elif prop == "hasSelfieCamera":
            value = discretize_selfie(value)
        elif prop == "hasRAM":
            value = discretize_ram(value)
        elif prop == "hasStorage":
            value = discretize_storage(value)
        
        all_triples.append((config_id, prop, value))
        phone_spec_count += 1

    print(f"Extracted {phone_spec_count} phone spec triples ({len(phone_ids)} unique phones)")

    # =========================================================================
    # 2. Extract User interests (user -> interestedIn -> usecase)
    # =========================================================================
    user_interest_query = """
    PREFIX sp: <http://example.org/smartphone#>

    SELECT ?user_id ?usecase WHERE {
        ?user a sp:User ;
              sp:interestedIn ?uc .
        BIND(STRAFTER(STR(?user), "instance/user/") AS ?user_id)
        BIND(STRAFTER(STR(?uc), "vocab/") AS ?usecase)
    }
    """

    user_interest_count = 0
    for row in g.query(user_interest_query):
        user_id = str(row.user_id)
        usecase = str(row.usecase)
        all_triples.append((f"user/{user_id}", "interestedIn", f"usecase/{usecase}"))
        user_interest_count += 1

    print(f"Extracted {user_interest_count} user interest triples")

    # =========================================================================
    # 3. Extract User likes (user -> likes -> config)
    # =========================================================================
    user_likes_query = """
    PREFIX sp: <http://example.org/smartphone#>

    SELECT ?user_id ?config_id WHERE {
        ?user a sp:User ;
              sp:likes ?config .
        BIND(STRAFTER(STR(?user), "instance/user/") AS ?user_id)
        BIND(STRAFTER(STR(?config), "instance/config/") AS ?config_id)
    }
    """

    user_likes_count = 0
    user_phones: dict[str, set[str]] = {}
    for row in g.query(user_likes_query):
        user_id = str(row.user_id)
        config_id = str(row.config_id)
        
        if not config_id:
            continue
        
        all_triples.append((f"user/{user_id}", "likes", config_id))
        
        if user_id not in user_phones:
            user_phones[user_id] = set()
        user_phones[user_id].add(config_id)
        user_likes_count += 1

    print(f"Extracted {user_likes_count} user-phone likes triples")

    # =========================================================================
    # 4. Derive phone -> suitableFor -> usecase
    # Based on: if user likes phone AND user interestedIn usecase 
    #           => phone suitableFor usecase
    # =========================================================================
    user_interests: dict[str, set[str]] = {}
    for row in g.query(user_interest_query):
        user_id = str(row.user_id)
        usecase = str(row.usecase)
        if user_id not in user_interests:
            user_interests[user_id] = set()
        user_interests[user_id].add(usecase)

    suitable_for_triples: set[tuple[str, str, str]] = set()
    for user_id, phones in user_phones.items():
        if user_id not in user_interests:
            continue
        for config_id in phones:
            for usecase in user_interests[user_id]:
                suitable_for_triples.add((config_id, "suitableFor", f"usecase/{usecase}"))

    for triple in suitable_for_triples:
        all_triples.append(triple)

    print(f"Derived {len(suitable_for_triples)} phone-usecase suitableFor triples")

    # =========================================================================
    # 5. Add usecase hierarchy/relations
    # =========================================================================
    usecase_relations = [
        ("usecase/ProGaming", "relatedTo", "usecase/EverydayUse"),
        ("usecase/ProPhotography", "relatedTo", "usecase/Vlogging"),
        ("usecase/Vlogging", "relatedTo", "usecase/ProPhotography"),
        ("usecase/Business", "relatedTo", "usecase/EverydayUse"),
    ]
    for triple in usecase_relations:
        all_triples.append(triple)
    
    print(f"Added {len(usecase_relations)} usecase relation triples")

    print(f"\nTotal training triples: {len(all_triples)}")

    # Create PyKEEN triples factory
    tf = TriplesFactory.from_labeled_triples(np.array(all_triples))
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    print(f"\nDataset Split:")
    print(f"  Training: {training.num_triples} triples")
    print(f"  Testing: {testing.num_triples} triples")
    print(f"  Validation: {validation.num_triples} triples")
    print(f"  Entities: {tf.num_entities}")
    print(f"  Relations: {tf.num_relations}")

    # Show entity breakdown
    entities = list(tf.entity_to_id.keys())
    phones = [e for e in entities if e.startswith("phone/")]
    users = [e for e in entities if e.startswith("user/")]
    usecases = [e for e in entities if e.startswith("usecase/")]
    brands = [e for e in entities if not any(e.startswith(p) for p in ["phone/", "user/", "usecase/", "battery_", "camera_", "refresh_", "true", "false"])]
    
    print(f"\nEntity breakdown:")
    print(f"  Phones: {len(phones)}")
    print(f"  Users: {len(users)}")
    print(f"  Use-cases: {len(usecases)}")
    print(f"  Brands: {len([b for b in brands if b and not b.startswith('battery') and not b.startswith('camera') and not b.startswith('refresh')])}")
    print(f"  Spec values: {len([e for e in entities if e.startswith('battery_') or e.startswith('camera_') or e.startswith('refresh_')])}")

    return training, testing, validation


def discretize_battery(value: str) -> str:
    """Discretize battery capacity into meaningful segments."""
    try:
        mah = int(float(value))
    except ValueError:
        return "battery_unknown"

    if mah >= 6000:
        return "battery_huge"      # Gaming/power user
    elif mah >= 5000:
        return "battery_large"     # All-day battery
    elif mah >= 4000:
        return "battery_medium"    # Standard
    else:
        return "battery_small"     # Compact phones


def discretize_refresh_rate(value: str) -> str:
    """Discretize refresh rate for use-case matching."""
    try:
        hz = int(float(value))
    except ValueError:
        return "refresh_standard"

    if hz >= 144:
        return "refresh_gaming"    # Pro gaming
    elif hz >= 120:
        return "refresh_smooth"    # Smooth scrolling
    elif hz >= 90:
        return "refresh_good"      # Better than 60
    else:
        return "refresh_standard"  # Basic 60Hz


def discretize_camera(value: str) -> str:
    """Discretize camera MP for photography matching."""
    try:
        mp = int(float(value))
    except ValueError:
        return "camera_unknown"

    if mp >= 200:
        return "camera_flagship"   # Pro photography
    elif mp >= 100:
        return "camera_excellent"  # High-end
    elif mp >= 50:
        return "camera_good"       # Modern standard
    elif mp >= 12:
        return "camera_basic"      # Entry level
    else:
        return "camera_minimal"


def discretize_selfie(value: str) -> str:
    """Discretize selfie camera MP for vlogging matching."""
    try:
        mp = int(float(value))
    except ValueError:
        return "selfie_unknown"

    if mp >= 32:
        return "selfie_vlogging"   # Great for vlogging
    elif mp >= 16:
        return "selfie_good"       # Good selfies
    else:
        return "selfie_basic"      # Basic


def discretize_ram(value: str) -> str:
    """Discretize RAM for performance matching."""
    try:
        gb = int(float(value))
    except ValueError:
        return "ram_unknown"

    if gb >= 16:
        return "ram_gaming"        # Pro gaming
    elif gb >= 12:
        return "ram_flagship"      # High-end
    elif gb >= 8:
        return "ram_good"          # Standard
    else:
        return "ram_basic"         # Entry


def discretize_storage(value: str) -> str:
    """Discretize storage capacity."""
    try:
        gb = int(float(value))
    except ValueError:
        return "storage_unknown"

    if gb >= 512:
        return "storage_huge"      # Power user
    elif gb >= 256:
        return "storage_large"     # Good capacity
    elif gb >= 128:
        return "storage_medium"    # Standard
    else:
        return "storage_small"     # Basic


def train_model() -> None:
    """Train RotatE model for phone recommendation.
    
    RotatE models relations as rotations in complex space, which captures:
    - Symmetric relations (e.g., relatedTo between use-cases)
    - Antisymmetric relations (e.g., likes, suitableFor)
    - Composition patterns (user -> likes -> phone -> suitableFor -> usecase)
    """

    training, testing, validation = load_triples_from_kg()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTraining RotatE model...")
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="RotatE",
        model_kwargs=dict(embedding_dim=128),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=0.001),
        loss="NSSALoss",
        loss_kwargs=dict(margin=9.0, adversarial_temperature=1.0),
        training_kwargs=dict(num_epochs=150, batch_size=256),
        evaluator_kwargs=dict(filtered=True),
        evaluation_kwargs=dict(batch_size=256),
        random_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    result.save_to_directory(MODEL_DIR)

    print(f"\nModel saved to {MODEL_DIR}")
    print("\nTest Metrics:")
    print(f"  MRR: {result.metric_results.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"  Hits@1: {result.metric_results.get_metric('hits_at_1'):.4f}")
    print(f"  Hits@3: {result.metric_results.get_metric('hits_at_3'):.4f}")
    print(f"  Hits@10: {result.metric_results.get_metric('hits_at_10'):.4f}")


if __name__ == "__main__":
    train_model()