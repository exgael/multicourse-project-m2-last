from pathlib import Path
from rdflib import Graph, Namespace
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
FINAL_KG_TTL = OUTPUT_DIR / "final_knowledge_graph.ttl"
MODEL_DIR = OUTPUT_DIR / "models" / "link_prediction"

SP = Namespace("http://example.org/smartphone#")
SPV = Namespace("http://example.org/smartphone/vocab/")
SPINST = Namespace("http://example.org/smartphone/instance/")


def load_triples_from_kg() -> tuple[TriplesFactory, TriplesFactory, TriplesFactory]:
    """Load triples directly from the knowledge graph."""
    print(f"Loading knowledge graph from {FINAL_KG_TTL}...")
    g = Graph()
    g.parse(FINAL_KG_TTL, format="turtle")
    print(f"Loaded {len(g)} RDF triples")

    all_triples: list[tuple[str, str, str]] = []

    # Extract phone datatype properties as triples
    # Removed: mainCameraMP, selfieCameraMP (not useful for use-case classification)
    phone_spec_query = """
    PREFIX sp: <http://example.org/smartphone#>

    SELECT ?phone_id ?property ?value WHERE {
        ?phone a sp:Smartphone .
        BIND(STRAFTER(STR(?phone), "phone/") AS ?phone_id)

        {
            ?phone sp:batteryCapacityMah ?val .
            BIND("batteryCapacityMah" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?phone sp:refreshRateHz ?val .
            BIND("refreshRateHz" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?phone sp:supports5G ?val .
            BIND("supports5G" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?phone sp:supportsNFC ?val .
            BIND("supportsNFC" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?phone sp:priceEUR ?val .
            BIND("priceEUR" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?phone sp:ramGB ?val .
            BIND("ramGB" AS ?property)
            BIND(STR(?val) AS ?value)
        } UNION {
            ?phone sp:storageGB ?val .
            BIND("storageGB" AS ?property)
            BIND(STR(?val) AS ?value)
        }
    }
    """

    for row in g.query(phone_spec_query):
        phone_id = str(row.phone_id)
        prop = str(row.property)
        value = str(row.value)
        # Discretize continuous values
        if prop == "priceEUR":
            value = discretize_price(value)
        elif prop == "batteryCapacityMah":
            value = discretize_battery(value)
        elif prop == "refreshRateHz":
            value = discretize_refresh_rate(value)
        elif prop == "ramGB":
            value = discretize_ram(value)
        elif prop == "storageGB":
            value = discretize_storage(value)
        all_triples.append((phone_id, prop, value))

    print(f"Extracted {len(all_triples)} phone spec triples")

    # Extract user interests
    user_interest_query = """
    PREFIX sp: <http://example.org/smartphone#>

    SELECT ?user_id ?usecase WHERE {
        ?user a sp:User ;
              sp:userId ?user_id ;
              sp:interestedIn ?uc .
        BIND(STRAFTER(STR(?uc), "vocab/") AS ?usecase)
    }
    """

    user_interest_count = 0
    for row in g.query(user_interest_query):
        user_id = str(row.user_id)
        usecase = str(row.usecase)
        all_triples.append((user_id, "interestedIn", usecase))
        user_interest_count += 1

    print(f"Extracted {user_interest_count} user interest triples")

    # Extract user-phone likes
    user_phone_query = """
    PREFIX sp: <http://example.org/smartphone#>

    SELECT ?user_id ?phone_id WHERE {
        ?user a sp:User ;
              sp:userId ?user_id ;
              sp:likes ?phone .
        BIND(STRAFTER(STR(?phone), "phone/") AS ?phone_id)
    }
    """

    user_phone_count = 0
    for row in g.query(user_phone_query):
        user_id = str(row.user_id)
        phone_id = str(row.phone_id)
        all_triples.append((user_id, "likes", phone_id))
        user_phone_count += 1

    print(f"Extracted {user_phone_count} user-phone like triples")

    # Derive suitableFor triples from user data
    # Logic: if user likes phone AND user interestedIn usecase → phone suitableFor usecase
    user_interests: dict[str, set[str]] = {}
    user_phones: dict[str, set[str]] = {}

    for row in g.query(user_interest_query):
        user_id = str(row.user_id)
        usecase = str(row.usecase)
        if user_id not in user_interests:
            user_interests[user_id] = set()
        user_interests[user_id].add(usecase)

    for row in g.query(user_phone_query):
        user_id = str(row.user_id)
        phone_id = str(row.phone_id)
        if user_id not in user_phones:
            user_phones[user_id] = set()
        user_phones[user_id].add(phone_id)

    # Derive: phone suitableFor usecase
    suitable_for_triples: set[tuple[str, str, str]] = set()
    for user_id in user_interests:
        if user_id not in user_phones:
            continue
        for phone_id in user_phones[user_id]:
            for usecase in user_interests[user_id]:
                suitable_for_triples.add((phone_id, "suitableFor", usecase))

    for triple in suitable_for_triples:
        all_triples.append(triple)

    print(f"Derived {len(suitable_for_triples)} phone-usecase suitableFor triples")
    print(f"Total training triples: {len(all_triples)}")

    # Create PyKEEN triples factory
    tf = TriplesFactory.from_labeled_triples(np.array(all_triples))
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    print(f"\nSplit:")
    print(f"  Training: {training.num_triples} triples")
    print(f"  Testing: {testing.num_triples} triples")
    print(f"  Validation: {validation.num_triples} triples")
    print(f"  Entities: {tf.num_entities}")
    print(f"  Relations: {tf.num_relations}")

    return training, testing, validation


def discretize_price(value: str) -> str:
    """Discretize price into segments matching SKOS concepts."""
    try:
        price = float(value)
    except ValueError:
        return "price_budget"

    if price > 900:
        return "price_flagship"
    elif price >= 400:
        return "price_midrange"
    else:
        return "price_budget"


def discretize_battery(value: str) -> str:
    """Discretize battery capacity - aligned with use-case rules (Gaming/EverydayUse need >= 4500)."""
    try:
        mah = int(float(value))
    except ValueError:
        return "battery_small"

    if mah >= 4500:
        return "battery_large"  # Matches Gaming/EverydayUse threshold
    else:
        return "battery_small"


def discretize_refresh_rate(value: str) -> str:
    """Discretize refresh rate - aligned with Gaming use-case (>= 144Hz)."""
    try:
        hz = int(float(value))
    except ValueError:
        return "refresh_standard"

    if hz >= 144:
        return "refresh_gaming"  # Matches Gaming threshold
    else:
        return "refresh_standard"


def discretize_ram(value: str) -> str:
    """Discretize RAM - aligned with Gaming use-case (>= 16GB)."""
    try:
        gb = int(float(value))
    except ValueError:
        return "ram_standard"

    if gb >= 16:
        return "ram_high"  # Matches Gaming threshold
    else:
        return "ram_standard"


def discretize_storage(value: str) -> str:
    """Discretize storage - aligned with Gaming use-case (>= 256GB)."""
    try:
        gb = int(float(value))
    except ValueError:
        return "storage_small"

    if gb >= 256:
        return "storage_large"  # Matches Gaming threshold
    else:
        return "storage_small"


def train_model() -> None:
    """Train TransE model for link prediction."""

    training, testing, validation = load_triples_from_kg()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="TransE",
        model_kwargs=dict(embedding_dim=128),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=0.001),
        training_kwargs=dict(num_epochs=100, batch_size=256),
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