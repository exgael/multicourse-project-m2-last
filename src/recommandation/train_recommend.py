import json
import torch
import numpy as np
from pathlib import Path
from rdflib import Graph, Namespace
from pykeen.triples import TriplesFactory, CoreTriplesFactory
from pykeen.pipeline import pipeline

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
TTL_FILE = OUTPUT_DIR / "final_knowledge_graph.ttl"

# --- Discretization Logic ---
def get_bucket(val, thresholds, labels):
    try:
        v = float(val)
        for t, l in zip(thresholds, labels):
            if v >= t: return l
        return labels[-1]
    except ValueError:
        return labels[-1]

# --- Main Training Flow ---
def main():
    if not TTL_FILE.exists():
        print(f"ERROR: File not found at {TTL_FILE}")
        return

    print("1. Loading Knowledge Graph...")
    g = Graph()
    g.parse(TTL_FILE, format="turtle")
    print(f"   Loaded {len(g)} triples.")

    all_triples = []

    # 2. Extract Phone Specs (Handling BasePhone vs Configuration)
    # We flatten everything to: (PhoneConfigID, relation, Value)
    print("2. Extracting Phone Features...")
    query_specs = """
    PREFIX sp: <http://example.org/smartphone#>
    SELECT ?pid ?ram ?storage ?battery ?price ?refresh WHERE {
        ?config a sp:PhoneConfiguration ;
                sp:hasBasePhone ?base .
        BIND(STRAFTER(STR(?config), "smartphone/") AS ?pid)
        
        OPTIONAL { ?config sp:ramGB ?ram }
        OPTIONAL { ?config sp:storageGB ?storage }
        OPTIONAL { ?base sp:batteryCapacityMah ?battery }
        OPTIONAL { ?base sp:refreshRateHz ?refresh }
        OPTIONAL { ?offer sp:forConfiguration ?config ; sp:priceValue ?price }
    }
    """
    
    for row in g.query(query_specs):
        pid = str(row.pid)
        # Create discrete features
        if row.ram: all_triples.append((pid, "ramGB", get_bucket(row.ram, [12], ["ram_high", "ram_std"])))
        if row.storage: all_triples.append((pid, "storageGB", get_bucket(row.storage, [256], ["storage_large", "storage_std"])))
        if row.battery: all_triples.append((pid, "battery", get_bucket(row.battery, [4500], ["battery_large", "battery_small"])))
        if row.refresh: all_triples.append((pid, "refresh", get_bucket(row.refresh, [120], ["refresh_high", "refresh_std"])))
        if row.price: all_triples.append((pid, "price", get_bucket(row.price, [900, 400], ["price_flagship", "price_midrange", "price_budget"])))

    # 3. Extract User Interactions & Interests
    print("3. Extracting User Data...")
    query_users = """
    PREFIX sp: <http://example.org/smartphone#>
    SELECT ?uid ?interest ?liked_pid WHERE {
        ?user a sp:User ; sp:userId ?uid_lit .
        BIND(STR(?uid_lit) AS ?uid)
        
        { ?user sp:interestedIn ?c . BIND(STRAFTER(STR(?c), "vocab/") AS ?interest) }
        UNION
        { ?user sp:likes ?p . BIND(STRAFTER(STR(?p), "smartphone/") AS ?liked_pid) }
    }
    """
    
    user_interests = {} # uid -> set(interests)
    user_likes = {}     # uid -> set(pids)

    for row in g.query(query_users):
        uid = str(row.uid)
        if row.interest:
            concept = f"vocab/{row.interest}"
            all_triples.append((uid, "interestedIn", concept))
            if uid not in user_interests: user_interests[uid] = set()
            user_interests[uid].add(concept)
        
        if row.liked_pid:
            pid = str(row.liked_pid)
            all_triples.append((uid, "likes", pid))
            if uid not in user_likes: user_likes[uid] = set()
            user_likes[uid].add(pid)

    # 4. Infer 'suitableFor' (The Magic Link)
    # If User likes Phone P AND User is interested in Concept C -> Phone P suitableFor C
    print("4. Inferring 'suitableFor' links...")
    count = 0
    for uid, liked_phones in user_likes.items():
        if uid in user_interests:
            for pid in liked_phones:
                for interest in user_interests[uid]:
                    all_triples.append((pid, "suitableFor", interest))
                    count += 1
    print(f"   Inferred {count} connections.")

    # 5. Prepare Training Data
    print("5. Training Model (RotatE)...")
    tf = TriplesFactory.from_labeled_triples(np.array(all_triples), create_inverse_triples=False)
    
    # Split
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)
    
    # Augment Training with Inverses (Manual injection for better performance)
    training_aug = CoreTriplesFactory.create(
        mapped_triples=training.mapped_triples,
        num_entities=tf.num_entities,
        num_relations=tf.num_relations,
        create_inverse_triples=True 
    )

    # Train
    result = pipeline(
        training=training_aug,
        testing=testing,
        validation=validation,
        model="RotatE",
        model_kwargs={"embedding_dim": 64},
        training_kwargs={"num_epochs": 100, "batch_size": 128},
        device="cpu" # Force CPU for compatibility
    )

    # 6. Save Artifacts Explicitly
    print("6. Saving Artifacts...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save Model Weights
    torch.save(result.model, OUTPUT_DIR / "model.pkl")
    
    # Save Maps as simple JSON (Crucial step)
    with open(OUTPUT_DIR / "entity_to_id.json", "w") as f:
        json.dump(tf.entity_to_id, f)
    with open(OUTPUT_DIR / "relation_to_id.json", "w") as f:
        json.dump(tf.relation_to_id, f)

    print(f"Done! Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()