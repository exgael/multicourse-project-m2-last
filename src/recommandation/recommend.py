from pathlib import Path
import argparse
import json
import torch
from pykeen.triples import TriplesFactory

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "output" / "models" / "link_prediction"


def load_model():
    return torch.load(MODEL_DIR / "trained_model.pkl", weights_only=False)


def load_triples_factory() -> TriplesFactory:
    return TriplesFactory.from_path_binary(MODEL_DIR / "training_triples")


def load_config_names() -> dict[str, str]:
    """Load phone configuration names, keyed by config ID format.

    Config IDs are like: apple_apple_iphone_11_64gb_4gb
    Maps to display names like: Apple iPhone 11 (64GB/4GB)
    """
    phones_file = DATA_DIR / "phones.json"
    with open(phones_file, "r") as f:
        phones = json.load(f)

    # Create a mapping from phone_id to phone_name
    phone_names = {p["phone_id"]: p["phone_name"] for p in phones}
    return phone_names


def config_id_to_display_name(config_id: str, phone_names: dict[str, str]) -> str:
    """Convert a config ID to a human-readable display name.

    Config ID format: {phone_id}_{storage}gb_{ram}gb
    Example: apple_apple_iphone_11_64gb_4gb -> Apple iPhone 11 (64GB/4GB)
    """
    # Parse config ID to extract storage and RAM
    parts = config_id.rsplit("_", 2)  # Split from the right to get [phone_id, storage, ram]
    if len(parts) >= 3 and parts[-1].endswith("gb") and parts[-2].endswith("gb"):
        phone_id = "_".join(parts[:-2])
        storage = parts[-2].upper()  # e.g., "64gb" -> "64GB"
        ram = parts[-1].upper()  # e.g., "4gb" -> "4GB"
        phone_name = phone_names.get(phone_id, phone_id)
        return f"{phone_name} ({storage}/{ram})"

    # Fallback if format doesn't match
    return config_id


def recommend(user_id: str, top_k: int = 10) -> None:
    """Recommend phone configurations for a user based on their embedding."""
    model = load_model()
    phone_names = load_config_names()
    tf = load_triples_factory()

    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    if user_id not in entity_to_id:
        print(f"User '{user_id}' not found in training data")
        return

    user_idx = entity_to_id[user_id]
    likes_rel_idx = relation_to_id["likes"]

    # Get all configuration IDs (format: brand_model_storage_ram without prefix)
    all_config_ids = [e for e in entity_to_id.keys() if e.endswith("gb")]
    config_indices = torch.tensor([entity_to_id[cid] for cid in all_config_ids], dtype=torch.long)

    h = torch.full((len(config_indices),), user_idx, dtype=torch.long)
    r = torch.full((len(config_indices),), likes_rel_idx, dtype=torch.long)

    hrt_batch = torch.stack([h, r, config_indices], dim=1)
    scores = model.score_hrt(hrt_batch).squeeze()
    top_indices = scores.argsort(descending=True)[:top_k]

    print(f"\nTop {top_k} recommendations for user '{user_id}':")
    print("-" * 80)

    for rank, idx in enumerate(top_indices, 1):
        config_id = all_config_ids[idx]
        display_name = config_id_to_display_name(config_id, phone_names)
        score = scores[idx].item()
        print(f"{rank:2d}. {display_name} (score: {score:.4f})")


def recommend_by_interests(interests: list[str], top_k: int = 10) -> None:
    """
    Recommend phone configurations based on use-case interests.

    Uses proper link prediction: predicts (config, suitableFor, interest) triples.
    The model learned suitableFor from user behavior data.
    """
    model = load_model()
    phone_names = load_config_names()
    tf = load_triples_factory()

    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    # Validate suitableFor relation exists
    if "suitableFor" not in relation_to_id:
        print("Error: 'suitableFor' relation not in model. Retrain the model.")
        return

    # Validate interests exist in the model
    valid_interests = [i for i in interests if i in entity_to_id]
    if not valid_interests:
        print(f"None of the interests {interests} found in model")
        return

    # Get all configuration IDs (format: brand_model_storage_ram ending with gb)
    all_config_ids = [e for e in entity_to_id.keys() if e.endswith("gb")]

    if not all_config_ids:
        print("No configurations found in model")
        return

    suitable_rel_idx = relation_to_id["suitableFor"]
    aggregate_scores: dict[str, float] = {cid: 0.0 for cid in all_config_ids}

    config_indices = torch.tensor([entity_to_id[cid] for cid in all_config_ids], dtype=torch.long)

    # For each interest, score all configs on (config, suitableFor, interest)
    for interest in valid_interests:
        interest_idx = entity_to_id[interest]

        h = config_indices
        r = torch.full((len(config_indices),), suitable_rel_idx, dtype=torch.long)
        t = torch.full((len(config_indices),), interest_idx, dtype=torch.long)

        hrt_batch = torch.stack([h, r, t], dim=1)
        scores = model.score_hrt(hrt_batch).squeeze()

        for config_id, score in zip(all_config_ids, scores):
            aggregate_scores[config_id] += score.item()

    sorted_configs = sorted(aggregate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\nTop {top_k} configurations for interests: {', '.join(interests)}")
    print("-" * 80)

    for rank, (config_id, score) in enumerate(sorted_configs, 1):
        display_name = config_id_to_display_name(config_id, phone_names)
        print(f"{rank:2d}. {display_name} (score: {score:.4f})")

def recommend_by_interest_embedding(interests: list[str], top_k: int = 5):
    model = load_model()
    tf = load_triples_factory()
    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    # 1. Get IDs for the input interests (e.g., "Gaming", "Photography")
    valid_interest_ids = [entity_to_id[i] for i in interests if i in entity_to_id]
    
    if not valid_interest_ids:
        print("No valid interests found.")
        return

    # 2. Calculate the "Composite Interest" (The Centroid)
    # Shape: (1, embedding_dim)
    entity_embeddings = model.entity_representations[0](indices=None)
    interest_vectors = entity_embeddings[torch.tensor(valid_interest_ids)]
    composite_interest_vector = torch.mean(interest_vectors, dim=0).unsqueeze(0)

    # 3. Predict phones using 'suitableFor' (Inverse)
    # Logic: Interest + suitableFor_inverse ≈ Phone
    # If your model doesn't have explicit inverses, we use the tail prediction logic:
    # Phone + suitableFor ≈ Interest  =>  We want to find 'h' given 'r' and 't'
    
    suitable_rel_idx = relation_to_id["suitableFor"]
    relation_vector = model.relation_representations[0](indices=torch.tensor([suitable_rel_idx]))
    
    all_phone_ids = [e for e in entity_to_id.keys() if e.startswith("instance/phone/")]
    phone_indices = torch.tensor([entity_to_id[pid] for pid in all_phone_ids])
    phone_vectors = entity_embeddings[phone_indices]

    # Score Calculation (TransE: -||h + r - t||)
    # We test every Phone (h) against fixed Relation (suitableFor) and fixed Target (Composite Interest)
    
    # Broadcast r and t to match number of phones
    # h + r - t
    term = phone_vectors + relation_vector - composite_interest_vector
    distances = torch.norm(term, dim=1, p=2)
    scores = -distances

    # 4. Rank and Print
    top_indices = scores.argsort(descending=True)[:top_k]
    
    print(f"Top Recommendations for mixed interests ({', '.join(interests)}):")
    for idx in top_indices:
        phone_id = all_phone_ids[idx]
        score = scores[idx].item()
        print(f"{phone_id} (Score: {score:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get phone recommendations")
    parser.add_argument("--user", type=str, help="User ID (e.g., gamer_0001)")
    parser.add_argument("--interests", nargs="+", help="Use-case interests (e.g., Gaming Photography)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")

    args = parser.parse_args()

    if args.user:
        recommend(args.user, args.top_k)
    elif args.interests:
        recommend_by_interests(args.interests, args.top_k)
    else:
        print("Provide either --user or --interests")
