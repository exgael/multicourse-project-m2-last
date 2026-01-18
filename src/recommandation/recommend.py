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


def load_phones() -> dict[str, str]:
    """Load phone names, keyed by entity ID format (instance/phone/phone_id)."""
    phones_file = DATA_DIR / "phones.json"
    with open(phones_file, "r") as f:
        phones = json.load(f)
    return {f"instance/phone/{p['phone_id']}": p["phone_name"] for p in phones}


def recommend(user_id: str, top_k: int = 10) -> None:
    model = load_model()
    phone_names = load_phones()
    tf = load_triples_factory()

    entity_to_id = tf.entity_to_id
    relation_to_id = tf.relation_to_id

    if user_id not in entity_to_id:
        print(f"User '{user_id}' not found in training data")
        return

    user_idx = entity_to_id[user_id]
    likes_rel_idx = relation_to_id["likes"]

    all_phone_ids = [e for e in entity_to_id.keys() if e.startswith("instance/phone/")]
    phone_indices = torch.tensor([entity_to_id[pid] for pid in all_phone_ids], dtype=torch.long)

    h = torch.full((len(phone_indices),), user_idx, dtype=torch.long)
    r = torch.full((len(phone_indices),), likes_rel_idx, dtype=torch.long)

    hrt_batch = torch.stack([h, r, phone_indices], dim=1)
    scores = model.score_hrt(hrt_batch).squeeze()
    top_indices = scores.argsort(descending=True)[:top_k]

    print(f"\nTop {top_k} recommendations for user '{user_id}':")
    print("-" * 80)

    for rank, idx in enumerate(top_indices, 1):
        phone_id = all_phone_ids[idx]
        phone_name = phone_names.get(phone_id, phone_id)
        score = scores[idx].item()
        print(f"{rank:2d}. {phone_name} (score: {score:.4f})")


def recommend_by_interests(interests: list[str], top_k: int = 10) -> None:
    """
    Recommend phones based on use-case interests.

    Uses proper link prediction: predicts (phone, suitableFor, interest) triples.
    The model learned suitableFor from user behavior data.
    """
    model = load_model()
    phone_names = load_phones()
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

    # Get all phone IDs (format: instance/phone/brand_model)
    all_phone_ids = [e for e in entity_to_id.keys() if e.startswith("instance/phone/")]

    if not all_phone_ids:
        print("No phones found in model")
        return

    suitable_rel_idx = relation_to_id["suitableFor"]
    aggregate_scores: dict[str, float] = {pid: 0.0 for pid in all_phone_ids}

    phone_indices = torch.tensor([entity_to_id[pid] for pid in all_phone_ids], dtype=torch.long)

    # For each interest, score all phones on (phone, suitableFor, interest)
    for interest in valid_interests:
        interest_idx = entity_to_id[interest]

        h = phone_indices
        r = torch.full((len(phone_indices),), suitable_rel_idx, dtype=torch.long)
        t = torch.full((len(phone_indices),), interest_idx, dtype=torch.long)

        hrt_batch = torch.stack([h, r, t], dim=1)
        scores = model.score_hrt(hrt_batch).squeeze()

        for phone_id, score in zip(all_phone_ids, scores):
            aggregate_scores[phone_id] += score.item()

    sorted_phones = sorted(aggregate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\nTop {top_k} phones for interests: {', '.join(interests)}")
    print("-" * 80)

    for rank, (phone_id, score) in enumerate(sorted_phones, 1):
        phone_name = phone_names.get(phone_id, phone_id)
        print(f"{rank:2d}. {phone_name} (score: {score:.4f})")


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
