"""
Explore what the model understands about concepts.

Usage:
    uv run src/recommandation/explore.py ProGaming
    uv run src/recommandation/explore.py ProGaming Budget
    uv run src/recommandation/explore.py --list  # Show available concepts
"""
import argparse
import torch
from pathlib import Path
from pykeen.triples import TriplesFactory

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT_DIR / "output" / "models" / "link_prediction"


def load_model_and_data():
    model = torch.load(MODEL_DIR / "trained_model.pkl", weights_only=False)
    tf = TriplesFactory.from_path_binary(MODEL_DIR / "training_triples")
    return model, tf


def get_embedding(model, tf, entity_name: str):
    """Get embedding for an entity."""
    entity_to_id = tf.entity_to_id
    if entity_name not in entity_to_id:
        return None
    idx = entity_to_id[entity_name]
    embeddings = model.entity_representations[0](indices=None).detach()
    return embeddings[idx]


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return torch.dot(emb1 / emb1.norm(), emb2 / emb2.norm()).item()


def explore(keywords: list[str], top_k: int = 15) -> None:
    """Explore what the model understands about given keywords."""
    model, tf = load_model_and_data()
    entity_to_id = tf.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    # Validate keywords
    valid_keywords = []
    for kw in keywords:
        if kw in entity_to_id:
            valid_keywords.append(kw)
        else:
            print(f"Warning: '{kw}' not found in model")

    if not valid_keywords:
        print("No valid keywords found.")
        return

    # Get embeddings and compute average
    embeddings = model.entity_representations[0](indices=None).detach()
    keyword_embeddings = [embeddings[entity_to_id[kw]] for kw in valid_keywords]
    combined_emb = torch.stack(keyword_embeddings).mean(dim=0)
    combined_norm = combined_emb / combined_emb.norm()

    print("=" * 60)
    print(f"MODEL UNDERSTANDING: {' + '.join(valid_keywords)}")
    print("=" * 60)

    # Categorize entities
    specs = []
    phones = []
    usecases = []
    users = []

    all_norms = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(all_norms, combined_norm)

    for idx, sim in enumerate(similarities):
        entity = id_to_entity[idx]
        sim_val = sim.item()

        if entity in valid_keywords:
            continue

        if entity.startswith(("battery", "camera", "selfie", "refresh", "year_", "true", "false")):
            specs.append((entity, sim_val))
        elif entity.startswith("instance/phone/"):
            phones.append((entity.replace("instance/phone/", ""), sim_val))
        elif any(entity.startswith(p) for p in ["casual", "pro_", "business", "student", "creator",
                                                   "traveler", "poweruser", "minimalist", "vlogger",
                                                   "influencer", "collector", "retro_lover", "senior", "backup_phone"]):
            users.append((entity, sim_val))
        elif entity in ["ProGaming", "CasualGaming", "ProPhotography", "CasualPhotography",
                        "Business", "EverydayUse", "Flagship", "MidRange", "Budget",
                        "AfterMarket", "Vlogging", "VintageCollector", "BasicPhone"]:
            usecases.append((entity, sim_val))

    # Sort by similarity
    specs.sort(key=lambda x: -x[1])
    phones.sort(key=lambda x: -x[1])
    usecases.sort(key=lambda x: -x[1])
    users.sort(key=lambda x: -x[1])

    # Print specs
    print("\n📊 SPEC ASSOCIATIONS:")
    print("-" * 40)
    for spec, sim in specs:
        bar_len = int(abs(sim) * 40)
        if sim > 0:
            bar = "█" * bar_len
            print(f"  {sim:+.3f} {bar:40} {spec}")
        else:
            bar = "░" * bar_len
            print(f"  {sim:+.3f} {bar:40} {spec}")

    # Print related usecases
    print("\n🎯 RELATED USE CASES:")
    print("-" * 40)
    for uc, sim in usecases[:5]:
        print(f"  {sim:+.3f}  {uc}")

    # Print top phones
    print(f"\n📱 TOP {top_k} SIMILAR PHONES:")
    print("-" * 40)
    for i, (phone, sim) in enumerate(phones[:top_k], 1):
        print(f"  {i:2d}. {sim:+.3f}  {phone}")

    # Print sample users
    print(f"\n👤 SAMPLE SIMILAR USERS (top 5):")
    print("-" * 40)
    for user, sim in users[:5]:
        print(f"  {sim:+.3f}  {user}")


def list_concepts(tf) -> None:
    """List available concepts in the model."""
    entity_to_id = tf.entity_to_id

    usecases = ["ProGaming", "CasualGaming", "ProPhotography", "CasualPhotography",
                "Business", "EverydayUse", "Flagship", "MidRange", "Budget",
                "AfterMarket", "Vlogging", "VintageCollector", "BasicPhone"]

    specs = [e for e in entity_to_id.keys() if e.startswith(("battery", "camera", "selfie", "refresh", "year_", "true", "false"))]

    print("Available concepts to explore:")
    print()
    print("USE CASES:")
    for uc in usecases:
        if uc in entity_to_id:
            print(f"  - {uc}")
    print()
    print("SPECS:")
    for spec in sorted(specs):
        print(f"  - {spec}")


def main():
    parser = argparse.ArgumentParser(description="Explore model understanding of concepts")
    parser.add_argument("keywords", nargs="*", help="Keywords to explore (e.g., ProGaming Budget)")
    parser.add_argument("--top-k", type=int, default=15, help="Number of phones to show")
    parser.add_argument("--list", action="store_true", help="List available concepts")

    args = parser.parse_args()

    if args.list:
        _, tf = load_model_and_data()
        list_concepts(tf)
    elif args.keywords:
        explore(args.keywords, args.top_k)
    else:
        print("Provide keywords to explore or use --list to see available concepts")
        print("Example: uv run src/recommandation/explore.py ProGaming Budget")


if __name__ == "__main__":
    main()
