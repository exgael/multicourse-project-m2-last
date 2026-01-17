from pathlib import Path
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
import torch
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "users"
MODEL_DIR = ROOT_DIR / "output" / "models" / "link_prediction"


def load_test_triples() -> TriplesFactory:
    phone_usecase = DATA_DIR / "phone_usecase_labels.csv"
    user_usecase = DATA_DIR / "user_usecase_labels.csv"
    user_phone = DATA_DIR / "user_phone_labels.csv"

    all_triples: list[tuple[str, str, str]] = []

    with open(phone_usecase, "r") as f:
        next(f)
        for line in f:
            phone_id, _, usecase = line.strip().split(",")
            all_triples.append((phone_id, "suitableFor", usecase))

    with open(user_usecase, "r") as f:
        next(f)
        for line in f:
            user_id, _, usecase = line.strip().split(",")
            all_triples.append((user_id, "interestedIn", usecase))

    with open(user_phone, "r") as f:
        next(f)
        for line in f:
            user_id, _, phone_id = line.strip().split(",")
            all_triples.append((user_id, "likes", phone_id))

    tf = TriplesFactory.from_labeled_triples(np.array(all_triples))
    _, testing, _ = tf.split([0.8, 0.1, 0.1], random_state=42)

    return testing


def evaluate_model() -> None:
    model = torch.load(MODEL_DIR / "trained_model.pkl", weights_only=False)
    testing = load_test_triples()
    tf = TriplesFactory.from_path_binary(MODEL_DIR / "training_triples")

    evaluator = RankBasedEvaluator(filtered=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,
        batch_size=256,
        additional_filter_triples=[tf.mapped_triples]
    )

    print("LINK PREDICTION EVALUATION")
    print()
    print(f"Test set: {testing.num_triples} triples")
    print()
    print("Metrics:")
    print(f"  MRR: {results.get_metric('mean_reciprocal_rank'):.4f}")
    print(f"  MR:  {results.get_metric('mean_rank'):.2f}")
    print(f"  Hits@1:  {results.get_metric('hits_at_1'):.4f}")
    print(f"  Hits@3:  {results.get_metric('hits_at_3'):.4f}")
    print(f"  Hits@5:  {results.get_metric('hits_at_5'):.4f}")
    print(f"  Hits@10: {results.get_metric('hits_at_10'):.4f}")
    print()

    relation_results = results.to_df()
    print("Per-Relation Metrics:")
    print(relation_results.to_string(index=False))


if __name__ == "__main__":
    evaluate_model()
