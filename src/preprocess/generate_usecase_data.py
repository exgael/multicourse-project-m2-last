from pathlib import Path
import json
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class UseCaseDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    desc: str = Field(description="Human-readable description")
    skos_uri: str = Field(description="SKOS URI from knowledge_graph/schema/skos.ttl")
    rules: Callable[[dict[str, Any]], bool] = Field(description="Classification function")


class SyntheticUser(BaseModel):
    user_id: str
    usecase: str
    price_segment: str


class DatasetStats(BaseModel):
    num_users: int
    num_phones: int
    num_usecases: int
    num_price_segments: int
    user_usecase_labels: int
    user_phone_labels: int
    usecase_price_matrix: dict[str, dict[str, int]]


# Functional use-cases
USE_CASES: dict[str, UseCaseDefinition] = {
    "Gaming": UseCaseDefinition(
        desc="Mobile gaming with smooth experience",
        skos_uri="spv:Gaming",
        rules=lambda p: (
            (p.get("refresh_rate_hz") or 60) >= 144 and
            (p.get("battery_mah") or 0) >= 4500
        )
    ),
    "EverydayUse": UseCaseDefinition(
        desc="General daily usage - modern phone with NFC and 5G",
        skos_uri="spv:EverydayUse",
        rules=lambda p: (
            p.get("nfc") is True and
            p.get("supports_5g") is True and
            (p.get("battery_mah") or 0) >= 4500
        )
    ),
    "Minimalist": UseCaseDefinition(
        desc="Simple phones without modern connectivity",
        skos_uri="spv:Minimalist",
        rules=lambda p: (
            p.get("nfc") is not True and
            p.get("supports_5g") is not True
        )
    ),
}

# Price segments
PRICE_SEGMENTS: dict[str, UseCaseDefinition] = {
    "Flagship": UseCaseDefinition(
        desc="Premium segment, higher prices",
        skos_uri="spv:Flagship",
        rules=lambda p: (p.get("price_eur") or 0) > 900
    ),
    "MidRange": UseCaseDefinition(
        desc="Middle segment, average prices",
        skos_uri="spv:MidRange",
        rules=lambda p: 400 <= (p.get("price_eur") or 0) <= 900
    ),
    "Budget": UseCaseDefinition(
        desc="Entry-level segment, lower prices",
        skos_uri="spv:Budget",
        rules=lambda p: 0 < (p.get("price_eur") or 0) < 400
    ),
}


def load_phones(merged_phones_file: Path) -> list[dict[str, Any]]:
    """Load merged phones (already includes storage/RAM/price)."""
    print(f"Loading merged phones from {merged_phones_file}...")
    with open(merged_phones_file, "r", encoding="utf-8") as f:
        phones: list[dict[str, Any]] = json.load(f)
    print(f"Loaded {len(phones)} phones")
    return phones


def classify_phone_usecases(phone: dict[str, Any]) -> list[str]:
    usecases: list[str] = []
    for usecase_name, usecase_def in USE_CASES.items():
        if usecase_def.rules(phone):
            usecases.append(usecase_name)
    return usecases


def get_price_segment(phone: dict[str, Any]) -> str:
    """Get the price segment for a phone."""
    for seg_name, seg_def in PRICE_SEGMENTS.items():
        if seg_def.rules(phone):
            return seg_name
    return "Budget"


def generate_synthetic_users(
    num_users: int,
    valid_combinations: list[tuple[str, str]],
) -> list[SyntheticUser]:
    """Generate users by randomly picking from valid (usecase, price_segment) combinations."""
    users: list[SyntheticUser] = []

    for i in range(num_users):
        usecase, price_seg = random.choice(valid_combinations)
        user_id = f"user_{i:04d}"

        users.append(SyntheticUser(
            user_id=user_id,
            usecase=usecase,
            price_segment=price_seg,
        ))

    return users


def create_training_labels(
    phones: list[dict[str, Any]],
    users: list[SyntheticUser],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build cross-tabulation: (usecase, price_segment) -> list of phone_ids
    usecase_segment_phones: dict[tuple[str, str], list[str]] = defaultdict(list)

    # Build the matrix for stats
    usecase_price_matrix: dict[str, dict[str, int]] = {
        uc: {seg: 0 for seg in PRICE_SEGMENTS} for uc in USE_CASES
    }

    for phone in phones:
        phone_id: str = phone["phone_id"]
        usecases = classify_phone_usecases(phone)
        price_seg = get_price_segment(phone)

        for uc in usecases:
            usecase_segment_phones[(uc, price_seg)].append(phone_id)
            usecase_price_matrix[uc][price_seg] += 1

    # Print cross-tabulation
    print("\nUse-Case x Price Segment Matrix:")
    header = f"{'Use-Case':<15} | " + " | ".join(f"{seg:>10}" for seg in PRICE_SEGMENTS)
    print(header)
    print("-" * len(header))
    for uc in USE_CASES:
        row = usecase_price_matrix[uc]
        values = " | ".join(f"{row[seg]:>10}" for seg in PRICE_SEGMENTS)
        print(f"{uc:<15} | {values}")

    # Write user interest labels
    user_usecase_file = output_dir / "user_usecase_labels.csv"
    with open(user_usecase_file, "w", encoding="utf-8") as f:
        f.write("user_id,relation,usecase\n")
        for user in users:
            f.write(f"{user.user_id},interestedIn,{user.usecase}\n")
            f.write(f"{user.user_id},interestedIn,{user.price_segment}\n")

    print(f"\nCreated {len(users) * 2} user→usecase labels")

    # Write user-phone labels
    user_phone_file = output_dir / "user_phone_labels.csv"
    positive_labels = 0

    with open(user_phone_file, "w", encoding="utf-8") as f:
        f.write("user_id,relation,phone_id\n")
        for user in users:
            matching_phones = usecase_segment_phones[(user.usecase, user.price_segment)]

            if not matching_phones:
                continue

            # Random sampling: 1-3 phones per user
            num_samples = min(len(matching_phones), random.randint(1, 3))
            sampled_phones = random.sample(matching_phones, num_samples)

            for phone_id in sampled_phones:
                f.write(f"{user.user_id},likes,{phone_id}\n")
                positive_labels += 1

    print(f"Created {positive_labels} user→phone labels")

    # Save stats
    stats = DatasetStats(
        num_users=len(users),
        num_phones=len(phones),
        num_usecases=len(USE_CASES),
        num_price_segments=len(PRICE_SEGMENTS),
        user_usecase_labels=len(users) * 2,
        user_phone_labels=positive_labels,
        usecase_price_matrix=usecase_price_matrix,
    )

    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats.model_dump(), f, indent=2)

    print(f"\nDataset statistics saved to {stats_file}")
    print_summary(stats)


def print_summary(stats: DatasetStats) -> None:
    print(f"\nUsers: {stats.num_users}")
    print(f"Phones: {stats.num_phones}")
    print(f"Use-cases: {stats.num_usecases}")
    print(f"Price segments: {stats.num_price_segments}")
    print(f"\nLabels:")
    print(f"  User→UseCase: {stats.user_usecase_labels}")
    print(f"  User→Phone: {stats.user_phone_labels}")


def generate_users(
    merged_phones_file: Path,
    output_dir: Path,
    num_users: int = 500,
    random_seed: int = 42,
) -> None:
    # If data already exists, skip generation
    if (output_dir / "users").exists() and any((output_dir / "users").iterdir()):
        print(f"Data already exists in {output_dir}, skipping generation.")
        return

    random.seed(random_seed)
    phones: list[dict[str, Any]] = load_phones(merged_phones_file)

    # Find valid (usecase, price_segment) combinations (non-empty)
    valid_combinations: list[tuple[str, str]] = []
    for uc in USE_CASES:
        for seg in PRICE_SEGMENTS:
            # Check if this combination has phones
            count = sum(
                1 for p in phones
                if uc in classify_phone_usecases(p) and get_price_segment(p) == seg
            )
            if count > 0:
                valid_combinations.append((uc, seg))

    print(f"Valid combinations: {valid_combinations}")

    users: list[SyntheticUser] = generate_synthetic_users(num_users, valid_combinations)
    create_training_labels(phones, users, output_dir)
