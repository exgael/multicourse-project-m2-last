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


class DatasetStats(BaseModel):
    num_users: int
    num_phones: int
    num_usecases: int
    user_usecase_labels: int
    user_phone_labels: int
    usecase_phone_counts: dict[str, int]


# Functional use-cases
USE_CASES: dict[str, UseCaseDefinition] = {
    "Gaming": UseCaseDefinition(
        desc="Mobile gaming with smooth experience",
        skos_uri="spv:Gaming",
        rules=lambda p: (
            (p.get("refresh_rate_hz") or 60) >= 144 and
            (p.get("battery_mah") or 0) >= 4500 and 
            (p.get("ram_gb") or 0) >= 16 and
            (p.get("storage_gb") or 0) >= 256 and 
            (p.get("screen_size_inches") or 0) >= 6.5 and
            p.get("supports_5g") is True

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
            p.get("supports_5g") is not True and  
            (p.get("screen_size_inches") or 0) <= 4
        )
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


def generate_synthetic_users(
    num_users: int,
    valid_usecases: list[str],
) -> list[SyntheticUser]:
    """Generate users by randomly picking from valid usecases."""
    users: list[SyntheticUser] = []

    for i in range(num_users):
        usecase = random.choice(valid_usecases)
        user_id = f"user_{i:04d}"

        users.append(SyntheticUser(
            user_id=user_id,
            usecase=usecase,
        ))

    return users


def create_training_labels(
    phones: list[dict[str, Any]],
    users: list[SyntheticUser],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build usecase -> list of phone_ids mapping
    usecase_phones: dict[str, list[str]] = defaultdict(list)

    for phone in phones:
        phone_id: str = phone["phone_id"]
        usecases = classify_phone_usecases(phone)

        for uc in usecases:
            usecase_phones[uc].append(phone_id)

    # Print usecase phone counts
    print("\nUse-Case Phone Counts:")
    print("-" * 30)
    for uc in USE_CASES:
        count = len(usecase_phones[uc])
        print(f"  {uc:<15}: {count:>5} phones")

    # Write user interest labels (only usecase, no price segment)
    user_usecase_file = output_dir / "user_usecase_labels.csv"
    with open(user_usecase_file, "w", encoding="utf-8") as f:
        f.write("user_id,relation,usecase\n")
        for user in users:
            f.write(f"{user.user_id},interestedIn,{user.usecase}\n")

    print(f"\nCreated {len(users)} user→usecase labels")

    # Write user-phone labels
    user_phone_file = output_dir / "user_phone_labels.csv"
    positive_labels = 0

    with open(user_phone_file, "w", encoding="utf-8") as f:
        f.write("user_id,relation,phone_id\n")
        for user in users:
            matching_phones = usecase_phones[user.usecase]

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
    usecase_phone_counts = {uc: len(usecase_phones[uc]) for uc in USE_CASES}

    stats = DatasetStats(
        num_users=len(users),
        num_phones=len(phones),
        num_usecases=len(USE_CASES),
        user_usecase_labels=len(users),
        user_phone_labels=positive_labels,
        usecase_phone_counts=usecase_phone_counts,
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

    # Find valid usecases (those with at least one phone)
    valid_usecases: list[str] = []
    for uc in USE_CASES:
        count = sum(1 for p in phones if uc in classify_phone_usecases(p))
        if count > 0:
            valid_usecases.append(uc)

    print(f"Valid usecases: {valid_usecases}")

    users: list[SyntheticUser] = generate_synthetic_users(num_users, valid_usecases)
    create_training_labels(phones, users, output_dir)
