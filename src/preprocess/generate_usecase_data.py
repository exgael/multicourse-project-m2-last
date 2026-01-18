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


class UserPersona(BaseModel):
    name_prefix: str = Field(description="Prefix for user ID generation")
    primary: str = Field(description="Primary use-case interest")
    secondary: list[str] = Field(description="Secondary use-case interests")


class SyntheticUser(BaseModel):
    user_id: str
    interests: list[str]
    persona: str


class DatasetStats(BaseModel):
    num_users: int
    num_phones: int
    num_usecases: int
    phone_usecase_labels: int
    user_usecase_labels: int
    user_phone_labels: int
    usecase_distribution: dict[str, int]
    persona_distribution: dict[str, int]

USE_CASES: dict[str, UseCaseDefinition] = {
    "Photography": UseCaseDefinition(
        desc="Taking photos and videos",
        skos_uri="spv:Photography",
        rules=lambda p: (
            (p.get("main_camera_mp") or 0) >= 48 and
            ("AMOLED" in (p.get("display_type") or "") or "OLED" in (p.get("display_type") or ""))
        )
    ),
    "Gaming": UseCaseDefinition(
        desc="Playing mobile games",
        skos_uri="spv:Gaming",
        rules=lambda p: (
            ((p.get("main_camera_mp") or 0) >= 40 or "gaming" in (p.get("chipset") or "").lower()) and
            (p.get("battery_mah") or 0) >= 4000 and
            (p.get("refresh_rate_hz") or 60) >= 90
        )
    ),
    "Business": UseCaseDefinition(
        desc="Work phone",
        skos_uri="spv:Business",
        rules=lambda p: (
            (p.get("battery_mah") or 0) >= 5000 and
            (p.get("main_camera_mp") or 0) >= 48
        )
    ),
    "EverydayUse": UseCaseDefinition(
        desc="General daily usage",
        skos_uri="spv:EverydayUse",
        rules=lambda p: (
            (p.get("year") or 2025) >= 2024 and
            (p.get("battery_mah") or 0) >= 3500 and
            (p.get("main_camera_mp") or 0) >= 12
        )
    ),
    "Flagship": UseCaseDefinition(
        desc="Premium segment, higher prices",
        skos_uri="spv:Flagship",
        rules=lambda p: p.get("_base_price_eur", 0) > 1200
    ),
    "MidRange": UseCaseDefinition(
        desc="Middle segment, average prices",
        skos_uri="spv:MidRange",
        rules=lambda p: 600 <= p.get("_base_price_eur", 0) <= 1200
    ),
    "Budget": UseCaseDefinition(
        desc="Entry-level segment, lower prices",
        skos_uri="spv:Budget",
        rules=lambda p: 0 < p.get("_base_price_eur", 0) < 600
    ),
}

USER_PERSONAS: list[UserPersona] = [
    UserPersona(name_prefix="photographer", primary="Photography", secondary=["Flagship"]),
    UserPersona(name_prefix="gamer", primary="Gaming", secondary=["Flagship"]),
    UserPersona(name_prefix="business", primary="Business", secondary=["MidRange"]),
    UserPersona(name_prefix="casual", primary="EverydayUse", secondary=["Budget"]),
    UserPersona(name_prefix="professional", primary="Business", secondary=["Photography", "Flagship"]),
    UserPersona(name_prefix="student", primary="EverydayUse", secondary=["Budget", "Photography"]),
    UserPersona(name_prefix="creator", primary="Photography", secondary=["Gaming", "Flagship"]),
    UserPersona(name_prefix="traveler", primary="Photography", secondary=["EverydayUse", "MidRange"]),
    UserPersona(name_prefix="poweruser", primary="Gaming", secondary=["Flagship", "Business"]),
    UserPersona(name_prefix="minimalist", primary="EverydayUse", secondary=["Budget"]),
]


def load_phone_prices(eur_prices_file: Path) -> dict[str, float]:
    print("Loading EUR prices...")
    with open(eur_prices_file, "r", encoding="utf-8") as f:
        phone_prices: dict[str, float] = json.load(f)

    print(f"Loaded prices for {len(phone_prices)} phones")
    return phone_prices


def load_phones(phones_file: Path, eur_prices_file: Path) -> list[dict[str, Any]]:
    print("Loading phones...")
    with open(phones_file, "r", encoding="utf-8") as f:
        phones: list[dict[str, Any]] = json.load(f)

    phone_prices = load_phone_prices(eur_prices_file)
    for phone in phones:
        phone["_base_price_eur"] = phone_prices.get(phone["phone_id"], 0)

    print(f"Loaded {len(phones)} phones")
    return phones


def classify_phone_usecases(phone: dict[str, Any]) -> list[str]:
    usecases: list[str] = []

    for usecase_name, usecase_def in USE_CASES.items():
        if usecase_def.rules(phone):
            usecases.append(usecase_name)

    return usecases


def generate_synthetic_users(num_users: int, secondary_interest_probability: float) -> list[SyntheticUser]:
    users: list[SyntheticUser] = []

    for i in range(num_users):
        persona = random.choice(USER_PERSONAS)
        user_id = f"{persona.name_prefix}_{i:04d}"
        interests = [persona.primary]

        for secondary in persona.secondary:
            if random.random() < secondary_interest_probability:
                interests.append(secondary)

        users.append(SyntheticUser(
            user_id=user_id,
            interests=interests,
            persona=persona.name_prefix
        ))

    return users


def create_training_labels(
    phones: list[dict[str, Any]],
    users: list[SyntheticUser],
    output_dir: Path,
    min_user_phone_samples: int = 1,
    max_user_phone_samples: int = 5
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify phones by use-case ONLY for matching users to phones
    # Phone specs will be exported from KG as triples for PyKEEN training
    phone_usecases: dict[str, list[str]] = {}
    usecase_phones: dict[str, list[str]] = defaultdict(list)

    for phone in phones:
        phone_id: str = phone["phone_id"]
        usecases = classify_phone_usecases(phone)
        phone_usecases[phone_id] = usecases

        for usecase in usecases:
            usecase_phones[usecase].append(phone_id)

    user_usecase_file = output_dir / "user_usecase_labels.csv"
    with open(user_usecase_file, "w", encoding="utf-8") as f:
        f.write("user_id,relation,usecase\n")
        for user in users:
            for interest in user.interests:
                f.write(f"{user.user_id},interestedIn,{interest}\n")

    print(f"Created {sum(len(u.interests) for u in users)} user→usecase labels")

    user_phone_file = output_dir / "user_phone_labels.csv"
    positive_labels = 0

    with open(user_phone_file, "w", encoding="utf-8") as f:
        f.write("user_id,relation,phone_id\n")
        for user in users:
            matching_phones: set[str] = set()
            for interest in user.interests:
                matching_phones.update(usecase_phones[interest])

            num_samples = min(
                len(matching_phones),
                random.randint(min_user_phone_samples, max_user_phone_samples)
            )
            sampled_phones = random.sample(list(matching_phones), num_samples)

            for phone_id in sampled_phones:
                f.write(f"{user.user_id},likes,{phone_id}\n")
                positive_labels += 1

    print(f"Created {positive_labels} user→phone labels")

    stats_file = output_dir / "dataset_stats.json"
    stats = DatasetStats(
        num_users=len(users),
        num_phones=len(phones),
        num_usecases=len(USE_CASES),
        phone_usecase_labels=0,  # No longer generated
        user_usecase_labels=sum(len(u.interests) for u in users),
        user_phone_labels=positive_labels,
        usecase_distribution={
            uc: len(phones_list) for uc, phones_list in usecase_phones.items()
        },
        persona_distribution={
            persona.name_prefix: sum(1 for u in users if u.persona == persona.name_prefix)
            for persona in USER_PERSONAS
        }
    )

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats.model_dump(), f, indent=2)

    print(f"\nDataset statistics saved to {stats_file}")
    print_summary(stats)


def print_summary(stats: DatasetStats) -> None:
    print(f"Users: {stats.num_users}")
    print(f"Phones: {stats.num_phones}")
    print(f"Use-cases: {stats.num_usecases}")
    print(f"\nLabels:")
    print(f"  User→UseCase: {stats.user_usecase_labels}")
    print(f"  User→Phone: {stats.user_phone_labels}")
    print(f"\nUse-case Distribution (for matching only):")
    for uc, count in sorted(stats.usecase_distribution.items(), key=lambda x: -x[1]):
        print(f"  {uc}: {count} phones")
    print(f"\nPersona Distribution:")
    for persona, count in sorted(stats.persona_distribution.items(), key=lambda x: -x[1]):
        print(f"  {persona}: {count} users")


def generate_users(
        phones_file: Path,
        eur_prices_file: Path,
        output_dir: Path,
        num_users: int = 500, 
        random_seed: int = 42, 
        secondary_interest_probability: float = 0.5,
        min_user_phone_samples: int = 1,
        max_user_phone_samples: int = 5
    ) -> None:
    # If data already exists, skip generation
    if (output_dir / "users").exists() and any((output_dir / "users").iterdir()):
        print(f"Data already exists in {output_dir}, skipping generation.")
        return

    random.seed(random_seed)
    phones: list[dict[str, Any]] = load_phones(phones_file, eur_prices_file)
    users: list[SyntheticUser] = generate_synthetic_users(num_users, secondary_interest_probability)
    create_training_labels(
        phones, 
        users, 
        output_dir,
        min_user_phone_samples,
        max_user_phone_samples
    )
