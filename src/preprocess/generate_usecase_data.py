from pathlib import Path
import json
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

    # "phone_id": "apple_apple_iphone_11_64gb_4gb",
    # "base_phone_id": "apple_apple_iphone_11",
    # "brand": "Apple",
    # "phone_name": "Apple iPhone 11",
    # "year": 2019,
    # "display_type": "Liquid Retina IPS LCD, 625 nits (typ)",
    # "screen_size_inches": 6.1,
    # "refresh_rate_hz": null,
    # "chipset": "Apple A13 Bionic (7 nm+)",
    # "main_camera_mp": 12,
    # "selfie_camera_mp": 12,
    # "battery_mah": 3110,
    # "supports_5g": false,
    # "nfc": true,
    # "storage_gb": 64,
    # "ram_gb": 4,
    # "price_eur": 171.45

class UseCaseDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    desc: str = Field(description="Human-readable description")
    skos_uri: str = Field(description="SKOS URI from knowledge_graph/schema/skos.ttl")
    rules: Callable[[dict[str, Any]], bool] = Field(description="Classification function")


class UserPersona(BaseModel):
    name_prefix: str = Field(description="Prefix for user ID generation")
    primary: str = Field(description="Primary use-case interest")
    price_segment: str = Field(description="Price segment interest")


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
    # Photography
    "Photography": UseCaseDefinition(
        desc="Taking photos and videos with good quality camera",
        skos_uri="spv:Photography",
        rules=lambda p: (
            (p.get("main_camera_mp") or 0) >= 100 and
            ("AMOLED" in (p.get("display_type") or "") or "OLED" in (p.get("display_type") or "")) and
            (p.get("storage_gb") or 0) >= 512 
        )
    ),
    # Additional use cases for expanded dataset
    "Vlogging": UseCaseDefinition(
        desc="Video blogging and selfie-focused content creation",
        skos_uri="spv:Vlogging",
        rules=lambda p: (p.get("selfie_camera_mp") or 0) >= 48 and
            (p.get("storage_gb") or 0) >= 256
    ),
    # Gaming
    "Gaming": UseCaseDefinition(
        desc="Mobile gaming with smooth experience",
        skos_uri="spv:Gaming",
        rules=lambda p: (
            (p.get("refresh_rate_hz") or 60) >= 144 and
            (p.get("battery_mah") or 0) >= 5000 and
            (p.get("ram_gb") or 0) >= 16 and
            (p.get("storage_gb") or 0) >= 256
        )
    ),
    # Other use cases
    "Business": UseCaseDefinition(
        desc="Work and productivity phone",
        skos_uri="spv:Business",
        rules=lambda p: (
            (p.get("year") or 2025) >= 2024 and
            (p.get("battery_mah") or 0) >= 7000 and
            (p.get("main_camera_mp") or 0) >= 48 and
            p.get("nfc") is True and
            p.get("supports_5g") is True
        )
    ),
    "EverydayUse": UseCaseDefinition(
        desc="General daily usage",
        skos_uri="spv:EverydayUse",
        rules=lambda p: (
            (p.get("year") or 2025) >= 2019 and
            (p.get("battery_mah") or 0) >= 3500 and
            (p.get("main_camera_mp") or 0) >= 20 and
            (p.get("selfie_camera_mp") or 0) >= 12  and
            p.get("nfc") is True and
            p.get("supports_5g") is True
        )
    ),
    # Price segments
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
    "AfterMarket": UseCaseDefinition(
        desc="Discontinued phones, no retail price",
        skos_uri="spv:AfterMarket",
        rules=lambda p: (p.get("price_eur") or 0) == 0
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

USER_PERSONAS: list[UserPersona] = [
    # Photography-focused personas (price segment differentiates casual vs pro)
    UserPersona(name_prefix="photographer", primary="Photography", price_segment="MidRange"),
    UserPersona(name_prefix="pro_photographer", primary="Photography", price_segment="Flagship"),
    # Gaming-focused personas (price segment differentiates casual vs pro)
    UserPersona(name_prefix="gamer", primary="Gaming", price_segment="MidRange"),
    UserPersona(name_prefix="pro_gamer", primary="Gaming", price_segment="Flagship"),
    # Business and productivity personas
    UserPersona(name_prefix="business", primary="Business", price_segment="MidRange"),
    UserPersona(name_prefix="professional", primary="Business", price_segment="Flagship"),
    # Everyday users
    UserPersona(name_prefix="casual", primary="EverydayUse", price_segment="Budget"),
    UserPersona(name_prefix="student", primary="EverydayUse", price_segment="MidRange"),
    # Power users and creators
    UserPersona(name_prefix="creator", primary="Photography", price_segment="MidRange"),
    UserPersona(name_prefix="traveler", primary="Photography", price_segment="Budget"),
    UserPersona(name_prefix="poweruser", primary="Gaming", price_segment="MidRange"),
    # Vlogging personas (selfie-focused)
    UserPersona(name_prefix="vlogger", primary="Vlogging", price_segment="MidRange"),
    UserPersona(name_prefix="influencer", primary="Vlogging", price_segment="Flagship"),
    # Minimalist phone users
    UserPersona(name_prefix="senior", primary="Minimalist", price_segment="AfterMarket"),
    UserPersona(name_prefix="simple_user", primary="Minimalist", price_segment="Budget"),
    UserPersona(name_prefix="budget_seeker", primary="EverydayUse", price_segment="AfterMarket"),
]


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


def generate_synthetic_users(num_users: int) -> list[SyntheticUser]:
    users: list[SyntheticUser] = []

    for i in range(num_users):
        persona = random.choice(USER_PERSONAS)
        user_id = f"{persona.name_prefix}_{i:04d}"
        interests = [persona.primary, persona.price_segment]

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

            # Gaussian: mean=2, std=2 (3σ reaches 8)
            raw_samples = int(random.gauss(2, 2))
            num_samples = max(1, min(len(matching_phones), raw_samples))
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
    users: list[SyntheticUser] = generate_synthetic_users(num_users)
    create_training_labels(phones, users, output_dir)
