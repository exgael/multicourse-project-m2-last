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
    # Photography hierarchy
    "CasualPhotography": UseCaseDefinition(
        desc="Casual photo taking with good quality camera",
        skos_uri="spv:CasualPhotography",
        rules=lambda p: (
            (p.get("main_camera_mp") or 0) >= 50 and
            ("AMOLED" in (p.get("display_type") or "") or "OLED" in (p.get("display_type") or ""))
        )
    ),
    "ProPhotography": UseCaseDefinition(
        desc="Professional-grade photography with high resolution camera",
        skos_uri="spv:ProPhotography",
        rules=lambda p: (
            (p.get("main_camera_mp") or 0) >= 100 and
            ("AMOLED" in (p.get("display_type") or "") or "OLED" in (p.get("display_type") or ""))
        )
    ),
    # Gaming hierarchy
    "CasualGaming": UseCaseDefinition(
        desc="Casual mobile gaming with smooth experience",
        skos_uri="spv:CasualGaming",
        rules=lambda p: (
            (p.get("refresh_rate_hz") or 60) >= 90 and
            (p.get("battery_mah") or 0) >= 4000
        )
    ),
    "ProGaming": UseCaseDefinition(
        desc="Competitive mobile gaming with high performance",
        skos_uri="spv:ProGaming",
        rules=lambda p: (
            (p.get("refresh_rate_hz") or 60) >= 120 and
            (p.get("screen_size_inches") or 0) >= 6.5 and
            (p.get("battery_mah") or 0) >= 5000 and
            (p.get("_max_storage_gb") or 0) >= 512 and
            (p.get("_max_ram_gb") or 0) >= 16
        )
    ),
    # Other use cases
    "Business": UseCaseDefinition(
        desc="Work and productivity phone",
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
            (p.get("year") or 2025) >= 2020 and
            (p.get("battery_mah") or 0) >= 3500 and
            (p.get("main_camera_mp") or 0) >= 12 and
            (p.get("selfie_camera_mp") or 0) >= 20
        )
    ),
    # Price segments
    "Flagship": UseCaseDefinition(
        desc="Premium segment, higher prices",
        skos_uri="spv:Flagship",
        rules=lambda p: p.get("_base_price_eur", 0) > 900
    ),
    "MidRange": UseCaseDefinition(
        desc="Middle segment, average prices",
        skos_uri="spv:MidRange",
        rules=lambda p: 400 <= p.get("_base_price_eur", 0) <= 900
    ),
    "Budget": UseCaseDefinition(
        desc="Entry-level segment, lower prices",
        skos_uri="spv:Budget",
        rules=lambda p: 0 < p.get("_base_price_eur", 0) < 400
    ),
    "AfterMarket": UseCaseDefinition(
        desc="Discontinued phones, no retail price",
        skos_uri="spv:AfterMarket",
        rules=lambda p: p.get("_base_price_eur", 0) == 0
    ),
    # Additional use cases for expanded dataset
    "Vlogging": UseCaseDefinition(
        desc="Video blogging and selfie-focused content creation",
        skos_uri="spv:Vlogging",
        rules=lambda p: (p.get("selfie_camera_mp") or 0) >= 48
    ),
    "VintageCollector": UseCaseDefinition(
        desc="Interest in classic and vintage phones",
        skos_uri="spv:VintageCollector",
        rules=lambda p: (
            (p.get("year") or 2025) < 2015 and
            p.get("nfc") is not True and
            p.get("supports_5g") is not True
        )
    ),
    "BasicPhone": UseCaseDefinition(
        desc="Simple phones for basic calling and texting",
        skos_uri="spv:BasicPhone",
        rules=lambda p: (
            (p.get("year") or 2025) < 2020 and
            (p.get("_base_price_eur") or 0) == 0
        )
    ),
}

USER_PERSONAS: list[UserPersona] = [
    # Photography-focused personas
    UserPersona(name_prefix="casual_photographer", primary="CasualPhotography", price_segment="MidRange"),
    UserPersona(name_prefix="pro_photographer", primary="ProPhotography", price_segment="Flagship"),
    # Gaming-focused personas
    UserPersona(name_prefix="casual_gamer", primary="CasualGaming", price_segment="MidRange"),
    UserPersona(name_prefix="pro_gamer", primary="ProGaming", price_segment="Flagship"),
    # Business and productivity personas
    UserPersona(name_prefix="business", primary="Business", price_segment="MidRange"),
    UserPersona(name_prefix="professional", primary="Business", price_segment="Flagship"),
    # Everyday users
    UserPersona(name_prefix="casual", primary="EverydayUse", price_segment="Budget"),
    UserPersona(name_prefix="student", primary="EverydayUse", price_segment="MidRange"),
    UserPersona(name_prefix="minimalist", primary="EverydayUse", price_segment="Budget"),
    # Power users and creators
    UserPersona(name_prefix="creator", primary="ProPhotography", price_segment="MidRange"),
    UserPersona(name_prefix="traveler", primary="CasualPhotography", price_segment="Budget"),
    UserPersona(name_prefix="poweruser", primary="ProGaming", price_segment="MidRange"),
    # Vlogging personas (selfie-focused)
    UserPersona(name_prefix="vlogger", primary="Vlogging", price_segment="MidRange"),
    UserPersona(name_prefix="influencer", primary="Vlogging", price_segment="Flagship"),
    # Vintage/retro phone enthusiasts
    UserPersona(name_prefix="collector", primary="VintageCollector", price_segment="AfterMarket"),
    UserPersona(name_prefix="retro_lover", primary="BasicPhone", price_segment="AfterMarket"),
    # Basic phone users (seniors, minimalists who want simple)
    UserPersona(name_prefix="senior", primary="BasicPhone", price_segment="AfterMarket"),
    UserPersona(name_prefix="budget_seeker", primary="EverydayUse", price_segment="AfterMarket"),
]


def load_phone_prices(eur_prices_file: Path) -> dict[str, float]:
    print("Loading EUR prices...")
    with open(eur_prices_file, "r", encoding="utf-8") as f:
        variant_prices: dict[str, float] = json.load(f)

    print(f"Loaded prices for {len(variant_prices)} variants")
    return variant_prices


def load_variants(variants_file: Path) -> dict[str, list[dict[str, Any]]]:
    """Load variants and group by phone_id."""
    print("Loading variants...")
    with open(variants_file, "r", encoding="utf-8") as f:
        variants: list[dict[str, Any]] = json.load(f)

    phone_variants: dict[str, list[dict[str, Any]]] = {}
    for v in variants:
        phone_id = v.get("phone_id")
        if phone_id:
            if phone_id not in phone_variants:
                phone_variants[phone_id] = []
            phone_variants[phone_id].append(v)

    print(f"Loaded {len(variants)} variants for {len(phone_variants)} phones")
    return phone_variants


def load_phones(phones_file: Path, variants_file: Path, eur_prices_file: Path) -> list[dict[str, Any]]:
    print("Loading phones...")
    with open(phones_file, "r", encoding="utf-8") as f:
        phones: list[dict[str, Any]] = json.load(f)

    variant_prices = load_phone_prices(eur_prices_file)
    phone_variants = load_variants(variants_file)

    for phone in phones:
        phone_id = phone["phone_id"]
        variants = phone_variants.get(phone_id, [])

        # Get max storage and RAM from variants
        phone["_max_storage_gb"] = max((v.get("storage_gb") or 0 for v in variants), default=0)
        phone["_max_ram_gb"] = max((v.get("ram_gb") or 0 for v in variants), default=0)

        # Get base (minimum) price from variants
        prices = [variant_prices.get(v["variant_id"], 0) for v in variants if variant_prices.get(v["variant_id"])]
        phone["_base_price_eur"] = min(prices) if prices else 0

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
        phones_file: Path,
        variants_file: Path,
        eur_prices_file: Path,
        output_dir: Path,
        num_users: int = 500,
        random_seed: int = 42,
    ) -> None:
    # If data already exists, skip generation
    if (output_dir / "users").exists() and any((output_dir / "users").iterdir()):
        print(f"Data already exists in {output_dir}, skipping generation.")
        return

    random.seed(random_seed)
    phones: list[dict[str, Any]] = load_phones(phones_file, variants_file, eur_prices_file)
    users: list[SyntheticUser] = generate_synthetic_users(num_users)
    create_training_labels(phones, users, output_dir)
