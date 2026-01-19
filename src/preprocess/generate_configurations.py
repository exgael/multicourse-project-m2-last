import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict


class PhoneInput(BaseModel):
    """Input phone from phones.json."""
    model_config = ConfigDict(extra="ignore")

    phone_id: str
    brand: str | None = None
    phone_name: str | None = None
    year: int | None = None
    display_type: str | None = None
    screen_size_inches: float | None = None
    refresh_rate_hz: int | None = None
    chipset: str | None = None
    main_camera_mp: int | None = None
    selfie_camera_mp: int | None = None
    battery_mah: int | None = None
    supports_5g: bool | None = None
    nfc: bool | None = None


class VariantInput(BaseModel):
    """Input variant from variants.json."""
    model_config = ConfigDict(extra="ignore")

    variant_id: str
    phone_id: str
    storage_gb: int | None = None
    ram_gb: int | None = None


class PhoneConfiguration(BaseModel):
    """Output phone configuration entity."""
    phone_id: str  # variant_id - used as PhoneConfiguration ID
    base_phone_id: str  # original phone_id - used as BasePhone ID
    brand: str | None = None
    phone_name: str | None = None
    year: int | None = None
    display_type: str | None = None
    screen_size_inches: float | None = None
    refresh_rate_hz: int | None = None
    chipset: str | None = None
    main_camera_mp: int | None = None
    selfie_camera_mp: int | None = None
    battery_mah: int | None = None
    supports_5g: bool | None = None
    nfc: bool | None = None
    storage_gb: int | None = None
    ram_gb: int | None = None


def generate_configurations(
    phones_file: Path,
    variants_file: Path,
    store_prices_file: Path,
    output_file: Path
) -> None:
    """Generate phone configurations from phones, variants, and store prices."""

    # Load source data
    with open(phones_file, "r", encoding="utf-8") as f:
        phones: list[PhoneInput] = [PhoneInput.model_validate(p) for p in json.load(f)]

    with open(variants_file, "r", encoding="utf-8") as f:
        variants: list[VariantInput] = [VariantInput.model_validate(v) for v in json.load(f)]

    # Extract variant IDs that have at least one store price
    variants_with_prices: set[str] = set()
    if store_prices_file.exists():
        with open(store_prices_file, "r", encoding="utf-8") as f:
            store_prices = json.load(f)
            variants_with_prices = {p["variant_id"] for p in store_prices}
    else:
        print(f"Warning: {store_prices_file} not found, no variants will be included")

    # Group variants by phone_id
    variants_by_phone: dict[str, list[VariantInput]] = {}
    for v in variants:
        if v.phone_id not in variants_by_phone:
            variants_by_phone[v.phone_id] = []
        variants_by_phone[v.phone_id].append(v)

    # Generate configurations (only for variants with prices)
    configurations: list[PhoneConfiguration] = []
    phones_with_variants = 0
    variants_included = 0
    variants_excluded = 0
    phones_without_variants = 0

    for phone in phones:
        phone_variants = variants_by_phone.get(phone.phone_id, [])

        if phone_variants:
            phones_with_variants += 1
            for v in phone_variants:
                if v.variant_id not in variants_with_prices:
                    variants_excluded += 1
                    continue

                variants_included += 1
                configurations.append(PhoneConfiguration(
                    phone_id=v.variant_id,
                    base_phone_id=phone.phone_id,
                    brand=phone.brand,
                    phone_name=phone.phone_name,
                    year=phone.year,
                    display_type=phone.display_type,
                    screen_size_inches=phone.screen_size_inches,
                    refresh_rate_hz=phone.refresh_rate_hz,
                    chipset=phone.chipset,
                    main_camera_mp=phone.main_camera_mp,
                    selfie_camera_mp=phone.selfie_camera_mp,
                    battery_mah=phone.battery_mah,
                    supports_5g=phone.supports_5g,
                    nfc=phone.nfc,
                    storage_gb=v.storage_gb,
                    ram_gb=v.ram_gb
                ))
        else:
            phones_without_variants += 1

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in configurations], f, indent=2)

    print(f"Generated {len(configurations)} phone configurations")
    print(f"  - {phones_with_variants} base phones had variants")
    print(f"  - {variants_included} variants with store prices (included)")
    print(f"  - {variants_excluded} variants without store prices (excluded)")
    print(f"  - {phones_without_variants} phones without variants (skipped)")
    print(f"Saved to {output_file}")
