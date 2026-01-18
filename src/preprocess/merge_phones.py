import json
from pathlib import Path
from pydantic import BaseModel, Field


class PhoneInput(BaseModel):
    """Input phone from phones.json."""
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

    class Config:
        extra = "ignore"


class VariantInput(BaseModel):
    """Input variant from variants.json."""
    variant_id: str
    phone_id: str
    storage_gb: int | None = None
    ram_gb: int | None = None

    class Config:
        extra = "ignore"


class MergedPhone(BaseModel):
    """Output merged phone entity."""
    phone_id: str = Field(description="Unique ID (variant_id if has variant, else original phone_id)")
    base_phone_id: str = Field(description="Original phone_id reference")
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
    price_eur: float | None = None

def merge_phones(phones_file: Path, variants_file: Path, prices_file: Path, output_file: Path) -> None:
    """Merge phones, variants, and prices into flat phone entities."""

    # Load and validate source data
    with open(phones_file, "r", encoding="utf-8") as f:
        phones: list[PhoneInput] = [PhoneInput.model_validate(p) for p in json.load(f)]

    with open(variants_file, "r", encoding="utf-8") as f:
        variants: list[VariantInput] = [VariantInput.model_validate(v) for v in json.load(f)]

    if prices_file.exists():
        with open(prices_file, "r", encoding="utf-8") as f:
            prices: dict[str, float] = json.load(f)
    else:
        print(f"Warning: {prices_file} not found, prices will be null")
        prices = {}

    # Group variants by phone_id
    variants_by_phone: dict[str, list[VariantInput]] = {}
    for v in variants:
        if v.phone_id not in variants_by_phone:
            variants_by_phone[v.phone_id] = []
        variants_by_phone[v.phone_id].append(v)

    # Merge
    merged: list[MergedPhone] = []
    phones_with_variants = 0
    phones_without_variants = 0

    for phone in phones:
        phone_variants: list[VariantInput] = variants_by_phone.get(phone.phone_id, [])

        if phone_variants:
            phones_with_variants += 1
            for v in phone_variants:
                merged.append(MergedPhone(
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
                    ram_gb=v.ram_gb,
                    price_eur=prices.get(v.variant_id),
                ))
        else:
            phones_without_variants += 1
            merged.append(MergedPhone(
                phone_id=phone.phone_id,
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
            ))

    # Save merged data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in merged], f, indent=2)

    print(f"Merged {len(phones)} phones into {len(merged)} entries")
    print(f"  - {phones_with_variants} phones had variants")
    print(f"  - {phones_without_variants} phones without variants")
    print(f"Saved to {output_file}")
