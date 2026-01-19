from pydantic import BaseModel, ConfigDict, Field
import re
import json
from pathlib import Path
from collections import defaultdict

# RAW DATA MODELS INPUTS


class ReviewTagInput(BaseModel):
    """Input review tag entry."""
    model_config = ConfigDict(extra="ignore")

    review_id: str
    phone_id: str
    is_interesting: bool = False
    positive_tags: list[str] = Field(default_factory=list)
    negative_tags: list[str] = Field(default_factory=list)


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


class PriceEntry(BaseModel):
    price_id: str
    variant_id: str
    phone_id: str
    currency: str = ""
    value: float | None = None
    store: str = ""
    raw: str = ""

    CURRENCY_SYMBOLS: dict[str, str] = {
        "₹": "INR",
        "£": "GBP",
        "€": "EUR",
        "$": "USD",
    }

    CURRENCY_TO_EUR: dict[str, float] = {
        "EUR": 1.0,
        "GBP": 1.17,
        "USD": 0.92,
        "INR": 0.011,
        "CAD": 0.67,
    }

    def store_id(self) -> str:
        """Generate a URL-safe store identifier from store name."""
        return re.sub(r'[^a-z0-9]+', '_', self.store.lower()).strip('_')

    def to_eur(self) -> float | None:
        if self.currency and self.value is not None:
            rate: float | None = self.CURRENCY_TO_EUR.get(self.currency)
            if rate:
                return round(self.value * rate, 2)
        elif not self.currency and self.raw:
            parsed: tuple[str, float] | None = self._parse_raw()
            if parsed:
                curr, val = parsed
                rate: float | None = self.CURRENCY_TO_EUR.get(curr)
                if rate:
                    return round(val * rate, 2)
        return None

    def _parse_raw(self) -> tuple[str, float] | None:
        for symbol, currency in self.CURRENCY_SYMBOLS.items():
            if symbol in self.raw:
                price_str: str = re.sub(r'[^\d.]', '', self.raw)
                try:
                    return currency, float(price_str)
                except ValueError:
                    continue
        return None


def create_store_prices(src: Path, dst: Path | None = None) -> None:
    store_prices: list[dict[str, object]] = [
        {
            "price_id": p.price_id,
            "variant_id": p.variant_id,
            "store": p.store,
            "store_id": p.store_id(),
            "price_eur": eur,
        }
        for p in map(PriceEntry.model_validate, json.loads(src.read_text("utf-8")))
        if (eur := p.to_eur()) and p.store
    ]

    if dst:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(store_prices, indent=2, ensure_ascii=False), "utf-8")
        print(f"Output: {dst} ({len(store_prices)} store price entries)")


def create_review_sentiments(src: Path, dst: Path) -> None:
    agg = defaultdict(lambda: {"positive": 0, "negative": 0})

    for r in map(ReviewTagInput.model_validate, json.loads(src.read_text("utf-8"))):
        for t in r.positive_tags:
            agg[(r.phone_id, t)]["positive"] += 1
        for t in r.negative_tags:
            agg[(r.phone_id, t)]["negative"] += 1

    dst.parent.mkdir(parents=True, exist_ok=True)

    dst.write_text(json.dumps([
        {
            "phone_id": phone_id,
            "tag": tag,
            "positive": counts["positive"],
            "negative": counts["negative"],
        }
        for (phone_id, tag), counts in agg.items()
    ], indent=2), "utf-8")


def create_configurations(phones_f: Path, variants_f: Path, prices_f: Path, out_f: Path) -> None:
    phones = list(map(PhoneInput.model_validate, json.loads(phones_f.read_text("utf-8"))))
    variants = list(map(VariantInput.model_validate, json.loads(variants_f.read_text("utf-8"))))

    priced = {p["variant_id"] for p in json.loads(prices_f.read_text("utf-8"))} if prices_f.exists() else set()

    by_phone: dict[str, list[VariantInput]] = defaultdict(list)
    for v in variants:
        if v.variant_id in priced:
            by_phone[v.phone_id].append(v)

    configs: list[dict[str, object]] = [
        {
            "phone_id": str(v.variant_id),
            "base_phone_id": p.phone_id,
            "brand": p.brand,
            "phone_name": p.phone_name,
            "year": p.year,
            "display_type": p.display_type,
            "screen_size_inches": p.screen_size_inches,
            "refresh_rate_hz": p.refresh_rate_hz,
            "chipset": p.chipset,
            "main_camera_mp": p.main_camera_mp,
            "selfie_camera_mp": p.selfie_camera_mp,
            "battery_mah": p.battery_mah,
            "supports_5g": p.supports_5g,
            "nfc": p.nfc,
            "storage_gb": v.storage_gb,
            "ram_gb": v.ram_gb,
        }
        for p in phones
        for v in by_phone[p.phone_id]
    ]

    out_f.parent.mkdir(parents=True, exist_ok=True)
    out_f.write_text(json.dumps(configs, indent=2), "utf-8")
    print(f"Generated {len(configs)} configurations → {out_f}")


if __name__ == "__main__":
    DATASET_DIR = Path(__file__).parent.parent / "data"
    create_store_prices(
        src=DATASET_DIR / "raw_pretty" / "prices.json",
        dst=DATASET_DIR / "preprocessed" / "store_prices.json",
    )
    create_review_sentiments(
        src=DATASET_DIR / "raw_pretty" / "review_tags.json",
        dst=DATASET_DIR / "preprocessed" / "review_sentiments.json",
    )
    create_configurations(
        phones_f=DATASET_DIR / "raw_pretty" / "phones.json",
        variants_f=DATASET_DIR / "raw_pretty" / "variants.json",
        prices_f=DATASET_DIR / "preprocessed" / "store_prices.json",
        out_f=DATASET_DIR / "preprocessed" / "phones_configuration.json",
    )