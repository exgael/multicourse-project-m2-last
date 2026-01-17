from pathlib import Path
import json
import re
from pydantic import BaseModel, Field


class PriceConfig(BaseModel):
    prices_file: Path
    eur_prices_file: Path
    currency_to_eur: dict[str, float] = Field(default={
        "EUR": 1.0,
        "GBP": 1.17,
        "USD": 0.92,
        "INR": 0.011,
    })
    currency_symbols: dict[str, str] = Field(default={
        "₹": "INR",
        "£": "GBP",
        "€": "EUR",
        "$": "USD",
    })


def parse_raw_price(raw: str, config: PriceConfig) -> tuple[str, float] | None:
    if not raw:
        return None

    for symbol, currency in config.currency_symbols.items():
        if symbol in raw:
            price_str = re.sub(r'[^\d.]', '', raw)
            try:
                value = float(price_str)
                return currency, value
            except ValueError:
                continue

    return None


def convert_to_eur(currency: str, value: float, config: PriceConfig) -> float | None:
    rate: float | None = config.currency_to_eur.get(currency)
    if rate is None:
        return None
    return value * rate


def process_prices(prices_file: Path, eur_prices_file: Path) -> dict[str, float]:
    config = PriceConfig(prices_file=prices_file, eur_prices_file=eur_prices_file)

    with open(config.prices_file, "r") as f:
        prices_data = json.load(f)

    eur_prices: dict[str, list[float]] = {}

    for price_entry in prices_data:
        phone_id = price_entry.get("phone_id")
        if not phone_id:
            continue

        currency = price_entry.get("currency", "")
        value = price_entry.get("value")
        raw = price_entry.get("raw", "")

        eur_value = None

        if currency and value is not None:
            eur_value = convert_to_eur(currency, value, config)
        elif currency == "" and raw:
            parsed = parse_raw_price(raw, config)
            if parsed:
                curr, val = parsed
                eur_value = convert_to_eur(curr, val, config)

        if eur_value:
            if phone_id not in eur_prices:
                eur_prices[phone_id] = []
            eur_prices[phone_id].append(round(eur_value, 2))

    phone_avg_prices = {phone_id: round(sum(prices) / len(prices), 2)
                        for phone_id, prices in eur_prices.items()}

    config.eur_prices_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config.eur_prices_file, "w") as f:
        json.dump(phone_avg_prices, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(prices_data)} price entries -> {len(phone_avg_prices)} phones with EUR prices")
    print(f"Output: {config.eur_prices_file}")

    return phone_avg_prices


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    process_prices(
        prices_file=ROOT_DIR / "data" / "prices.json",
        eur_prices_file=ROOT_DIR / "data" / "eur_prices.json"
    )
