from pathlib import Path
import json
import re
from pydantic import BaseModel
from typing import Any


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


class PriceEntry(BaseModel):
    price_id: str
    variant_id: str
    phone_id: str
    currency: str = ""
    value: float | None = None
    store: str = ""
    raw: str = ""

    def store_id(self) -> str:
        """Generate a URL-safe store identifier from store name."""
        return re.sub(r'[^a-z0-9]+', '_', self.store.lower()).strip('_')

    def to_eur(self) -> float | None:
        if self.currency and self.value is not None:
            rate: float | None = CURRENCY_TO_EUR.get(self.currency)
            if rate:
                return round(self.value * rate, 2)
        elif not self.currency and self.raw:
            parsed: tuple[str, float] | None = self._parse_raw()
            if parsed:
                curr, val = parsed
                rate: float | None = CURRENCY_TO_EUR.get(curr)
                if rate:
                    return round(val * rate, 2)
        return None

    def _parse_raw(self) -> tuple[str, float] | None:
        for symbol, currency in CURRENCY_SYMBOLS.items():
            if symbol in self.raw:
                price_str: str = re.sub(r'[^\d.]', '', self.raw)
                try:
                    return currency, float(price_str)
                except ValueError:
                    continue
        return None


class StorePriceEntry(BaseModel):
    """Individual price entry with store information for RDF mapping."""
    price_id: str
    variant_id: str
    store: str
    store_id: str
    price_eur: float


def process_prices(prices_file: Path, eur_prices_file: Path, store_prices_file: Path | None = None) -> dict[str, float]:
    with open(prices_file, "r") as f:
        prices_data: list[Any] = json.load(f)

    prices: list[PriceEntry] = [PriceEntry(**entry) for entry in prices_data]

    # Extract EUR prices and store prices
    eur_prices: dict[str, list[float]] = {}
    store_prices: list[StorePriceEntry] = []

    for price in prices:
        eur_value: float | None = price.to_eur()
        if eur_value:
            if price.variant_id not in eur_prices:
                eur_prices[price.variant_id] = []
            eur_prices[price.variant_id].append(eur_value)

            # Collect individual store prices
            if price.store:
                store_prices.append(StorePriceEntry(
                    price_id=price.price_id,
                    variant_id=price.variant_id,
                    store=price.store,
                    store_id=price.store_id(),
                    price_eur=eur_value,
                ))

    # Average prices for each variant
    variant_avg_prices: dict[str, float] = {
        variant_id: round(sum(vals) / len(vals), 2)
        for variant_id, vals in eur_prices.items()
    }

    # Write averaged prices output file (backward compatible)
    eur_prices_file.parent.mkdir(parents=True, exist_ok=True)
    with open(eur_prices_file, "w") as f:
        json.dump(variant_avg_prices, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(prices)} price entries -> {len(variant_avg_prices)} variants with EUR prices")
    print(f"Output: {eur_prices_file}")

    # Write store prices output file (new)
    if store_prices_file:
        store_prices_file.parent.mkdir(parents=True, exist_ok=True)
        with open(store_prices_file, "w") as f:
            json.dump([sp.model_dump() for sp in store_prices], f, indent=2, ensure_ascii=False)
        print(f"Output: {store_prices_file} ({len(store_prices)} store price entries)")

    return variant_avg_prices
