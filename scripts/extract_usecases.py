#!/usr/bin/env python3
"""
Extract sentiment tags from reviews using Ollama LLM (zero-shot).

Two-pass approach per review:
1. Filter: Is this review useful? (contains actual phone feedback)
2. Extract: If yes, extract sentiment tags

Usage:
    uv run scripts/extract_usecases.py [--model MODEL] [--limit N] [--dry-run]
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
import requests
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

REVIEWS_PATH = Path("data/reviews.json")
OUTPUT_PATH = Path("knowledge_graph/review_scores.ttl")

OLLAMA_URL = "http://localhost:11434/api/generate"

# Normalize LLM output to canonical tags
TAG_NORMALIZATION: dict[str, str] = {
    # Camera
    "camera": "Camera", "cam": "Camera", "photo": "Camera", "photos": "Camera",
    "video": "Camera", "lens": "Camera", "photography": "Camera",
    # Display
    "display": "Display", "screen": "Display", "resolution": "Display",
    "refresh": "Display", "oled": "Display", "amoled": "Display", "lcd": "Display",
    # Battery
    "battery": "Battery", "charging": "Battery", "charge": "Battery",
    "batterylife": "Battery",
    # Performance
    "performance": "Performance", "speed": "Performance", "fast": "Performance",
    "slow": "Performance", "lag": "Performance", "processor": "Performance",
    "cpu": "Performance", "chip": "Performance", "smooth": "Performance",
    # Storage
    "storage": "Storage", "memory": "Storage", "space": "Storage", "gb": "Storage",
    # Build
    "build": "Build", "design": "Build", "quality": "Build", "durability": "Build",
    "premium": "Build", "plastic": "Build", "glass": "Build", "metal": "Build",
    # OS
    "os": "OS", "android": "OS", "ios": "OS", "software": "OS", "update": "OS",
    "updates": "OS", "ui": "OS",
    # Gaming
    "gaming": "Gaming", "games": "Gaming", "game": "Gaming",
    # General sentiment
    "good": "Overall", "bad": "Overall", "great": "Overall", "terrible": "Overall",
    "amazing": "Overall", "awful": "Overall", "excellent": "Overall", "poor": "Overall",
    "love": "Overall", "hate": "Overall", "best": "Overall", "worst": "Overall",
    "overall": "Overall", "general": "Overall",
    # Price/Value
    "price": "Value", "value": "Value", "expensive": "Value", "cheap": "Value",
    "worth": "Value", "money": "Value",
}

CANONICAL_TAGS = ["Camera", "Display", "Battery", "Performance", "Storage", "Build",
                  "OS", "Gaming", "Overall", "Value"]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_reviews() -> dict[str, list[str]]:
    """Load reviews and group them by phone_id."""
    with open(REVIEWS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    grouped: dict[str, list[str]] = defaultdict(list)
    for r in raw:
        text = r.get("text")
        if text:
            grouped[r["phone_id"]].append(text)

    return grouped

# =============================================================================
# OLLAMA
# =============================================================================

def query_ollama(prompt: str, model: str, max_tokens: int = 200) -> str:
    """Send prompt to Ollama and return raw response."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.RequestException as e:
        print(f"      Ollama error: {e}")
        return ""

# =============================================================================
# PASS 1: FILTER
# =============================================================================

def build_filter_prompt(review: str) -> str:
    return f"""
Does this text contain ANY opinion or feedback about the phone?

Text: "{review}"

Look for ANY of these - even just one phrase counts:
- "good", "bad", "nice", "terrible", "amazing", "disappointing"
- Comments about camera, battery, screen, performance, build, price
- "revolutionary", "changed everything", "cute phone", "can't use it"

Ignore questions, usernames, dates - focus on whether there's ANY opinion.

Answer YES if ANY useful opinion exists. Answer NO only if it's purely jokes/memes/off-topic.

Answer: YES or NO
""".strip()


def is_review_useful(review: str, model: str) -> bool:
    """Pass 1: Check if review is worth extracting from."""
    prompt = build_filter_prompt(review)
    response = query_ollama(prompt, model, max_tokens=10).strip().upper()
    return response.startswith("YES")

# =============================================================================
# PASS 2: EXTRACT
# =============================================================================

def build_extract_prompt(review: str) -> str:
    return f"""
Extract sentiment from this smartphone review.

Review: "{review}"

Output aspects mentioned with + (positive) or - (negative).
Common aspects: Camera, Display, Battery, Performance, Storage, Build, OS, Gaming, Value, Overall

Format - one per line:
<Aspect>: +
<Aspect>: -

Examples:
Camera: +
Battery: -
Overall: +

Rules:
- Only output aspects clearly mentioned with sentiment
- Use "Overall" for general good/bad sentiment
- No explanations, just the tags
""".strip()


def normalize_tag(tag: str) -> str | None:
    """Normalize a tag to canonical form."""
    tag_lower = tag.lower().strip()

    if tag_lower in TAG_NORMALIZATION:
        return TAG_NORMALIZATION[tag_lower]

    for canonical in CANONICAL_TAGS:
        if tag_lower == canonical.lower():
            return canonical

    return tag.title()


def extract_tags(review: str, model: str) -> dict[str, str]:
    """Pass 2: Extract sentiment tags from review."""
    prompt = build_extract_prompt(review)
    response = query_ollama(prompt, model, max_tokens=100)

    mentions: dict[str, str] = {}
    for line in response.splitlines():
        line = line.strip()
        match = re.match(r"([A-Za-z]+):\s*([+-])", line)
        if match:
            raw_tag = match.group(1)
            sign = match.group(2)
            normalized = normalize_tag(raw_tag)
            if normalized:
                mentions[normalized] = sign

    return mentions

# =============================================================================
# TTL GENERATION
# =============================================================================

def generate_ttl(phone_counts: dict[str, dict[str, tuple[int, int]]]) -> str:
    """Generate TTL output."""
    lines = [
        "@prefix sp: <http://example.org/smartphone#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "# Phone sentiment counts extracted from reviews using LLM (two-pass)",
        "",
    ]

    for phone_id, counts in sorted(phone_counts.items()):
        if not counts:
            continue

        has_data = False
        phone_lines = [f"# {phone_id}"]

        for tag in sorted(counts.keys()):
            pos, neg = counts[tag]
            if pos > 0:
                phone_lines.append(f'sp:{phone_id} sp:{tag.lower()}Positive "{pos}"^^xsd:integer .')
                has_data = True
            if neg > 0:
                phone_lines.append(f'sp:{phone_id} sp:{tag.lower()}Negative "{neg}"^^xsd:integer .')
                has_data = True

        if has_data:
            lines.extend(phone_lines)
            lines.append("")

    return "\n".join(lines)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def classify_phone(
    phone_id: str,
    reviews: list[str],
    model: str,
    verbose: bool = False,
) -> dict[str, tuple[int, int]]:
    """Classify a phone using two-pass per review."""
    positive_counts: dict[str, int] = defaultdict(int)
    negative_counts: dict[str, int] = defaultdict(int)

    useful_count = 0
    skipped_count = 0

    review_iter = list(enumerate(reviews, 1))
    if not verbose:
        review_iter = tqdm(review_iter, desc=f"  {phone_id}", leave=False)

    for i, review in review_iter:
        if verbose:
            print(f"\n    [{i}/{len(reviews)}] {review}")

        # Pass 1: Filter
        useful = is_review_useful(review, model)

        if not useful:
            skipped_count += 1
            if verbose:
                print(f"      -> SKIP (not useful)")
            continue

        useful_count += 1

        # Pass 2: Extract
        mentions = extract_tags(review, model)

        if verbose:
            print(f"      -> {mentions if mentions else 'no tags'}")

        for tag, sign in mentions.items():
            if sign == '+':
                positive_counts[tag] += 1
            else:
                negative_counts[tag] += 1

    if verbose:
        print(f"\n    Summary: {useful_count} useful, {skipped_count} skipped")

    # Combine into tuples
    result: dict[str, tuple[int, int]] = {}
    all_tags = set(positive_counts.keys()) | set(negative_counts.keys())
    for tag in all_tags:
        result[tag] = (positive_counts[tag], negative_counts[tag])

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, help="Limit number of phones to process")
    parser.add_argument("--dry-run", action="store_true", help="Verbose output, don't write file")
    args = parser.parse_args()

    reviews_by_phone = load_reviews()
    phones = list(reviews_by_phone.items())

    if args.limit:
        phones = phones[:args.limit]

    print(f"Processing {len(phones)} phones with model {args.model}")
    print(f"Two-pass: 1) filter useful reviews, 2) extract tags\n")

    phone_counts: dict[str, dict[str, tuple[int, int]]] = {}
    total_useful = 0
    total_skipped = 0

    phone_iter = phones
    if not args.dry_run:
        phone_iter = tqdm(phones, desc="Phones")

    for phone_id, reviews in phone_iter:
        if args.dry_run:
            print(f"\n{phone_id} ({len(reviews)} reviews)")
        phone_counts[phone_id] = classify_phone(
            phone_id=phone_id,
            reviews=reviews,
            model=args.model,
            verbose=args.dry_run,
        )

        # Count stats
        useful = sum(p + n for p, n in phone_counts[phone_id].values())
        skipped = len(reviews) - useful
        total_useful += useful
        total_skipped += skipped

        if args.dry_run:
            summary = {tag: f"+{p}/-{n}" for tag, (p, n) in phone_counts[phone_id].items()}
            print(f"    FINAL -> {summary if summary else 'none'}\n")

    ttl = generate_ttl(phone_counts)

    if args.dry_run:
        print("\n--- TTL OUTPUT ---\n")
        print(ttl)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(ttl, encoding="utf-8")
        print(f"\nWritten output to {args.output}")

    # Summary
    print("\n--- Tag Totals ---")
    total_pos: dict[str, int] = defaultdict(int)
    total_neg: dict[str, int] = defaultdict(int)
    for counts in phone_counts.values():
        for tag, (p, n) in counts.items():
            total_pos[tag] += p
            total_neg[tag] += n

    all_tags = set(total_pos.keys()) | set(total_neg.keys())
    for tag in sorted(all_tags):
        print(f"  {tag}: +{total_pos[tag]} / -{total_neg[tag]}")

# =============================================================================

if __name__ == "__main__":
    main()
