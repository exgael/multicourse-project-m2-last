import json
import re
from pathlib import Path
from typing import Any, Optional
import requests
from tqdm import tqdm
from pydantic import BaseModel

REVIEWS_PATH = Path("data/reviews.json")
OUTPUT_PATH = Path("data/review_tags.json")
OLLAMA_URL = "http://localhost:11434/api/generate"

TAG_NORMALIZATION: dict[str, str] = {
    "camera": "Camera", "cam": "Camera", "photo": "Camera", "photos": "Camera",
    "video": "Camera", "lens": "Camera", "photography": "Camera",
    "display": "Display", "screen": "Display", "resolution": "Display",
    "refresh": "Display", "oled": "Display", "amoled": "Display", "lcd": "Display",
    "battery": "Battery", "charging": "Battery", "charge": "Battery",
    "batterylife": "Battery",
    "performance": "Performance", "speed": "Performance", "fast": "Performance",
    "slow": "Performance", "lag": "Performance", "processor": "Performance",
    "cpu": "Performance", "chip": "Performance", "smooth": "Performance",
    "storage": "Storage", "memory": "Storage", "space": "Storage", "gb": "Storage",
    "build": "Build", "design": "Build", "quality": "Build", "durability": "Build",
    "premium": "Build", "plastic": "Build", "glass": "Build", "metal": "Build",
    "os": "OS", "android": "OS", "ios": "OS", "software": "OS", "update": "OS",
    "updates": "OS", "ui": "OS",
    "gaming": "Gaming", "games": "Gaming", "game": "Gaming",
    "good": "Overall", "bad": "Overall", "great": "Overall", "terrible": "Overall",
    "amazing": "Overall", "awful": "Overall", "excellent": "Overall", "poor": "Overall",
    "love": "Overall", "hate": "Overall", "best": "Overall", "worst": "Overall",
    "overall": "Overall", "general": "Overall",
    "price": "Value", "value": "Value", "expensive": "Value", "cheap": "Value",
    "worth": "Value", "money": "Value",
}

CANONICAL_TAGS = ["Camera", "Display", "Battery", "Performance", "Storage", "Build",
                  "OS", "Gaming", "Overall", "Value"]


class ReviewInput(BaseModel):
    review_id: str
    phone_id: str
    text: str
    user: Optional[str] = None
    user_id: Optional[str] = None
    date: Optional[str] = None


class ReviewResult(BaseModel):
    review_id: str
    phone_id: str
    is_interesting: bool
    positive_tags: list[str]
    negative_tags: list[str]


def load_reviews(path: Path) -> list[ReviewInput]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ReviewInput(**r) for r in raw if r.get("text")]


def query_ollama(prompt: str, model: str, max_tokens: int = 200) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        },
        timeout=120,
    )
    r.raise_for_status()
    result = r.json()
    response = result.get("response", "").strip()

    if not response and "thinking" in result:
        thinking = result.get("thinking", "").strip()
        if thinking:
            lines = [l.strip() for l in thinking.split('\n') if l.strip()]
            response = lines[-1] if lines else thinking

    return response


def is_review_interesting(text: str, model: str) -> bool:
    prompt = f"""Text: "{text}"

Does this review contain actual phone feedback or opinions? Answer YES or NO only.

YES if it mentions: camera, battery, display, performance, build quality, price/value, or contains sentiment (good/bad/love/hate).
NO if it's only a question, joke, or off-topic.

Answer:""".strip()

    response = query_ollama(prompt, model, max_tokens=200).strip().upper()
    return response.startswith("YES") or "YES" in response


def extract_tags(text: str, model: str) -> dict[str, str]:
    prompt = f"""Extract sentiment from this smartphone review.

Review: "{text}"

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
- No explanations, just the tags""".strip()

    response = query_ollama(prompt, model, max_tokens=300)
    mentions: dict[str, str] = {}

    for line in response.splitlines():
        line = line.strip()
        match = re.match(r"([A-Za-z]+):\s*([+-])", line)
        if match:
            raw_tag = match.group(1)
            sign = match.group(2)
            tag_lower = raw_tag.lower().strip()

            if tag_lower in TAG_NORMALIZATION:
                normalized = TAG_NORMALIZATION[tag_lower]
            elif any(tag_lower == t.lower() for t in CANONICAL_TAGS):
                normalized = next(t for t in CANONICAL_TAGS if t.lower() == tag_lower)
            else:
                normalized = raw_tag.title()

            if normalized:
                mentions[normalized] = sign

    return mentions


def process_review(review: ReviewInput, model: str) -> ReviewResult:
    is_interesting: bool = is_review_interesting(review.text, model)
    positive_tags: list[str] = []
    negative_tags: list[str] = []

    if is_interesting:
        mentions: dict[str, str] = extract_tags(review.text, model)
        for tag, sign in mentions.items():
            if sign == '+':
                positive_tags.append(tag)
            else:
                negative_tags.append(tag)

    return ReviewResult(
        review_id=review.review_id,
        phone_id=review.phone_id,
        is_interesting=is_interesting,
        positive_tags=positive_tags,
        negative_tags=negative_tags,
    )


def analyse_reviews(
    input_path: Path = REVIEWS_PATH,
    output_path: Path = OUTPUT_PATH,
    model: str = "qwen2.5:latest",
    limit: Optional[int] = None,
    force: bool = False,
) -> None:
    """Analyze reviews with LLM to extract sentiment tags."""

    # If review tag file exists, skip processing
    if output_path.exists() and not force:
        print(f"Review tags file already exists: {output_path}")
        return

    reviews: list[ReviewInput] = load_reviews(input_path)
    if limit:
        reviews = reviews[:limit]

    results: list[ReviewResult] = []
    for review in tqdm(reviews, desc="Analyzing reviews"):
        result: ReviewResult = process_review(review, model)
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        results_dict: list[dict[str, Any]] = [r.model_dump() for r in results]
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
