import json
from pathlib import Path
from typing import Any

REVIEW_TAGS_PATH = Path("data/review_tags.json")
FLATTENED_POSITIVE_PATH = Path("data/review_tags_positive.json")
FLATTENED_NEGATIVE_PATH = Path("data/review_tags_negative.json")


def flatten_review_tags(
    input_path: Path = REVIEW_TAGS_PATH,
    output_positive: Path = FLATTENED_POSITIVE_PATH,
    output_negative: Path = FLATTENED_NEGATIVE_PATH
) -> None:
    """Flatten review_tags.json into separate positive/negative tag files for RML mapping."""
    with open(input_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    positive_tags: list[dict[str, Any]] = []
    negative_tags: list[dict[str, Any]] = []

    for review in reviews:
        review_id = review["review_id"]

        for tag in review.get("positive_tags", []):
            positive_tags.append({
                "review_id": review_id,
                "tag": tag
            })

        for tag in review.get("negative_tags", []):
            negative_tags.append({
                "review_id": review_id,
                "tag": tag
            })

    output_positive.parent.mkdir(parents=True, exist_ok=True)
    with open(output_positive, "w", encoding="utf-8") as f:
        json.dump(positive_tags, f, indent=2, ensure_ascii=False)

    output_negative.parent.mkdir(parents=True, exist_ok=True)
    with open(output_negative, "w", encoding="utf-8") as f:
        json.dump(negative_tags, f, indent=2, ensure_ascii=False)
