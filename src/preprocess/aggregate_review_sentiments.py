import json
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, Field


class ReviewTagInput(BaseModel):
    """Input review tag entry."""
    model_config = ConfigDict(extra="ignore")

    review_id: str
    phone_id: str
    is_interesting: bool = False
    positive_tags: list[str] = Field(default_factory=list)
    negative_tags: list[str] = Field(default_factory=list)


class TagSentimentFlat(BaseModel):
    """Flat sentiment entry for RML mapping."""
    phone_id: str
    tag: str
    positive: int
    negative: int


def aggregate_review_sentiments(
    review_tags_file: Path,
    output_file: Path
) -> None:
    """Aggregate review tags into flat per-phone/tag sentiment counts."""

    # Load review tags
    with open(review_tags_file, "r", encoding="utf-8") as f:
        reviews = [ReviewTagInput.model_validate(r) for r in json.load(f)]

    # Aggregate by (phone_id, tag)
    sentiments: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0}
    )

    for review in reviews:
        for tag in review.positive_tags:
            sentiments[(review.phone_id, tag)]["positive"] += 1

        for tag in review.negative_tags:
            sentiments[(review.phone_id, tag)]["negative"] += 1

    # Convert to flat output
    results: list[TagSentimentFlat] = [
        TagSentimentFlat(
            phone_id=phone_id,
            tag=tag,
            positive=counts["positive"],
            negative=counts["negative"]
        )
        for (phone_id, tag), counts in sentiments.items()
    ]

    # Sort for consistent output
    results.sort(key=lambda x: (x.phone_id, x.tag))

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    # Stats
    unique_phones = len(set(r.phone_id for r in results))
    unique_tags = len(set(r.tag for r in results))

    print(f"Aggregated {len(results)} sentiment entries")
    print(f"  - {unique_phones} phones")
    print(f"  - {unique_tags} unique tags")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent
    aggregate_review_sentiments(
        review_tags_file=base_path / "data" / "review_tags.json",
        output_file=base_path / "output" / "data" / "review_sentiments.json"
    )
