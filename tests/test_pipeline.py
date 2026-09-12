import polars as pl
import pytest
from pydantic import ValidationError
from extract_entities import ReviewAnalysis

def test_pydantic_schema_validation():
    """Ensure ReviewAnalysis parses valid responses and rejects invalid ones."""
    payload = {
        "captains_mentioned": ["Mike", "Dave"],
        "sentiment": "Positive"
    }
    validated = ReviewAnalysis(**payload)
    assert validated.sentiment == "Positive"
    assert "Mike" in validated.captains_mentioned

    # Should raise validation error if sentiment is missing
    with pytest.raises(ValidationError):
        ReviewAnalysis(captains_mentioned=["Mike"])

def test_polars_date_parsing_and_ranking():
    """Ensure ranking logic and date parsing work correctly."""
    mock_data = [
        {"place_name": "A", "captain_name": "Mike", "published_at": "2024-06-15T12:00:00Z"},
        {"place_name": "A", "captain_name": "Mike", "published_at": "2024-06-16"},
        {"place_name": "A", "captain_name": "Dave", "published_at": "2024-06-17"},
    ]
    df = pl.DataFrame(mock_data)

    processed = (
        df.with_columns(
            pl.col("published_at")
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("parsed_date")
        )
        .with_columns(
            pl.col("parsed_date").dt.year().alias("year"),
            pl.col("parsed_date").dt.month().alias("month")
        )
    )

    ranking = (
        processed.group_by(["place_name", "captain_name"])
        .len(name="mentions")
        .with_columns(
            pl.col("mentions").rank(descending=True, method="min").alias("rank")
        )
        .sort("rank")
    )

    # Mike has 2 mentions, so he should be rank 1
    top_captain = ranking.filter(pl.col("rank") == 1)["captain_name"].to_list()
    assert "Mike" in top_captain