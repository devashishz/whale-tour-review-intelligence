import os

import polars as pl
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

load_dotenv()

# Initialize the modern Gemini Client
# It automatically picks up the GEMINI_API_KEY environment variable
client = genai.Client()

# -------------------------------------------------------------
# 1. Define Strict Pydantic Output Schema
# -------------------------------------------------------------
class ReviewExtraction(BaseModel):
    has_captain_mention: bool = Field(
        description="True if an individual is explicitly identified as the boat captain, skipper, or tour guide leader. False otherwise."
    )
    captains: list[str] = Field(
        default_factory=list,
        description="Standardized first names or full names of boat captains mentioned (e.g., 'Mike', 'Dave', 'Pete'). Do NOT include deckhands, general office staff, or company names."
    )
    sentiment: str = Field(
        description="Overall sentiment regarding the captain's service: 'Positive', 'Neutral', or 'Negative'."
    )


# -------------------------------------------------------------
# 2. Extraction Function with Gemini
# -------------------------------------------------------------
def extract_entities_from_review(review_text: str) -> ReviewExtraction:
    """Passes review text to Gemini and enforces Pydantic structured output."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=review_text,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert NLP entity extraction pipeline. Your task is to identify "
                    "boat captains or skippers named in whale watching tour reviews. "
                    "Standardize all captain names to Title Case. If no captain is explicitly mentioned, "
                    "set has_captain_mention to false and return an empty list."
                ),
                response_mime_type="application/json",
                response_schema=ReviewExtraction,
                temperature=0.0,  # Deterministic results
            ),
        )
        
        # When response_schema is passed, response.parsed automatically returns the Pydantic model
        return response.parsed

    except (errors.APIError, ValidationError) as e:
        print(f"Error parsing review: {e}")
        return ReviewExtraction(has_captain_mention=False, captains=[], sentiment="Unknown")


# -------------------------------------------------------------
# 3. Batch Processing Pipeline
# -------------------------------------------------------------
def process_reviews(input_jsonl_path: str, output_parquet_path: str):
    if not os.path.exists(input_jsonl_path):
        raise FileNotFoundError(f"Cannot find input file: {input_jsonl_path}")

    # Load scraped dataset
    df = pl.read_ndjson(input_jsonl_path)
    print(f"Loaded {len(df)} reviews from {input_jsonl_path}...")

    reviews = df.to_dicts()
    extracted_records = []

    print("Running Gemini Entity Extraction...")
    for row in tqdm(reviews, desc="Extracting Captains"):
        text = row.get("text", "")
        if not text or len(text.strip()) < 10:
            continue

        result = extract_entities_from_review(text)

        # Unpack each identified captain into an individual row
        if result and result.has_captain_mention and result.captains:
            for cap_name in result.captains:
                extracted_records.append({
                    "place_name": row.get("place_name"),
                    "review_id": row.get("review_id"),
                    "published_at": row.get("published_at"),
                    "stars": row.get("stars"),
                    "captain_name": cap_name.strip(),
                    "sentiment": result.sentiment,
                })
        else:
            # Reviews where no captain was mentioned
            extracted_records.append({
                "place_name": row.get("place_name"),
                "review_id": row.get("review_id"),
                "published_at": row.get("published_at"),
                "stars": row.get("stars"),
                "captain_name": "None Mentioned",
                "sentiment": result.sentiment if result else "Unknown",
            })

    # Convert back to Polars DataFrame
    output_df = pl.DataFrame(extracted_records)

    # Save to Parquet and JSONL for downstream ranking aggregations
    output_df.write_parquet(output_parquet_path)
    output_df.write_ndjson("data/enriched_reviews.jsonl")

    print(f"\nExtraction complete! Saved {len(output_df)} rows to {output_parquet_path}")
    print(output_df.head(5))


if __name__ == "__main__":
    process_reviews(
        input_jsonl_path="data/scraped_reviews.jsonl",
        output_parquet_path="data/enriched_reviews.parquet",
    )