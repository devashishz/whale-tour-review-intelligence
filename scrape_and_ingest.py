import os
from pathlib import Path

import polars as pl
from apify_client import ApifyClient
from dotenv import load_dotenv

# 1. Environment & Path Configuration
load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

if not APIFY_TOKEN:
    raise ValueError("Error: APIFY_API_TOKEN is missing. Please set it in your .env file.")

# Ensure destination directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "scraped_reviews.jsonl"

# 2. Target Settings
TARGET_URLS = [
    "https://www.google.com/maps/place/Alaska+Galore+Tours/@58.3753,-134.6291,15z",
    "https://www.google.com/maps/place/Harv+and+Marv's+Whale+Watching/@58.3742,-134.6312,15z",
]

RUN_INPUT = {
    "searchStringsArray": [
        "Alaska Galore Tours Juneau Alaska",
        "Harv and Marv's Whale Watching Juneau Alaska"
    ],
    "maxReviews": 50,
    "reviewsSort": "newest",
    "language": "en",
    "personalData": False,
}

def run_pipeline():
    client = ApifyClient(APIFY_TOKEN)
    
    print("Initiating cloud scraping task on Apify...")
    run = client.actor("compass/google-maps-reviews-scraper").call(run_input=RUN_INPUT)
    
    dataset_id = run.get("defaultDatasetId")
    print(f"Extraction job finished. Fetching dataset: {dataset_id}")
    
    # 3. Stream raw payload into memory
    raw_records = []
    for item in client.dataset(dataset_id).iterate_items():
        raw_records.append({
            "place_name": item.get("title", "Unknown"),
            "review_id": item.get("reviewId"),
            "published_at": item.get("publishedAtDate"),
            "stars": item.get("stars"),
            "text": item.get("text", "")
        })
        
    if not raw_records:
        print("Warning: No records were returned by the scraper.")
        return

    # 4. Data Processing with Polars
    print("Normalizing schema and applying filters via Polars...")
    df = pl.DataFrame(raw_records)

    cleaned_df = (
        df.with_columns(
            # Remove trailing 'Z' or timezone offsets to allow clean ISO datetime parsing
            pl.col("published_at")
            .str.slice(0, 19)
            .str.to_datetime("%Y-%m-%dT%H:%M:%S")
            .alias("published_at"),
            pl.col("text").str.strip_chars().alias("text")
        )
        # Filter: Only reviews from 2023 onward with non-empty review text
        .filter(pl.col("published_at") >= pl.datetime(2023, 1, 1))
        .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
    )

    # 5. Export for downstream LLM parsing
    cleaned_df.write_ndjson(OUTPUT_FILE)
    
    print(f"Successfully processed and stored {len(cleaned_df)} reviews to {OUTPUT_FILE}")
    print(cleaned_df.select(["place_name", "published_at", "stars", "text"]).head(3))

if __name__ == "__main__":
    run_pipeline()