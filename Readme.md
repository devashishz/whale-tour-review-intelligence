# Whale Tour Review Intelligence & Captain Leaderboard

An end-to-end data and AI engineering pipeline that extracts named entities (tour captains) from unstructured Google/TripAdvisor reviews and builds multi-dimensional performance leaderboards using Polars window functions.

## Architecture

1. **Ingestion (`scrape_and_ingest.py`)**: Batches review extraction across multiple tour operators using managed APIs, filtering for active operating windows (2023–Present).
2. **AI Entity Extraction (`extract_entities.py`)**: Uses OpenAI structured outputs (`Pydantic`) to parse personnel names and sentiment out of unstructured review narratives.
3. **Analytics & Windowing (`pipeline_analytics.py`)**: Transforms and aggregates data using **Polars**, calculating monthly, yearly, and global rankings across companies via `.over()` window functions.
4. **Delivery (`Whale_Tour_Captain_Leaderboards.xlsx`)**: Generates an automated, multi-tab formatted Excel workbook for executive reporting.

## Tech Stack
* **Language:** Python 3.11+
* **Data Processing:** Polars (high-performance columnar operations)
* **AI / Schemas:** Pydantic v2, OpenAI API
* **Reporting:** XlsxWriter (native Excel table styling)

## Setup & Run
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run analytics and build the Excel deliverable
python pipeline_analytics.py