import polars as pl
import xlsxwriter

def build_excel_report(
    input_file: str = "data/enriched_reviews.jsonl",
    output_excel: str = "Whale_Tour_Captain_Leaderboards.xlsx"
):
    # 1. Load the data
    df = pl.read_ndjson(input_file)

    # 2. Filter valid captain mentions & parse date safely
    valid_df = (
        df.filter(
            (pl.col("captain_name") != "None Mentioned") & 
            (pl.col("captain_name").is_not_null())
        )
        .with_columns(
            pl.col("published_at")
            .str.slice(0, 10)
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("parsed_date")
        )
        .filter(pl.col("parsed_date").is_not_null())
        .with_columns(
            pl.col("parsed_date").dt.year().alias("year"),
            pl.col("parsed_date").dt.month().alias("month")
        )
    )

    # -------------------------------------------------------------
    # 3. Aggregations & Window Functions
    # -------------------------------------------------------------

    # Sheet 1: Overall Leaderboard
    overall_df = (
        valid_df.group_by(["captain_name", "place_name"])
        .len(name="review_mentions")
        .with_columns(
            pl.col("review_mentions")
            .rank(descending=True, method="min")
            .alias("global_rank")
        )
        .sort(["global_rank", "review_mentions"], descending=[False, True])
        .select(["global_rank", "captain_name", "place_name", "review_mentions"])
    )

    # Sheet 2: Rankings by Tour Operator / Place
    company_df = (
        valid_df.group_by(["place_name", "captain_name"])
        .len(name="review_mentions")
        .with_columns(
            pl.col("review_mentions")
            .rank(descending=True, method="min")
            .over("place_name")
            .alias("company_rank")
        )
        .sort(["place_name", "company_rank"])
        .select(["place_name", "company_rank", "captain_name", "review_mentions"])
    )

    # Sheet 3: Yearly Leaderboard
    yearly_df = (
        valid_df.group_by(["year", "place_name", "captain_name"])
        .len(name="review_mentions")
        .with_columns(
            pl.col("review_mentions")
            .rank(descending=True, method="min")
            .over(["year", "place_name"])
            .alias("yearly_rank")
        )
        .sort(["year", "place_name", "yearly_rank"])
        .select(["year", "place_name", "yearly_rank", "captain_name", "review_mentions"])
    )

    # Sheet 4: Monthly Leaderboard
    monthly_df = (
        valid_df.group_by(["year", "month", "place_name", "captain_name"])
        .len(name="review_mentions")
        .with_columns(
            pl.col("review_mentions")
            .rank(descending=True, method="min")
            .over(["year", "month", "place_name"])
            .alias("monthly_rank")
        )
        .sort(["year", "month", "place_name", "monthly_rank"])
        .select(["year", "month", "place_name", "monthly_rank", "captain_name", "review_mentions"])
    )

    # -------------------------------------------------------------
    # 4. Generate Multi-Tab Excel Deliverable
    # -------------------------------------------------------------
    print(f"Generating workbook: {output_excel}...")
    
    with xlsxwriter.Workbook(output_excel) as workbook:
        overall_df.write_excel(
            workbook=workbook,
            worksheet="Overall Leaderboard",
            autofit=True,
            table_style="Table Style Medium 2"
        )
        
        company_df.write_excel(
            workbook=workbook,
            worksheet="Rankings by Company",
            autofit=True,
            table_style="Table Style Medium 4"
        )

        yearly_df.write_excel(
            workbook=workbook,
            worksheet="Yearly Breakdown",
            autofit=True,
            table_style="Table Style Medium 9"
        )

        monthly_df.write_excel(
            workbook=workbook,
            worksheet="Monthly Breakdown",
            autofit=True,
            table_style="Table Style Medium 10"
        )

    print("Success! Deliverable workbook generated.")

if __name__ == "__main__":
    build_excel_report()