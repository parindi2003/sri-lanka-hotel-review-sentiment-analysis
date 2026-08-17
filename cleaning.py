"""
Agoda Reviews - Data Cleaning Script
--------------------------------------
Cleans the raw scraped reviews: removes duplicates, blank/too-short
reviews, and normalizes text. Outputs a clean CSV ready for sentiment
analysis.

Run this in the same folder as agoda_reviews_raw.csv:
    python cleaning.py
"""

import pandas as pd

INPUT_FILE = "agoda_reviews_raw.csv"
OUTPUT_FILE = "agoda_reviews_clean.csv"
MIN_WORD_COUNT = 3  # reviews shorter than this are dropped

def clean_reviews(df):
    original_count = len(df)
    print(f"Starting with {original_count} raw reviews.")

    # 1. Drop rows where review_text is missing entirely
    df = df.dropna(subset=['review_text'])
    print(f"After removing missing text: {len(df)}")

    # 2. Strip extra whitespace from review text
    df['review_text'] = df['review_text'].astype(str).str.strip()

    # 3. Remove blank / too-short reviews (e.g. "ok", "good", "")
    df['word_count'] = df['review_text'].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= MIN_WORD_COUNT]
    print(f"After removing too-short reviews (<{MIN_WORD_COUNT} words): {len(df)}")

    # 4. Remove duplicate reviews (same reviewId or exact same text)
    df = df.drop_duplicates(subset=['reviewId'])
    df = df.drop_duplicates(subset=['review_text'])
    print(f"After removing duplicates: {len(df)}")

    # 5. Drop rows missing a rating or review_date
    df = df.dropna(subset=['rating', 'review_date'])
    print(f"After removing rows with missing rating/date: {len(df)}")

    # 6. Convert review_date to a proper datetime type
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.dropna(subset=['review_date'])

    # Clean up helper column
    df = df.drop(columns=['word_count'])

    removed = original_count - len(df)
    print(f"\nCleaning complete. Removed {removed} rows total.")
    print(f"Final clean dataset: {len(df)} reviews")

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    clean_df = clean_reviews(df)
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved clean data to {OUTPUT_FILE}")

    # Quick summary stats, useful for your README/LinkedIn post later
    print("\n--- Quick Summary ---")
    print(f"Rating distribution:\n{clean_df['rating'].value_counts().sort_index()}")
    print(f"\nDate range: {clean_df['review_date'].min()} to {clean_df['review_date'].max()}")