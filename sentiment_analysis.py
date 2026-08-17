"""
Agoda Reviews - Sentiment Analysis Script
--------------------------------------------
Runs each cleaned review through a pre-trained DistilBERT sentiment
model (Hugging Face) and labels it POSITIVE or NEGATIVE with a
confidence score.

Install requirements first:
    pip install transformers torch

Run this in the same folder as agoda_reviews_clean.csv:
    python sentiment_analysis.py
"""

import pandas as pd
from transformers import pipeline

INPUT_FILE = "agoda_reviews_clean.csv"
OUTPUT_FILE = "agoda_reviews_with_sentiment.csv"

# DistilBERT can only handle ~512 tokens; long reviews get truncated
MAX_CHARS = 1000

def main():
    print("Loading clean reviews...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} reviews.")

    print("\nLoading sentiment model (first run downloads ~250MB, please wait)...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully.\n")

    labels = []
    scores = []

    total = len(df)
    for i, text in enumerate(df['review_text'].astype(str)):
        # Truncate very long reviews so the model doesn't error out
        truncated_text = text[:MAX_CHARS]
        result = sentiment_model(truncated_text)[0]

        labels.append(result['label'])
        scores.append(round(result['score'], 4))

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} reviews...")

    df['sentiment'] = labels
    df['sentiment_score'] = scores

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone! Saved results to {OUTPUT_FILE}")

    # Quick summary
    print("\n--- Sentiment Summary ---")
    counts = df['sentiment'].value_counts()
    percentages = df['sentiment'].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"{label}: {counts[label]} reviews ({percentages[label]:.1f}%)")


if __name__ == "__main__":
    main()