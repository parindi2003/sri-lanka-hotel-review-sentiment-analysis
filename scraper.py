"""
Agoda Google Play Review Scraper
----------------------------------
Scrapes hotel booking app reviews from Agoda's Google Play Store page.
"""

from google_play_scraper import Sort, reviews
import pandas as pd
import time

# Agoda's official Google Play package ID
APP_ID = "com.agoda.mobile.consumer"

# How many reviews we want in total
TARGET_COUNT = 2000
BATCH_SIZE = 200  # reviews fetched per request

def scrape_reviews(app_id, target_count):
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < target_count:
        result, continuation_token = reviews(
            app_id,
            lang='en',        # English reviews
            country='lk',     # Sri Lanka store
            sort=Sort.NEWEST, # newest reviews first
            count=BATCH_SIZE,
            continuation_token=continuation_token
        )

        if not result:
            print("No more reviews available.")
            break

        all_reviews.extend(result)
        print(f"Collected {len(all_reviews)} reviews so far...")

        if continuation_token is None:
            break

        time.sleep(1)

    return all_reviews[:target_count]


if __name__ == "__main__":
    print(f"Starting scrape for app: {APP_ID}")
    raw_reviews = scrape_reviews(APP_ID, TARGET_COUNT)

    df = pd.DataFrame(raw_reviews)[[
        'reviewId', 'userName', 'content', 'score',
        'thumbsUpCount', 'at', 'appVersion'
    ]]
    df.rename(columns={
        'content': 'review_text',
        'score': 'rating',
        'at': 'review_date'
    }, inplace=True)

    output_path = "agoda_reviews_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDone! Saved {len(df)} reviews to {output_path}")