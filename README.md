#  Sri Lanka Hotel Booking App Review Sentiment Analysis

An end-to-end NLP sentiment analysis pipeline and interactive dashboard, built from real hotel booking app reviews on the Sri Lanka Google Play Store.



---

##  Overview

This project scrapes, cleans, analyzes, and visualizes ~2,000 customer reviews of a popular hotel booking app to uncover overall customer sentiment, key pain points, and trends over time — using a fully automated Python pipeline, powered by a pre-trained Transformer model (DistilBERT).

**Live Dashboard:** (https://sl-hotel-review-sentiment.streamlit.app/)

---

##  Key Findings

| Metric | Value |
|---|---|
| Total reviews analyzed | 1,748 (cleaned from 2,000 raw) |
| Negative sentiment | 85.4% |
| Positive sentiment | 14.6% |
| Average rating | 1.71 / 5 ⭐ |
| Date range | April 2026 – August 2026 |

- Sentiment is heavily skewed negative, consistent with the star rating distribution (mostly 1-star and 5-star reviews, few in between).
- The most common complaints center around **refunds, trust/scam concerns, and booking issues**.
- Negative sentiment trended consistently high through the review period, with a slight improvement in the most recent months.

---

##  Pipeline — Step by Step

### 1️. Data Collection
**Script:** `scraper.py`

Scrapes reviews directly from the Google Play Store using the `google-play-scraper` Python library, targeting the Sri Lanka store (`country='lk'`).

- Collected **2,000 raw reviews** in batches of 200, using pagination (`continuation_token`)
- A 1-second delay between requests to avoid overloading Google's servers
- Output: `agoda_reviews_raw.csv`

```bash
python scraper.py
```

### 2️. Data Cleaning
**Script:** `cleaning.py`

Raw scraped data is noisy — this step prepares it for reliable analysis.

- Removed reviews with missing text
- Stripped extra whitespace
- Removed blank/too-short reviews (fewer than 3 words)
- Removed duplicate reviews (by ID and by exact text match)
- Removed rows with missing rating or date
- Converted review dates to proper datetime format

**Result:** 2,000 → **1,748 clean reviews** (12.6% removed)

Output: `agoda_reviews_clean.csv`

```bash
python cleaning.py
```

### 3️. Sentiment Analysis
**Script:** `sentiment_analysis.py`

Each cleaned review is passed through a pre-trained **DistilBERT** model to classify it as `POSITIVE` or `NEGATIVE`, with a confidence score.

- **Model:** [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) (Hugging Face)
- DistilBERT is a smaller, faster, distilled version of Google's BERT model, fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset
- Long reviews are truncated to 1,000 characters (model input limit ~512 tokens)
- Runs via Hugging Face's `pipeline("sentiment-analysis")` API

Output: `agoda_reviews_with_sentiment.csv`

```bash
python sentiment_analysis.py
```

### 4️. Interactive Dashboard
**Script:** `dashboard.py`

An interactive web dashboard built with **Streamlit** and **Plotly**, featuring:

- Key metrics overview (total reviews, sentiment %, average rating) with hover tooltips
- Sentiment distribution pie chart
- Star rating distribution bar chart
- Sentiment trend over time (line chart)
- Filterable, searchable review explorer table

```bash
streamlit run dashboard.py
```

### 5️. Deployment
**Live on:** Streamlit Community Cloud

The dashboard is deployed for free, public access using [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Code and a `requirements.txt` (listing `streamlit`, `plotly`, `pandas`) were pushed to this GitHub repository
2. Signed in to Streamlit Cloud via GitHub OAuth
3. Selected **"Deploy a public app from GitHub"**, specifying:
   - Repository: this repo
   - Branch: `main`
   - Main file path: `dashboard.py`
4. Streamlit Cloud automatically clones the repo, installs dependencies from `requirements.txt`, and runs the app on a public URL
5. The app auto-redeploys whenever new changes are pushed to the `main` branch (continuous deployment)

Note: the heavier ML dependencies (`torch`, `transformers`) are **not** required for deployment, since the dashboard only reads the already-processed CSV output — keeping the deployed app lightweight and fast to build.

---

##  Tech Stack

| Category | Tool |
|---|---|
| Language | Python |
| Data Collection | `google-play-scraper` |
| Data Processing | `pandas` |
| NLP / Sentiment Model | DistilBERT via `transformers` (Hugging Face) |
| ML Backend | `torch` (PyTorch) |
| Visualization | `plotly` |
| Dashboard / Web App | `streamlit` |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |
| Editor | VS Code |

---

##  Repository Structure

```
sri-lanka-hotel-review-sentiment-analysis/
├── scraper.py                          # Step 1: Data collection
├── cleaning.py                         # Step 2: Data cleaning
├── sentiment_analysis.py               # Step 3: Sentiment inference
├── dashboard.py                        # Step 4: Streamlit dashboard
├── requirements.txt                    # Dependencies for deployment
├── agoda_reviews_raw.csv               # Raw scraped data
├── agoda_reviews_clean.csv             # Cleaned data
├── agoda_reviews_with_sentiment.csv    # Final data with sentiment labels
└── README.md
```

---

# 1. Clone the repository
git clone https://github.com/parindi2003/sri-lanka-hotel-review-sentiment-analysis.git
cd sri-lanka-hotel-review-sentiment-analysis

# 2. Install dependencies
pip install google-play-scraper pandas transformers torch streamlit plotly

# 3. Run the full pipeline, in order
python scraper.py
python cleaning.py
python sentiment_analysis.py

# 4. Launch the dashboard
streamlit run dashboard.py

##  Future Improvements

- Add Sinhala/Tamil language support with a custom Unicode-based language detector (kept out of scoring for transparency, similar to Project 1)
- Topic/reason classification to identify *why* a review is negative — planned as **Project 3** in this NLP series
- Compare sentiment across multiple hotel booking apps

Links
Live Dashboard: https://sl-hotel-review-sentiment.streamlit.app/
GitHub Repository: https://github.com/parindi2003/sri-lanka-hotel-review-sentiment-analysis

---

##  Disclaimer

This is an independent portfolio project created for educational purposes. It is not affiliated with, endorsed by, or sponsored by any hotel booking platform. All data was sourced from publicly available Google Play Store reviews.

---

*Part of an ongoing NLP project series. Feedback welcome!*