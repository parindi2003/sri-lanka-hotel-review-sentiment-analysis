"""
Sri Lanka Hotel Booking App Review Sentiment Analysis - Dashboard
---------------------------------------------------------------------
Streamlit dashboard to visualize sentiment analysis results.

Install requirements first:
    pip install streamlit plotly

Run this in the same folder as agoda_reviews_with_sentiment.csv:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hotel Review Sentiment Dashboard", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("agoda_reviews_with_sentiment.csv")
    df['review_date'] = pd.to_datetime(df['review_date'])
    return df

df = load_data()

# ---------- Header ----------
st.title("🏨 Sri Lanka Hotel Booking App Review Sentiment Analysis")
st.markdown("Sentiment analysis of a hotel booking app's reviews from the Sri Lanka Google Play Store, powered by DistilBERT (Hugging Face Transformers).")

# ---------- Key metrics ----------
total_reviews = len(df)
positive_count = len(df[df['sentiment'] == 'POSITIVE'])
negative_count = len(df[df['sentiment'] == 'NEGATIVE'])
positive_pct = round(positive_count / total_reviews * 100, 1)
negative_pct = round(negative_count / total_reviews * 100, 1)
avg_rating = round(df['rating'].mean(), 2)

st.subheader("📊 Key Metrics Overview")
st.caption("A quick snapshot of overall customer sentiment based on the cleaned review dataset.")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Total Reviews", f"{total_reviews:,}",
    help="Number of cleaned reviews analyzed after removing duplicates and blank entries"
)
col2.metric(
    "Positive %", f"{positive_pct}%",
    help="Percentage of reviews classified as POSITIVE by the DistilBERT sentiment model"
)
col3.metric(
    "Negative %", f"{negative_pct}%",
    help="Percentage of reviews classified as NEGATIVE by the DistilBERT sentiment model"
)
col4.metric(
    "Average Rating", f"{avg_rating} ⭐",
    help="Average star rating (out of 5) across all analyzed reviews"
)

st.divider()

# ---------- Charts row 1 ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_pie = px.pie(
        sentiment_counts, names='sentiment', values='count',
        color='sentiment',
        color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.subheader("Star Rating Distribution")
    rating_counts = df['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['rating', 'count']
    fig_bar = px.bar(
        rating_counts, x='rating', y='count',
        color='rating', color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------- Trend over time ----------
st.subheader("Sentiment Trend Over Time")
df['month'] = df['review_date'].dt.to_period('M').astype(str)
trend = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
fig_trend = px.line(
    trend, x='month', y='count', color='sentiment',
    color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'},
    markers=True
)
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ---------- Sample reviews explorer ----------
st.subheader("🔍 Explore Reviews")
sentiment_filter = st.selectbox("Filter by sentiment", ["All", "POSITIVE", "NEGATIVE"])

filtered_df = df if sentiment_filter == "All" else df[df['sentiment'] == sentiment_filter]

st.dataframe(
    filtered_df[['review_date', 'rating', 'sentiment', 'sentiment_score', 'review_text']]
    .sort_values('review_date', ascending=False)
    .head(50),
    use_container_width=True,
    hide_index=True
)

st.caption("Disclaimer: Independent portfolio project. Not affiliated with or sponsored by Agoda. Data sourced from public Google Play reviews.")