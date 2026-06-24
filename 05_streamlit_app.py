from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsRisk — Financial Sentiment Analyser",
    page_icon="📰",
    layout="wide"
)

# ── Title ──────────────────────────────────────────────────────────────
st.title("📰 NewsRisk — Financial Sentiment Analyser")
st.markdown("Analyse sentiment of financial news headlines using NLP")
st.divider()

# ── Load VADER once (cached so it doesn't reload on every interaction) ─
# @st.cache_resource means: load this once and keep it in memory
# Without this, VADER would reload every time user types something
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

vader = load_vader()

# ── Section 1: Single Headline Analyser ───────────────────────────────
st.header("🔍 Analyse a Headline")

headline = st.text_input(
    "Enter a financial news headline:",
    placeholder="e.g. Apple reports record quarterly earnings, stock surges"
)

if st.button("Analyse Sentiment", type="primary"):
    if headline.strip():
        scores = vader.polarity_scores(headline)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "POSITIVE"
            color = "green"
            emoji = "🟢"
        elif compound <= -0.05:
            label = "NEGATIVE"
            color = "red"
            emoji = "🔴"
        else:
            label = "NEUTRAL"
            color = "gray"
            emoji = "⚪"

        # Show result in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sentiment", f"{emoji} {label}")
        col2.metric("Compound Score", f"{compound:.3f}")
        col3.metric("Positive", f"{scores['pos']:.3f}")
        col4.metric("Negative", f"{scores['neg']:.3f}")

        # Word highlighting
        st.markdown("**Word-level sentiment drivers:**")
        words = headline.split()
        highlighted = ""
        for word in words:
            clean = word.strip(".,!?\"'").lower()
            score = vader.polarity_scores(clean)["compound"]
            if score > 0.05:
                highlighted += f'<span style="background-color:#90EE90; padding:2px 4px; margin:2px; border-radius:3px">{word}</span> '
            elif score < -0.05:
                highlighted += f'<span style="background-color:#FFB6C1; padding:2px 4px; margin:2px; border-radius:3px">{word}</span> '
            else:
                highlighted += f'{word} '

        st.markdown(highlighted, unsafe_allow_html=True)
        st.caption("🟢 Green = positive words  🔴 Pink = negative words")
    else:
        st.warning("Please enter a headline first.")

st.divider()

# ── Section 2: Company News Dashboard ─────────────────────────────────
st.header("📊 Company Sentiment Dashboard")
st.markdown("Fetch and analyse real news for major financial companies")

companies = {
    "Apple": "AAPL",
    "JPMorgan": "JPM",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Mastercard": "MA"
}

selected = st.multiselect(
    "Select companies to analyse:",
    options=list(companies.keys()),
    default=["Apple", "JPMorgan", "Goldman Sachs"]
)

num_articles = st.slider("Number of articles per company:", 5, 20, 10)

if st.button("Fetch & Analyse News", type="primary"):
    API_KEY = os.environ.get("NEWS_API_KEY")
    if not API_KEY:
        st.error("NEWS_API_KEY not found. Please set it in your .env file.")
    else:
        all_headlines = []
        progress = st.progress(0)

        for i, company in enumerate(selected):
            with st.spinner(f"Fetching news for {company}..."):
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": company,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": num_articles,
                    "apiKey": API_KEY
                }
                response = requests.get(url, params=params)
                data = response.json()

                if data["status"] == "ok":
                    for article in data["articles"]:
                        if article["title"] and article["title"] != "[Removed]":
                            scores = vader.polarity_scores(article["title"])
                            compound = scores["compound"]
                            if compound >= 0.05:
                                label = "POSITIVE"
                            elif compound <= -0.05:
                                label = "NEGATIVE"
                            else:
                                label = "NEUTRAL"

                            all_headlines.append({
                                "company": company,
                                "headline": article["title"],
                                "source": article["source"]["name"],
                                "published": article["publishedAt"][:10],
                                "sentiment": label,
                                "score": round(compound, 3)
                            })

            progress.progress((i + 1) / len(selected))

        if all_headlines:
            df = pd.DataFrame(all_headlines)

            # Summary chart
            st.subheader("Average Sentiment by Company")
            summary = df.groupby("company")["score"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["green" if x > 0 else "red" for x in summary]
            ax.barh(summary.index, summary.values, color=colors, alpha=0.7)
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("Average Sentiment Score")
            plt.tight_layout()
            st.pyplot(fig)

            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            dist = df.groupby(["company", "sentiment"]).size().unstack(fill_value=0)
            st.bar_chart(dist)

            # Headlines table
            st.subheader("All Headlines")
            st.dataframe(
                df[["company", "headline", "sentiment", "score", "published"]],
                use_container_width=True
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="news_sentiment.csv",
                mime="text/csv"
            )