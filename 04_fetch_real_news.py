from dotenv import load_dotenv
load_dotenv()
import requests
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Get API key from environment variable ──────────────────────────────
# We NEVER hardcode API keys in code
# Always read from environment variables
API_KEY = os.environ.get("NEWS_API_KEY")
if not API_KEY:
    raise ValueError("NEWS_API_KEY environment variable not set!")

# ── Define companies to track ──────────────────────────────────────────
companies = {
    "Apple": "AAPL",
    "JPMorgan": "JPM",
    "Goldman Sachs": "GS",
    "Visa": "V",
    "Mastercard": "MA"
}

# ── Fetch news for each company ────────────────────────────────────────
print("Fetching real financial news...")
print("=" * 55)

all_headlines = []

for company, ticker in companies.items():
    # NewsAPI endpoint
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company,           # search query
        "language": "en",       # English only
        "sortBy": "publishedAt",# most recent first
        "pageSize": 10,         # 10 headlines per company
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] == "ok":
        articles = data["articles"]
        print(f"{company}: {len(articles)} articles fetched")
        for article in articles:
            if article["title"] and article["title"] != "[Removed]":
                all_headlines.append({
                    "company": company,
                    "ticker": ticker,
                    "headline": article["title"],
                    "source": article["source"]["name"],
                    "published": article["publishedAt"][:10]
                })
    else:
        print(f"{company}: Error - {data.get('message', 'Unknown error')}")

print(f"\nTotal headlines fetched: {len(all_headlines)}")
df = pd.DataFrame(all_headlines)

# ── Run VADER on all headlines ─────────────────────────────────────────
print("\nRunning VADER sentiment analysis...")
vader = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "POSITIVE", score
    elif score <= -0.05:
        return "NEGATIVE", score
    else:
        return "NEUTRAL", score

df[["vader_label", "vader_score"]] = df["headline"].apply(
    lambda x: pd.Series(get_vader_sentiment(x))
)

# ── Show results ──────────────────────────────────────────────────────
print("\nSample headlines with sentiment:")
print("=" * 70)
for company in companies.keys():
    company_df = df[df["company"] == company].head(3)
    print(f"\n{company}:")
    for _, row in company_df.iterrows():
        print(f"  [{row['vader_label']:>8}] {row['headline'][:60]}")

# ── Company sentiment summary ─────────────────────────────────────────
print("\n\nSentiment Summary by Company:")
print("=" * 55)
summary = df.groupby("company")["vader_score"].agg(["mean", "count"])
summary.columns = ["avg_sentiment", "article_count"]
summary = summary.sort_values("avg_sentiment", ascending=False)
print(summary.round(3))

# ── Save results ──────────────────────────────────────────────────────
import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/news_sentiment.csv", index=False)
print("\nSaved to data/news_sentiment.csv")

# ── Visualise ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["green" if x > 0 else "red" for x in summary["avg_sentiment"]]
bars = ax.barh(summary.index, summary["avg_sentiment"],
               color=colors, alpha=0.7)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_title("Average News Sentiment by Company\n(Based on Real Headlines Today)")
ax.set_xlabel("Average VADER Sentiment Score")
for bar, val in zip(bars, summary["avg_sentiment"]):
    ax.text(val + 0.005 if val >= 0 else val - 0.005,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig("company_sentiment.png")
plt.show()
print("Chart saved as company_sentiment.png")