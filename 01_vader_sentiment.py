from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# ── What is SentimentIntensityAnalyzer? ────────────────────────────────
# This is the VADER model — it contains the dictionary of 7,500+ words
# with their sentiment scores. We create it once and reuse it.
analyzer = SentimentIntensityAnalyzer()

# ── Test with sample financial headlines ──────────────────────────────
headlines = [
    "Apple reports record quarterly earnings, stock surges",
    "JPMorgan faces massive fraud investigation by regulators",
    "Goldman Sachs beats analyst expectations for third quarter",
    "Federal Reserve raises interest rates amid inflation fears",
    "Visa announces new partnership with major e-commerce platform",
    "Mastercard shares fall after disappointing revenue report",
    "Apple launches new iPhone with revolutionary AI features",
    "JPMorgan CEO warns of economic recession risk in 2026",
    "Goldman Sachs upgrades Apple stock to strong buy rating",
    "Market uncertainty grows as trade tensions escalate"
]

print("VADER Sentiment Analysis on Financial Headlines")
print("=" * 65)
print(f"{'Headline':<45} {'Score':>8} {'Label':>10}")
print("-" * 65)

results = []
for headline in headlines:
    # scores returns 4 values:
    # neg = negative score (0 to 1)
    # neu = neutral score (0 to 1)
    # pos = positive score (0 to 1)
    # compound = overall score (-1 to +1) — this is what we use
    scores = analyzer.polarity_scores(headline)
    compound = scores["compound"]

    # Classify based on compound score
    # Industry standard thresholds:
    # compound >= 0.05  → Positive
    # compound <= -0.05 → Negative
    # between           → Neutral
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    results.append({
        "headline": headline,
        "compound": compound,
        "label": label
    })

    # Truncate headline for display
    short = headline[:43] + ".." if len(headline) > 43 else headline
    print(f"{short:<45} {compound:>8.3f} {label:>10}")

print()
print("Summary:")
df = pd.DataFrame(results)
print(df["label"].value_counts())