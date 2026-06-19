from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Load both models ───────────────────────────────────────────────────
print("Loading VADER...")
vader = SentimentIntensityAnalyzer()

print("Loading FinBERT...")
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)
print("Both models loaded!")
print()

# ── Same 10 headlines ─────────────────────────────────────────────────
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

# ── Run both models ───────────────────────────────────────────────────
results = []
for headline in headlines:
    # VADER
    vader_score = vader.polarity_scores(headline)["compound"]
    if vader_score >= 0.05:
        vader_label = "POSITIVE"
    elif vader_score <= -0.05:
        vader_label = "NEGATIVE"
    else:
        vader_label = "NEUTRAL"

    # FinBERT
    finbert_result = finbert(headline)[0]
    finbert_label = finbert_result["label"].upper()
    finbert_conf = finbert_result["score"]

    # Do they agree?
    agree = "✅" if vader_label == finbert_label else "❌"

    results.append({
        "headline": headline[:40] + "..",
        "vader_label": vader_label,
        "vader_score": vader_score,
        "finbert_label": finbert_label,
        "finbert_confidence": round(finbert_conf, 3),
        "agree": agree
    })

df = pd.DataFrame(results)

# ── Print comparison table ─────────────────────────────────────────────
print("VADER vs FinBERT Comparison")
print("=" * 75)
print(f"{'Headline':<42} {'VADER':>10} {'FinBERT':>10} {'Agree':>6}")
print("-" * 75)
for _, row in df.iterrows():
    print(f"{row['headline']:<42} {row['vader_label']:>10} "
          f"{row['finbert_label']:>10} {row['agree']:>6}")

agreements = df["agree"].value_counts()
print()
print(f"Agreement: {agreements.get('✅', 0)}/10 headlines")
print(f"Disagreement: {agreements.get('❌', 0)}/10 headlines")

# ── Visualisation ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Color mapping
color_map = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}

# Chart 1: VADER scores
vader_colors = [color_map[l] for l in df["vader_label"]]
axes[0].barh(range(len(df)), df["vader_score"],
             color=vader_colors, alpha=0.7)
axes[0].axvline(x=0, color="black", linewidth=0.8)
axes[0].axvline(x=0.05, color="green", linewidth=0.5, linestyle="--")
axes[0].axvline(x=-0.05, color="red", linewidth=0.5, linestyle="--")
axes[0].set_yticks(range(len(df)))
axes[0].set_yticklabels(df["headline"], fontsize=7)
axes[0].set_title("VADER Sentiment Scores")
axes[0].set_xlabel("Compound Score (-1 to +1)")

# Chart 2: FinBERT confidence
finbert_colors = [color_map[l] for l in df["finbert_label"]]
axes[1].barh(range(len(df)), df["finbert_confidence"],
             color=finbert_colors, alpha=0.7)
axes[1].set_yticks(range(len(df)))
axes[1].set_yticklabels(df["headline"], fontsize=7)
axes[1].set_title("FinBERT Confidence Scores")
axes[1].set_xlabel("Confidence (0 to 1)")

# Legend
pos = mpatches.Patch(color="green", label="Positive")
neg = mpatches.Patch(color="red", label="Negative")
neu = mpatches.Patch(color="gray", label="Neutral")
fig.legend(handles=[pos, neg, neu], loc="lower center",
           ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.suptitle("NewsRisk — VADER vs FinBERT Sentiment Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("vader_vs_finbert.png", bbox_inches="tight")
plt.show()
print("Chart saved as vader_vs_finbert.png")