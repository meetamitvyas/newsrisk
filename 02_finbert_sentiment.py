from transformers import pipeline
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Load FinBERT model ─────────────────────────────────────────────────
# What is a pipeline? In plain English:
# A pipeline is a ready-made wrapper that handles everything for you:
# - Loading the model
# - Tokenizing the text (splitting into tokens)
# - Running the model
# - Returning the result
# We just pass text in and get sentiment out

print("Loading FinBERT model...")
print("(First run downloads ~500MB - this may take a few minutes)")
print()

# ProsusAI/finbert is the most widely used financial sentiment model
# It was trained specifically on financial news and analyst reports
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)
print("FinBERT loaded successfully!")
print()

# ── Same headlines as VADER for direct comparison ─────────────────────
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

print("FinBERT Sentiment Analysis on Financial Headlines")
print("=" * 70)
print(f"{'Headline':<45} {'Score':>8} {'Label':>12}")
print("-" * 70)

results = []
for headline in headlines:
    # FinBERT returns label and score (confidence 0 to 1)
    result = finbert(headline)[0]
    label = result["label"].upper()
    score = result["score"]

    results.append({
        "headline": headline,
        "label": label,
        "confidence": round(score, 3)
    })

    short = headline[:43] + ".." if len(headline) > 43 else headline
    print(f"{short:<45} {score:>8.3f} {label:>12}")

print()
print("Summary:")
df = pd.DataFrame(results)
print(df["label"].value_counts())