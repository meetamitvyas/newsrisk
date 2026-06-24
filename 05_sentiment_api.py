from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uvicorn

# ── Create API ────────────────────────────────────────────────────────
app = FastAPI(title="NewsRisk Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load VADER once when server starts ────────────────────────────────
print("Loading VADER sentiment analyser...")
vader = SentimentIntensityAnalyzer()
print("Ready!")

# ── Input schema ──────────────────────────────────────────────────────
class HeadlineInput(BaseModel):
    headline: str

# ── Helper: find words driving sentiment ─────────────────────────────
def get_word_scores(headline):
    """Score each word individually to find sentiment drivers."""
    words = headline.split()
    word_scores = []
    for word in words:
        clean = word.strip(".,!?\"'").lower()
        score = vader.polarity_scores(clean)["compound"]
        word_scores.append({
            "word": word,
            "score": round(score, 3)
        })
    return word_scores

# ── Health check endpoint ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "NewsRisk Sentiment API is running"}

# ── Main sentiment endpoint ────────────────────────────────────────────
@app.post("/analyse")
def analyse_sentiment(input: HeadlineInput):
    headline = input.headline.strip()

    if not headline:
        return {"error": "Please provide a headline"}

    # Get overall sentiment
    scores = vader.polarity_scores(headline)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    # Get word-level scores for highlighting
    word_scores = get_word_scores(headline)

    return {
        "headline": headline,
        "label": label,
        "compound_score": round(compound, 4),
        "positive_score": round(scores["pos"], 4),
        "negative_score": round(scores["neg"], 4),
        "neutral_score": round(scores["neu"], 4),
        "word_scores": word_scores
    }