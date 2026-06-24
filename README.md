# NewsRisk — Financial News Sentiment Analyser

An NLP-powered system that analyses sentiment of financial news headlines
using two approaches — VADER (rule-based) and FinBERT (transformer-based)
— with a live Streamlit dashboard for real-time analysis.

---

## What This Project Does

NewsRisk fetches real financial news headlines from NewsAPI, analyses their
sentiment using both VADER and FinBERT, compares the two models, and
provides an interactive dashboard where analysts can type any headline
and instantly see its sentiment with word-level highlighting.

---

## Key Finding

**VADER vs FinBERT — 50% agreement on financial headlines.**

VADER (general dictionary) misclassified finance-specific phrases:
- "Apple reports record quarterly earnings beating all expectations" → NEGATIVE ❌
- "Goldman Sachs beats analyst expectations" → NEUTRAL ❌

FinBERT (trained on financial text) correctly classified both → POSITIVE ✅

**Lesson: Domain-specific models are essential for financial NLP.**

---

## Live Dashboard Features

- **Single headline analyser** — type any headline, get instant sentiment
- **Word highlighting** — green = positive words, pink = negative words
- **Company dashboard** — fetch real news for 5 major companies
- **Sentiment charts** — average sentiment and distribution by company
- **CSV download** — export all results for further analysis

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| VADER | Rule-based sentiment (fast, general purpose) |
| FinBERT (HuggingFace) | Transformer-based sentiment (finance-specific) |
| NewsAPI | Real-time financial news headlines |
| Streamlit | Interactive dashboard frontend |
| Pandas | Data manipulation |
| Matplotlib | Charts and visualisations |
| python-dotenv | Secure environment variable management |

---

## Project Structure

newsrisk/

├── 01_vader_sentiment.py      # VADER baseline analysis

├── 02_finbert_sentiment.py    # FinBERT analysis

├── 03_compare_models.py       # VADER vs FinBERT comparison

├── 04_fetch_real_news.py      # Fetch real news from NewsAPI

├── 05_streamlit_app.py        # Interactive Streamlit dashboard

├── .env                       # API keys (NOT committed to GitHub)

└── .gitignore                 # Excludes venv, .env, data files


---

## How to Run

```bash
git clone https://github.com/meetamitvyas/newsrisk.git
cd newsrisk
python -m venv venv
venv\Scripts\activate
pip install pandas requests vaderSentiment transformers torch
pip install streamlit python-dotenv matplotlib yfinance
```

Create a `.env` file:

NEWS_API_KEY=your_newsapi_key_here
HF_TOKEN=your_huggingface_token_here

Run the dashboard:
```bash
streamlit run 05_streamlit_app.py
```

---

## Models Used

**VADER** — Valence Aware Dictionary and sEntiment Reasoner
- Rule-based, no training needed
- Works on general English text
- Fast — analyses thousands of headlines per second
- Limitation: misses financial domain vocabulary

**FinBERT** — BERT fine-tuned on financial text
- Trained on financial news, earnings calls, analyst reports
- Understands "beats expectations", "raises outlook", "misses guidance"
- More accurate for financial NLP
- Trade-off: slower, requires ~500MB model download

---

## Author

**Amit Vyas** — Principal Data Engineer
13+ years experience in data warehousing, analytics, and AI/ML
[GitHub](https://github.com/meetamitvyas) |
[LinkedIn](https://linkedin.com/in/meetamitvyas)

