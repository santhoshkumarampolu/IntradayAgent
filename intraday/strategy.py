# Scoring logic for stock selection

import pandas as pd, numpy as np
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
load_dotenv()

# Helper to check for negative news
NEGATIVE_KEYWORDS = ["fraud", "scam", "loss", "probe", "investigation", "default", "fire", "strike", "layoff", "bankrupt", "penalty", "fine", "downturn", "lawsuit", "resign", "crash", "collapse"]

def has_negative_news(newsapi, symbol):
    query = f"{symbol} stock India"
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=10)
    for article in articles.get('articles', []):
        text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
        if any(word in text for word in NEGATIVE_KEYWORDS):
            return True
    return False

def positive_news_score(newsapi, symbol):
    query = f"{symbol} stock India positive"
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
    # Boost score for each positive article found
    return len(articles.get('articles', [])) * 0.1  # 0.1 score per positive article

def score_stocks(bhav: pd.DataFrame, pre: dict) -> pd.DataFrame:
    bhav = bhav.copy()
    # Try to find the correct open price column
    open_col = None
    for col in bhav.columns:
        if col.strip().lower() in ["open", "open_price", "openprc", "open pric", "openprice", "opnpric"]:
            open_col = col
            break
    if open_col is None:
        raise ValueError(f"Could not find an open price column in Bhavcopy. Columns: {list(bhav.columns)}")
    bhav["pct_change"] = (bhav["ClsPric"] - bhav[open_col]) / bhav[open_col]
    # Filter for liquid stocks (turnover ≥ ₹10 Cr)
    bhav["turnover"] = bhav["ClsPric"] * bhav["TtlTradgVol"]
    liquid = bhav[bhav["turnover"] >= 10e7]
    picks = liquid.nlargest(5, "pct_change")
    if picks.empty:
        return picks
    picks = picks.copy()
    picks["entry"] = picks["ClsPric"]
    picks["stop"] = picks["entry"] * 0.99  # 1% SL
    picks["target"] = picks["entry"] * 1.02  # 2% target
    return picks.reset_index(drop=True)
