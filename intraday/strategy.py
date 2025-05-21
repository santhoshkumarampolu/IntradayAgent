# Scoring logic for stock selection

import pandas as pd, numpy as np
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
import openai
import ta  # Technical Analysis library
import yfinance as yf
import joblib
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

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_news_sentiment_openai(newsapi, symbol):
    query = f"{symbol} stock India"
    articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=5)
    headlines = [article.get('title', '') for article in articles.get('articles', [])]
    if not headlines:
        return 'neutral'  # No news, treat as neutral
    prompt = (
        "You are a financial news sentiment analyst. "
        "Given the following news headlines about an Indian stock, classify the overall sentiment as 'positive', 'neutral', or 'negative'. "
        "Headlines: " + " | ".join(headlines)
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        sentiment = response.choices[0].message.content.strip().lower()
        if 'positive' in sentiment:
            return 'positive'
        elif 'negative' in sentiment:
            return 'negative'
        else:
            return 'neutral'
    except Exception as e:
        print(f"OpenAI API error for {symbol}: {e}")
        return 'neutral'

def find_col(df, candidates):
    for c in candidates:
        for col in df.columns:
            if col.strip().lower() == c:
                return col
    return None

# --- Enhancement: Fundamental Data Placeholder ---
def get_fundamental_data(symbol):
    # TODO: Replace with real API or CSV lookup
    # Example: return dict with keys: 'pe', 'debt_equity', 'promoter_holding'
    return {'pe': 20, 'debt_equity': 0.2, 'promoter_holding': 55}

# --- Enhancement: Market Regime Awareness Placeholder ---
def get_market_regime():
    # TODO: Replace with real index data
    # Example: return dict with keys: 'trend' ('bull', 'bear', 'sideways'), 'volatility' (float)
    return {'trend': 'bull', 'volatility': 0.015}

# --- Advanced Enhancements Scaffold ---

def get_realtime_price(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        price = ticker.fast_info['lastPrice']
        return price
    except Exception as e:
        print(f"Realtime price fetch failed for {symbol}: {e}")
        return None

def get_order_book_features(symbol):
    # Example: Simulate order book features (replace with broker API for real data)
    # In production, connect to broker API (e.g., Zerodha Kite, Upstox) for live order book
    # Return dict with features like bid_ask_spread, imbalance, large_orders
    # Here, we use yfinance as a placeholder for bid/ask (not true order book)
    try:
        ticker = yf.Ticker(symbol + ".NS")
        info = ticker.fast_info
        bid = info.get('bid', None)
        ask = info.get('ask', None)
        bid_ask_spread = None
        if bid is not None and ask is not None and ask > 0:
            bid_ask_spread = (ask - bid) / ask
        # Simulate imbalance and large_orders as None (API required for real values)
        return {
            'bid_ask_spread': bid_ask_spread,
            'imbalance': None,  # Placeholder
            'large_orders': None  # Placeholder
        }
    except Exception as e:
        print(f"Order book fetch failed for {symbol}: {e}")
        return {'bid_ask_spread': None, 'imbalance': None, 'large_orders': None}

ml_model = None
try:
    ml_model = joblib.load('ml_intraday_model.pkl')  # Place your trained model in the project root
except Exception as e:
    print(f"ML model not loaded: {e}")

def ml_predict_intraday_move(features):
    global ml_model
    if ml_model is not None:
        # Convert features dict to DataFrame for model prediction
        try:
            import pandas as pd
            X = pd.DataFrame([features])
            prob = ml_model.predict_proba(X)[0, 1]  # Probability of positive class
            return prob
        except Exception as e:
            print(f"ML prediction error: {e}")
            return 0.5  # Neutral if error
    return 0.5  # Neutral if model not loaded

def detect_candlestick_patterns(row):
    # Example: Simple bullish engulfing pattern (requires open, close columns)
    # You can expand this with more patterns as needed
    try:
        open_price = row.get('OpnPric') or row.get('open') or row.get('Open')
        close_price = row.get('ClsPric') or row.get('close') or row.get('Close')
        prev_open = row.get('PrevOpnPric')
        prev_close = row.get('PrevClsPric')
        patterns = []
        if open_price and close_price and prev_open and prev_close:
            # Bullish engulfing: previous red, current green, current close > prev open, current open < prev close
            if prev_close < prev_open and close_price > open_price and close_price > prev_open and open_price < prev_close:
                patterns.append('bullish_engulfing')
        return patterns
    except Exception as e:
        print(f"Pattern detection error: {e}")
        return []

def get_sector_strength(symbol):
    # Example: Map symbol to sector and score sector strength
    # TODO: Replace with real sector mapping and sector index/ETF data
    sector_map = {
        'TCS': 'IT', 'INFY': 'IT', 'RELIANCE': 'Energy', 'HDFCBANK': 'Banking',
        # ...add more mappings as needed...
    }
    sector_strengths = {
        'IT': 1, 'Energy': 0, 'Banking': -1  # Example: +1 strong, 0 neutral, -1 weak
    }
    sector = sector_map.get(symbol.upper(), None)
    if sector:
        return sector_strengths.get(sector, 0)
    return 0  # Neutral if unknown

def check_event_driven(symbol):
    # Example: Check for earnings or results in the last 2 days using yfinance
    try:
        ticker = yf.Ticker(symbol + ".NS")
        cal = ticker.get_calendar()
        if cal is not None and not cal.empty:
            # Check for earnings date within last 2 days
            earnings_dates = cal.loc[cal.index.str.contains('Earnings Date')]
            if not earnings_dates.empty:
                event_date = pd.to_datetime(earnings_dates.iloc[0, 0])
                if abs((pd.Timestamp.now().normalize() - event_date).days) <= 2:
                    return True
        # TODO: Add more event checks (news, macro, etc.)
    except Exception as e:
        print(f"Event-driven check failed for {symbol}: {e}")
    return False

def get_correlation_pairs(symbol, universe):
    # Compute rolling correlation with other stocks in the universe (using close prices)
    # Placeholder: returns empty list unless you have historical price data for all symbols
    # For a real implementation, fetch historical close prices for all symbols, compute correlations
    # and return a list of highly correlated or anti-correlated pairs
    # Example: return [(other_symbol, correlation_value), ...]
    return []

def get_user_feedback(symbol):
    # Placeholder: Check for user feedback from a local file or database
    # Example: If a file 'user_feedback.csv' exists, read overrides or ratings
    import os
    feedback_file = 'user_feedback.csv'
    if os.path.exists(feedback_file):
        try:
            df = pd.read_csv(feedback_file)
            row = df[df['symbol'].str.upper() == symbol.upper()]
            if not row.empty:
                # Example: return a dict with feedback info
                return row.iloc[0].to_dict()
        except Exception as e:
            print(f"User feedback read error: {e}")
    return None  # No feedback found

def post_trade_analytics(trade_log):
    # trade_log: DataFrame with columns ['Stock', 'Entry', 'Target', 'Stop', 'Exit', 'PnL', 'log_time', ...]
    # Example analytics: win rate, avg PnL, max drawdown, slippage, execution quality
    if trade_log.empty:
        print("No trades to analyze.")
        return {}
    results = {}
    results['total_trades'] = len(trade_log)
    results['win_rate'] = (trade_log['PnL'] > 0).mean()
    results['avg_pnl'] = trade_log['PnL'].mean()
    results['max_drawdown'] = (trade_log['PnL'].cumsum().cummax() - trade_log['PnL'].cumsum()).max()
    results['avg_slippage'] = (trade_log['Entry'] - trade_log['Exit']).abs().mean() if 'Exit' in trade_log.columns else None
    # Print or log analytics
    print("Post-Trade Analytics:")
    for k, v in results.items():
        print(f"{k}: {v}")
    return results

def score_stocks(bhav: pd.DataFrame, pre: dict) -> pd.DataFrame:
    bhav = bhav.copy()
    # Detect open, high, low, close columns (add OpnPric, HghPric, LwPric, ClsPric)
    open_col = find_col(bhav, ["open", "open_price", "openprc", "open pric", "openprice", "opnpric", "opnpric"])
    high_col = find_col(bhav, ["high", "highpric", "high_price", "highprc", "high pric", "highprice", "hghpric"])
    low_col = find_col(bhav, ["low", "lowpric", "low_price", "lowprc", "low pric", "lowprice", "lwpric"])
    close_col = find_col(bhav, ["close", "clspric", "close_price", "closeprc", "close pric", "closeprice", "clspric"])
    for name, col in [("open", open_col), ("high", high_col), ("low", low_col), ("close", close_col)]:
        if col is None:
            raise ValueError(f"Could not find a {name} price column in Bhavcopy. Columns: {list(bhav.columns)})")
    bhav["pct_change"] = (bhav[close_col] - bhav[open_col]) / bhav[open_col]
    bhav["turnover"] = bhav[close_col] * bhav["TtlTradgVol"]
    liquid = bhav[bhav["turnover"] >= 10e7].copy()
    # --- Technical indicators ---
    liquid["RSI"] = ta.momentum.RSIIndicator(close=liquid[close_col], window=14).rsi()
    liquid["EMA20"] = ta.trend.ema_indicator(liquid[close_col], window=20)
    liquid["EMA50"] = ta.trend.ema_indicator(liquid[close_col], window=50)
    liquid["ADX"] = ta.trend.ADXIndicator(high=liquid[high_col], low=liquid[low_col], close=liquid[close_col], window=14).adx()
    liquid["VWAP"] = (liquid[close_col] * liquid["TtlTradgVol"]).cumsum() / liquid["TtlTradgVol"].cumsum()
    atr = ta.volatility.AverageTrueRange(high=liquid[high_col], low=liquid[low_col], close=liquid[close_col], window=14)
    liquid["ATR"] = atr.average_true_range()
    # --- Market regime awareness ---
    regime = get_market_regime()
    if regime['trend'] == 'bear':
        rsi_min, rsi_max = 30, 60
    elif regime['trend'] == 'bull':
        rsi_min, rsi_max = 40, 80
    else:
        rsi_min, rsi_max = 35, 70
    # --- Filtering ---
    newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
    filtered = []
    for _, row in liquid.iterrows():
        # --- Fundamental filter ---
        fdata = get_fundamental_data(row["TckrSymb"])
        if fdata['pe'] > 40 or fdata['debt_equity'] > 1 or fdata['promoter_holding'] < 30:
            continue
        # --- Technicals ---
        if pd.isna(row["RSI"]) or pd.isna(row["EMA20"]) or pd.isna(row["EMA50"]) or pd.isna(row["ADX"]) or pd.isna(row["VWAP"]) or pd.isna(row["ATR"]):
            continue
        if not (rsi_min < row["RSI"] < rsi_max):
            continue
        if row["EMA20"] < row["EMA50"]:
            continue
        if row["ADX"] < 15:
            continue
        if row[close_col] < row["VWAP"]:
            continue
        if row["ATR"] <= 0 or row["ATR"] > row[close_col] * 0.2:
            continue
        # --- News/Sentiment filter (as before) ---
        symbol = row["TckrSymb"]
        sentiment = get_news_sentiment_openai(newsapi, symbol)
        if sentiment == 'negative':
            continue
        # --- Real-time price (if available) ---
        realtime_price = get_realtime_price(row["TckrSymb"])
        # --- Order book features ---
        ob_features = get_order_book_features(row["TckrSymb"])
        # --- ML model prediction ---
        features = row.to_dict()  # plus any engineered features
        ml_score = ml_predict_intraday_move(features)
        # --- Candlestick pattern recognition ---
        patterns = detect_candlestick_patterns(row)
        # --- Sector/theme rotation ---
        sector_strength = get_sector_strength(row["TckrSymb"])
        # --- Event-driven trading ---
        has_event = check_event_driven(row["TckrSymb"])
        # --- Correlation/pair trading ---
        pairs = get_correlation_pairs(row["TckrSymb"], liquid["TckrSymb"])
        # --- User feedback loop ---
        feedback = get_user_feedback(row["TckrSymb"])
        # --- Example: Use these features in filtering/scoring (customize as needed) ---
        if ml_score < 0.4:
            continue
        if sector_strength < 0 and not has_event:
            continue
        filtered.append(row)
    picks = pd.DataFrame(filtered)
    # --- Fallback: if <3 picks, fill with top liquid stocks by pct_change (ignore technicals/sentiment) ---
    if picks.shape[0] < 3:
        fallback = liquid.nlargest(3, "pct_change").drop_duplicates(subset=["TckrSymb"])
        picks = pd.concat([picks, fallback]).drop_duplicates(subset=["TckrSymb"]).head(5)
    if picks.empty:
        return picks
    picks = picks.nlargest(5, "pct_change")
    picks = picks.copy()
    # --- Risk management ---
    max_risk_per_trade = 0.01  # 1% of capital
    max_daily_risk = 0.03      # 3% of capital
    capital = 100000  # Example
    # Optionally, use ML score or pattern match as confidence
    picks["confidence"] = picks.apply(lambda row: ml_predict_intraday_move(row.to_dict()), axis=1)
    # Position size: risk per trade / (stop distance), scaled by confidence (min 0.5, max 1.5)
    picks["risk_amt"] = capital * max_risk_per_trade
    picks["stop_dist"] = picks["entry"] - picks["stop"]
    picks["confidence_scale"] = picks["confidence"].clip(0.5, 1.5)
    picks["position_size"] = (picks["risk_amt"] / picks["stop_dist"]) * picks["confidence_scale"]
    # Cap position size to 20% of capital
    picks["position_size"] = picks["position_size"].clip(upper=capital * 0.2)
    # Cap total daily risk
    total_risk = (picks["position_size"] * picks["stop_dist"]).sum()
    if total_risk > capital * max_daily_risk:
        scale = (capital * max_daily_risk) / total_risk
        picks["position_size"] *= scale
    # --- Execution/slippage adjustment ---
    slippage = 0.001  # 0.1%
    picks["entry"] = picks[close_col] * (1 + slippage)
    picks["stop"] = picks["entry"] - picks["ATR"] * 1.5
    picks["target"] = picks["entry"] + np.minimum(picks["ATR"] * 2, picks["entry"] * 0.05)
    # --- Logging for backtesting/monitoring ---
    picks["log_time"] = pd.Timestamp.now()
    return picks.reset_index(drop=True)
