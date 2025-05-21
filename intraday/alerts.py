import os
import time
import pandas as pd
import yagmail
from dotenv import load_dotenv
import yfinance as yf

load_dotenv()

ALERT_INTERVAL = 300  # seconds between checks (5 minutes)
MARKET_START = "09:15"
MARKET_END = "15:30"

PICKS_DB = "data/agent.db"

# Helper to send alert email
def send_alert_email(stock, event, price):
    yag = yagmail.SMTP(user=os.getenv("EMAIL_USER"),
                       password=os.getenv("EMAIL_PASS"),
                       host=os.getenv("SMTP_HOST"), port=int(os.getenv("SMTP_PORT")))
    subject = f"\U0001F514 Intraday Alert: {stock} {event} at {price:.2f}"
    body = f"Alert: {stock} has {event} at price {price:.2f}.\n\n--Your Trading Bot"
    yag.send(to=os.getenv("EMAIL_TO"), subject=subject, contents=body)

def get_live_price(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        return ticker.fast_info['lastPrice']
    except Exception:
        return None

def load_today_picks():
    import sqlite3
    today = pd.Timestamp.now().normalize()
    with sqlite3.connect(PICKS_DB) as cx:
        try:
            df = pd.read_sql("SELECT * FROM picks", cx)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] == today]
            return df
        except Exception:
            return pd.DataFrame()

def main():
    picks = load_today_picks()
    if picks.empty:
        print("No picks for today.")
        return
    # Track alert state to avoid duplicate alerts
    alerted = {s: {'entry': False, 'target': False, 'stop': False} for s in picks['TckrSymb']}
    while True:
        now = pd.Timestamp.now()
        if not (now.strftime("%H:%M") >= MARKET_START and now.strftime("%H:%M") <= MARKET_END):
            print("Market closed. Exiting alert monitor.")
            break
        for _, row in picks.iterrows():
            symbol = row['TckrSymb']
            entry = row.get('entry')
            target = row.get('target')
            stop = row.get('stop')
            price = get_live_price(symbol)
            if price is None:
                continue
            # Entry alert
            if not alerted[symbol]['entry'] and price >= entry:
                send_alert_email(symbol, 'reached entry', price)
                alerted[symbol]['entry'] = True
            # Target alert
            if not alerted[symbol]['target'] and price >= target:
                send_alert_email(symbol, 'hit target', price)
                alerted[symbol]['target'] = True
            # Stop-loss alert
            if not alerted[symbol]['stop'] and price <= stop:
                send_alert_email(symbol, 'hit stop-loss', price)
                alerted[symbol]['stop'] = True
        time.sleep(ALERT_INTERVAL)

if __name__ == "__main__":
    main()
