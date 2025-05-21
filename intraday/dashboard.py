import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime

def load_today_picks():
    db_path = os.path.join('data', 'agent.db')
    today = pd.Timestamp.now().normalize()
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as cx:
        try:
            df = pd.read_sql("SELECT * FROM picks", cx)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] == today]
            return df
        except Exception:
            return pd.DataFrame()

def main():
    st.set_page_config(page_title="Intraday Trading Dashboard", layout="wide")
    st.title("📈 Intraday Trading Dashboard")
    # --- Add button to run today's picks ---
    if st.button("Run Today's Picks (Refresh)"):
        import subprocess
        result = subprocess.run(["python", "-m", "intraday.main"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("Today's picks generated!")
        else:
            st.error(f"Error running agent: {result.stderr}")
    picks = load_today_picks()
    if picks.empty:
        st.info("No picks for today. Run the agent to generate picks.")
        return
    st.subheader("Today's Picks")
    st.dataframe(picks, use_container_width=True)
    # --- Trade Rationales (moved from mailer.py) ---
    st.subheader("Trade Rationales (Simple Terms)")
    for _, row in picks.iterrows():
        rationale = f"{row['TckrSymb']}: "
        rationale += f"RSI={row.get('RSI', 'N/A'):.1f}, " if 'RSI' in row else ''
        rationale += f"EMA20={'above' if row.get('EMA20',0) > row.get('EMA50',0) else 'below'} EMA50, " if 'EMA20' in row and 'EMA50' in row else ''
        rationale += f"ADX={row.get('ADX', 'N/A'):.1f}, " if 'ADX' in row else ''
        rationale += f"Sector strength={'strong' if row.get('confidence',0)>0.7 else 'neutral/weak'}, " if 'confidence' in row else ''
        rationale += f"ML confidence={row.get('confidence', 'N/A'):.2f}, " if 'confidence' in row else ''
        rationale += f"Recent event detected, " if row.get('has_event', False) else ''
        rationale += f"No negative news."
        st.markdown(rationale.strip().rstrip(','))
    # --- New Features ---
    # 1. Live Price Checker
    st.subheader("Live Price Checker")
    symbol = st.text_input("Enter NSE Symbol (e.g., TCS, INFY):")
    if symbol:
        import yfinance as yf
        try:
            price = yf.Ticker(symbol + ".NS").fast_info['lastPrice']
            st.success(f"Live price for {symbol.upper()}: ₹{price}")
        except Exception:
            st.error("Could not fetch live price.")
    # 2. Manual Feedback/Override
    st.subheader("Manual Feedback / Override")
    st.info("You can upload a CSV named user_feedback.csv with columns: symbol, feedback, override_entry, override_stop, override_target.")
    uploaded = st.file_uploader("Upload user_feedback.csv", type=["csv"])
    if uploaded:
        df_feedback = pd.read_csv(uploaded)
        st.dataframe(df_feedback)
        df_feedback.to_csv("user_feedback.csv", index=False)
        st.success("Feedback uploaded and saved.")
    # 3. Analytics (if trade log exists)
    st.subheader("Post-Trade Analytics")
    trade_log_path = os.path.join('data', 'trade_log.csv')
    if os.path.exists(trade_log_path):
        trade_log = pd.read_csv(trade_log_path)
        st.dataframe(trade_log)
        st.write("Win Rate:", (trade_log['PnL'] > 0).mean())
        st.write("Average PnL:", trade_log['PnL'].mean())
        st.write("Max Drawdown:", (trade_log['PnL'].cumsum().cummax() - trade_log['PnL'].cumsum()).max())
    else:
        st.info("No trade log found. Analytics will appear after trades are executed and logged.")

if __name__ == "__main__":
    main()
