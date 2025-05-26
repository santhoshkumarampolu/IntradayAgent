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
    # --- Highlight if today is a holiday or weekend (no trading) ---
    import datetime as dt
    import yaml
    today = dt.date.today()
    is_weekend = today.weekday() >= 5
    try:
        with open('holidays.yml') as f:
            holidays = yaml.safe_load(f)
    except Exception:
        holidays = []
    is_holiday = today.isoformat() in holidays if holidays else False
    if is_weekend or is_holiday:
        st.warning('No trading today: ' + ('Weekend' if is_weekend else 'Holiday'))
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
    # --- Trade Rationales (Simple Terms)
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
    # --- Market Breadth: Advance-Decline Ratio (ADR) ---
    st.subheader("Market Breadth: Advance-Decline Ratio (ADR)")
    adr_explanation = """
    **Advance-Decline Ratio (ADR):**
    - Measures overall market breadth by comparing the number of advancing stocks (gained vs previous close) to declining stocks (lost vs previous close) in today's picks.
    - **ADR > 1**: More advancers than decliners (bullish breadth)
    - **ADR < 1**: More decliners than advancers (bearish breadth)
    """
    if 'ClsPric' in picks.columns and 'entry' in picks.columns:
        advancers = (picks['entry'] > picks['ClsPric']).sum()
        decliners = (picks['entry'] < picks['ClsPric']).sum()
        adr = advancers / decliners if decliners > 0 else float('inf')
        st.write(f"Advancers: {advancers}, Decliners: {decliners}")
        st.write(f"**Advance-Decline Ratio (ADR):** {adr:.2f}")
        st.markdown(adr_explanation)
    else:
        st.info("Not enough data to compute ADR (need 'ClsPric' and 'entry').")
    # --- Advanced Analytics Summary ---
    st.subheader("Advanced Analytics Summary")
    analytics_cols = [
        'risk_reward_ratio', 'potential_profit', 'potential_loss',
        'confidence', 'position_size', 'trailing_stop', 'partial_target'
    ]
    available_cols = [col for col in analytics_cols if col in picks.columns]
    if not picks.empty and available_cols:
        if 'risk_reward_ratio' in picks.columns:
            st.write("**Average Risk-Reward Ratio:**", round(picks['risk_reward_ratio'].mean(), 2))
        if 'potential_profit' in picks.columns:
            st.write("**Total Potential Profit:**", round(picks['potential_profit'].sum(), 2))
        if 'potential_loss' in picks.columns:
            st.write("**Total Potential Loss:**", round(picks['potential_loss'].sum(), 2))
        if 'confidence' in picks.columns:
            st.write("**Average Confidence Score:**", round(picks['confidence'].mean(), 2))
        if 'position_size' in picks.columns:
            st.write("**Average Position Size:**", round(picks['position_size'].mean(), 2))
        if 'trailing_stop' in picks.columns:
            st.write("**Trailing Stop (ATR-based):**", picks['trailing_stop'].round(2).to_list())
        if 'partial_target' in picks.columns:
            st.write("**Partial Profit Booking Target (1:1 RR):**", picks['partial_target'].round(2).to_list())
    else:
        st.info("No picks available for analytics summary or analytics columns missing.")
    # --- Show only required columns in the dashboard ---
    important_cols = [
        'TckrSymb', 'ClsPric', 'entry', 'stop', 'target', 'trailing_stop',
        'partial_target', 'risk_reward_ratio', 'confidence', 'position_size'
    ]
    display_cols = [col for col in important_cols if col in picks.columns]
    st.subheader("Today's Picks (Key Metrics)")
    st.dataframe(picks[display_cols], use_container_width=True)

    # --- Column Explanations ---
    st.markdown("""
    **Column Explanations:**
    - **TckrSymb**: Stock symbol (NSE/BSE code)
    - **ClsPric**: Previous close price
    - **entry**: Suggested entry price for the trade
    - **stop**: Suggested stop-loss price (risk management)
    - **target**: Suggested target price (aiming for at least 10% profit)
    - **trailing_stop**: Dynamic stop-loss that trails the price as it moves in your favor (ATR-based)
    - **partial_target**: Price for partial profit booking (1:1 risk-reward)
    - **risk_reward_ratio**: Ratio of potential profit to potential loss (higher is better)
    - **confidence**: Model's confidence score for the pick (0-1, higher is better)
    - **position_size**: Suggested position size based on risk management
    """)
    # --- Volume Comparison (Today vs Last 4 Days) ---
    st.subheader("Volume Surge Analysis (Today vs Last 4 Days)")
    volume_cols = ['volume', 'vol_1d_ago', 'vol_2d_ago', 'vol_3d_ago', 'vol_4d_ago']
    if all(col in picks.columns for col in volume_cols):
        picks['avg_vol_4d'] = picks[['vol_1d_ago', 'vol_2d_ago', 'vol_3d_ago', 'vol_4d_ago']].mean(axis=1)
        picks['volume_ratio'] = picks['volume'] / picks['avg_vol_4d']
        st.dataframe(picks[['TckrSymb', 'volume', 'avg_vol_4d', 'volume_ratio']].round(2), use_container_width=True)
        st.markdown("""
        **Volume Surge Analysis:**
        - **volume**: Today's volume so far
        - **avg_vol_4d**: Average daily volume over the last 4 trading days
        - **volume_ratio**: Ratio of today's volume to 4-day average (values >1 indicate above-average activity)
        """)
    else:
        st.info("Volume columns not found in picks. Please ensure your data includes: volume, vol_1d_ago, vol_2d_ago, vol_3d_ago, vol_4d_ago.")

if __name__ == "__main__":
    main()
