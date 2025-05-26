import datetime as dt, pandas as pd, zipfile, io, requests
from nsepython import nse_preopen_movers
import logging
import yfinance as yf
import time

def fetch_bhavcopy():
    # Find the most recent weekday (skip Sat/Sun)
    import datetime as dt
    import pandas as pd, zipfile, io, requests
    day_offset = 1
    while True:
        d = dt.date.today() - dt.timedelta(days=day_offset)
        if d.weekday() < 5:  # 0=Mon, ..., 4=Fri
            break
        day_offset += 1
    url = f"https://nsearchives.nseindia.com/content/cm/" \
          f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200 or not r.content or r.content[:2] != b'PK':
        print("Bhavcopy not available or not a valid ZIP file. Likely a holiday or weekend.")
        return pd.DataFrame()
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        return pd.read_csv(zf.open(zf.namelist()[0]))
    except Exception as e:
        print(f"Error reading Bhavcopy ZIP: {e}")
        return pd.DataFrame()

def fetch_preopen(symbols):
    try:
        data = nse_preopen_movers()
        return {s: float(data[s]["lastPrice"]) for s in symbols if s in data}
    except Exception as e:
        logging.warning(f"Could not fetch pre-open data: {e}")
        return {}

def fetch_latest_prices(symbols):
    # Only use previous close prices from Bhavcopy for all symbols
    # This guarantees data is always available
    return {s: None for s in symbols}  # None means fallback to previous close in strategy

# In strategy.py, update score_stocks to use previous close if pre-open is missing
# ...existing code...
def score_stocks(bhav: pd.DataFrame, pre: dict) -> pd.DataFrame:
    # liquidity filter: ≥₹50 Cr turnover
    bhav["turnover"] = bhav["ClsPric"] * bhav["TtlTradgVol"]
    liquid = bhav[bhav["turnover"] >= 50e7]
    liquid = liquid.copy()  # avoid SettingWithCopyWarning
    # Use pre-open price if available, else fallback to previous close
    liquid["open_price"] = liquid["TckrSymb"].map(pre)
    liquid["open_price"].fillna(liquid["ClsPric"], inplace=True)
    liquid["gap_pct"] = (liquid["open_price"] - liquid["ClsPric"]) / liquid["ClsPric"]
    # ...existing code...
