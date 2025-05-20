import datetime as dt, pandas as pd, zipfile, io, requests
from nsepython import nse_preopen_movers
import logging
import yfinance as yf

def fetch_bhavcopy():
    d = dt.date.today() - dt.timedelta(days=1)
    url = f"https://nsearchives.nseindia.com/content/cm/" \
          f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    return pd.read_csv(zf.open(zf.namelist()[0]))

def fetch_preopen(symbols):
    try:
        data = nse_preopen_movers()
        return {s: float(data[s]["lastPrice"]) for s in symbols if s in data}
    except Exception as e:
        logging.warning(f"Could not fetch pre-open data: {e}")
        return {}

def fetch_latest_prices(symbols):
    # yfinance expects NSE symbols as 'SYMBOL.NS'
    yf_symbols = [s + '.NS' for s in symbols]
    data = yf.download(yf_symbols, period='1d', interval='1m', progress=False, group_by='ticker')
    latest_prices = {}
    for s in symbols:
        try:
            price = data[(s + '.NS')]['Close'].dropna()[-1]
            latest_prices[s] = float(price)
        except Exception:
            continue
    return latest_prices
