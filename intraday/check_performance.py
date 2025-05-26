import yfinance as yf
import pandas as pd

# List of NSE symbols and their Yahoo Finance codes
symbols = [
    ("ABINFRA", "ABINFRA.NS"),
    ("TTML", "TTML.NS"),
    ("JKTYRE", "JKTYRE.NS"),
    ("ECOSMOBLTY", "ECOSMOBLTY.NS"),
    ("TRIDENT", "TRIDENT.NS"),
    ("BANCOINDIA", "BANCOINDIA.NS"),
    ("QPOWER", "QPOWER.NS"),
    ("KRISHANA", "KRISHANA.NS"),
]

results = []
for name, code in symbols:
    try:
        df = yf.download(code, start="2025-05-22", end="2025-05-23", interval="1d")
        if not df.empty:
            row = df.iloc[0]
            results.append({
                "symbol": name,
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "close": row["Close"]
            })
        else:
            results.append({"symbol": name, "open": None, "high": None, "low": None, "close": None})
    except Exception as e:
        results.append({"symbol": name, "open": None, "high": None, "low": None, "close": None, "error": str(e)})

# Print results as a table
print(pd.DataFrame(results))
