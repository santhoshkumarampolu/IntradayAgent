from intraday import markets, strategy, mailer
import pandas as pd, sqlite3, datetime as dt, pathlib, yaml

# Orchestrator script

def is_trading_day():
    if dt.date.today().weekday() >= 5:          # Sat/Sun
        return False
    holidays = yaml.safe_load(open("holidays.yml"))
    return dt.date.today().isoformat() not in holidays

def run():
    if not is_trading_day():
        return

    bhav = markets.fetch_bhavcopy()
    if bhav.empty:
        print("Bhavcopy DataFrame is empty. Skipping downstream processing.")
        return bhav
    # Use yfinance for latest prices instead of pre-open
    latest = markets.fetch_latest_prices(bhav.TckrSymb.to_list())

    picks = strategy.score_stocks(bhav, latest)
    mailer.send_email(picks)

    # log result for later P/L evaluation
    pathlib.Path("data").mkdir(exist_ok=True)
    with sqlite3.connect("data/agent.db") as cx:
        # Drop the picks table if it exists to avoid schema mismatch
        cx.execute("DROP TABLE IF EXISTS picks;")
        # Save all columns from picks, not just a subset, for dashboard analytics
        picks_to_save = picks.copy()
        picks_to_save['date'] = dt.date.today()
        picks_to_save.to_sql("picks", cx, if_exists="replace", index=False)

if __name__ == "__main__":
    run()
