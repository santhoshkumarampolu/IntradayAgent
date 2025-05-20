# Intraday Trading Agent

Automated agent for selecting and emailing daily intraday stock picks for NSE India, based on liquidity and price gap strategies.

## Features

- Fetches latest NSE Bhavcopy and live prices
- Scores and ranks stocks using volume and gap filters
- Emails top picks with entry, stop, and target levels
- Logs picks to a local SQLite database for later analysis
- Skips weekends and NSE holidays (see [`holidays.yml`](holidays.yml))

## Requirements

- Python 3.12+
- See [`requirements.txt`](requirements.txt) for dependencies

## Setup

1. **Clone the repository**

   ```sh
   git clone <repo-url>
   cd IntraDayTradingAgent
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy `.env.example` to `.env` and fill in your email credentials:

   ```sh
   cp .env.example .env
   ```

   Edit `.env` with your email and SMTP details.

4. **Run the agent**

   ```sh
   python -m intraday.main
   ```

## Scheduling

This project includes a [GitHub Actions workflow](.github/workflows%20/intraday.yml) to run the agent automatically every trading day at 8:15 AM UTC.

## File Structure

- [`intraday/main.py`](intraday/main.py): Orchestrates the workflow
- [`intraday/markets.py`](intraday/markets.py): Fetches market data
- [`intraday/strategy.py`](intraday/strategy.py): Stock scoring logic
- [`intraday/mailer.py`](intraday/mailer.py): Email sending
- [`holidays.yml`](holidays.yml): List of NSE holidays

## Disclaimer

This project is for educational and personal use only. Not investment advice.

---

© 2024
