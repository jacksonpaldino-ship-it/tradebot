import os
import json
import requests
from datetime import datetime, timedelta
from alpaca_trade_api import REST

# -----------------------------------------
# CONFIG
# -----------------------------------------
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"   # Change if live trading

api = REST(API_KEY, API_SECRET, BASE_URL)

TICKERS = ["SPY", "QQQ", "IWM", "DIA"]
LAST_TRADE_FILE = "last_trade.json"


# -----------------------------------------
# Load & Save Persistent Data
# -----------------------------------------
def load_last_trade():
    if not os.path.exists(LAST_TRADE_FILE):
        return {}
    with open(LAST_TRADE_FILE, "r") as f:
        return json.load(f)


def save_last_trade(data):
    with open(LAST_TRADE_FILE, "w") as f:
        json.dump(data, f)


last_trade = load_last_trade()


# -----------------------------------------
# Fetch Latest Price + VWAP
# -----------------------------------------
def fetch_price_vwap(ticker):
    try:
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest"
        headers = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET}
        q = requests.get(url, headers=headers).json()

        price = q["quote"]["ap"]   # ask price
        bid = q["quote"]["bp"]     # bid price
        mid = (price + bid) / 2

        # VWAP endpoint
        vwap_url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars/latest?timeframe=1Min"
        vwap_data = requests.get(vwap_url, headers=headers).json()
        vwap = vwap_data["bar"]["vwap"]

        return mid, vwap

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, None


# -----------------------------------------
# Signal Calculation
# -----------------------------------------
def compute_signal_score(price, vwap):
    gap = abs(price - vwap)
    spread = max(0.01, price * 0.001)  # Hard minimum to avoid divide-by-zero
    score = max(0, 1 - (gap / spread))
    return score


# -----------------------------------------
# Trade Conditions
# -----------------------------------------
def should_trade(ticker, price, vwap):
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Prevent multiple trades per ticker per day
    if last_trade.get(ticker) == today:
        print(f"Already traded {ticker} today, skipping.")
        return False

    score = compute_signal_score(price, vwap)
    print(f"{ticker} score: {score:.2f}")

    # Require strong score (stable threshold)
    if score < 0.65:
        return False

    return True


# -----------------------------------------
# Execute Buy
# -----------------------------------------
def place_buy(ticker, qty):
    try:
        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        print(f"Submitted BUY {qty} {ticker} at {datetime.utcnow()}")
        return True
    except Exception as e:
        print(f"Order error for {ticker}: {e}")
        return False


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------
def run_bot():
    print(f"Starting hybrid_tradebot at {datetime.utcnow()}")

    global last_trade

    for ticker in TICKERS:
        price, vwap = fetch_price_vwap(ticker)

        if price is None or vwap is None:
            print(f"Skipping {ticker}, missing data.")
            continue

        if should_trade(ticker, price, vwap):
            if place_buy(ticker, 1):
                last_trade[ticker] = datetime.utcnow().strftime("%Y-%m-%d")

    save_last_trade(last_trade)
    print("Run complete.")


# -----------------------------------------
# Execute
# -----------------------------------------
if __name__ == "__main__":
    run_bot()
