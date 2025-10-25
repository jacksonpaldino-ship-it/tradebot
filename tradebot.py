import os
import time
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime
import pytz

# === CONFIG ===
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"  # use live URL for live trading
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "PLTR", "CRSP"]
MAX_TRADES_PER_SYMBOL = 2  # limit per day

# Market hours
MARKET_TZ = pytz.timezone("US/Eastern")
MARKET_OPEN = datetime.now(MARKET_TZ).replace(hour=9, minute=30, second=0, microsecond=0)
MARKET_CLOSE = datetime.now(MARKET_TZ).replace(hour=16, minute=0, second=0, microsecond=0)

# === INIT ===
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")
trade_log = {}

def is_market_open():
    now = datetime.now(MARKET_TZ)
    return MARKET_OPEN <= now <= MARKET_CLOSE

def get_bars(symbol):
    try:
        df = yf.download(symbol, period="60d", interval="15m", progress=False)
        df["sma_short"] = df["Close"].rolling(window=10).mean()
        df["sma_long"] = df["Close"].rolling(window=50).mean()
        return df
    except Exception as e:
        print(f"Failed bars for {symbol}: {e}")
        return None

def get_signal(df):
    try:
        df["sma_short"] = df["Close"].rolling(window=10).mean()
        df["sma_long"] = df["Close"].rolling(window=30).mean()

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        latest_short = float(latest["sma_short"].iloc[0] if hasattr(latest["sma_short"], "iloc") else latest["sma_short"])
        latest_long  = float(latest["sma_long"].iloc[0]  if hasattr(latest["sma_long"], "iloc")  else latest["sma_long"])
        prev_short   = float(prev["sma_short"].iloc[0]   if hasattr(prev["sma_short"], "iloc")   else prev["sma_short"])
        prev_long    = float(prev["sma_long"].iloc[0]    if hasattr(prev["sma_long"], "iloc")    else prev["sma_long"])

        if latest_short > latest_long and prev_short <= prev_long:
            return "BUY"
        elif latest_short < latest_long and prev_short >= prev_long:
            return "SELL"
        else:
            return "HOLD"

    except Exception as e:
        print(f"Error generating signal: {e}")
        return "HOLD"

def place_trade(symbol, side):
    try:
        position_qty = 1  # fixed position size
        order = api.submit_order(
            symbol=symbol,
            qty=position_qty,
            side=side.lower(),
            type="market",
            time_in_force="gtc",
        )
        print(f"🟩 Order submitted: {order.side.upper()} {order.qty} {symbol} @ market")
    except Exception as e:
        print(f"❌ Trade failed for {symbol}: {e}")

def main():
    run_time = datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== Tradebot run {run_time} ===")

    if not is_market_open():
        print("Market is closed. Exiting.")
        return

    for symbol in SYMBOLS:
        print(f"\nChecking {symbol}...")

        df = get_bars(symbol)
        signal = get_signal(df)

        print(f"[{datetime.now(MARKET_TZ).strftime('%H:%M:%S')}] {symbol}: Signal = {signal}")

        # Enforce max trades per day
        if symbol not in trade_log:
            trade_log[symbol] = {"count": 0, "last_side": None}

        if trade_log[symbol]["count"] >= MAX_TRADES_PER_SYMBOL:
            print(f"⏸️ Max trades reached for {symbol}")
            continue

        if signal == "BUY" and trade_log[symbol]["last_side"] != "BUY":
            place_trade(symbol, "BUY")
            trade_log[symbol]["count"] += 1
            trade_log[symbol]["last_side"] = "BUY"

        elif signal == "SELL" and trade_log[symbol]["last_side"] != "SELL":
            place_trade(symbol, "SELL")
            trade_log[symbol]["count"] += 1
            trade_log[symbol]["last_side"] = "SELL"

        else:
            print(f"➖ No trade action for {symbol}")

    print("\n✅ Trade check complete. Exiting cleanly.\n")

if __name__ == "__main__":
    main()
