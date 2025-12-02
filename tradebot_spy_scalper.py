# hybrid_tradebot_spy_signals.py

import os
import time
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -------------------- CONFIG --------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]  # Primary symbols to monitor
VWAP_THRESHOLD = 0.0015    # Example loosened threshold
MAX_CANDLE_SPREAD = 0.5    # Example loosened threshold
MIN_VOLUME = 50000         # Example minimum volume
FALLBACK_TIME_ET = 14      # ET hour to start checking Google Sheet if no primary trade
PAPER = True               # True for Alpaca paper trading

# Alpaca API setup from secrets
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL")

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# Paths
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
LOG_FILE = "bot.log"

# -------------------- LOGGING --------------------
import logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logging.info("Bot started.")

# -------------------- HELPER FUNCTIONS --------------------
def now_et():
    return datetime.now(pytz.timezone("US/Eastern"))

def save_trade(trade):
    # Append trade to CSV
    df = pd.DataFrame([trade])
    if os.path.exists(TRADES_CSV):
        df.to_csv(TRADES_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(TRADES_CSV, index=False)

    # Update stats
    stats = {}
    if os.path.exists(STATS_JSON):
        with open(STATS_JSON, "r") as f:
            stats = json.load(f)

    symbol = trade["symbol"]
    if symbol not in stats:
        stats[symbol] = {"wins":0,"losses":0,"trades":0}

    stats[symbol]["trades"] += 1
    if trade.get("result") == "WIN":
        stats[symbol]["wins"] += 1
    elif trade.get("result") == "LOSS":
        stats[symbol]["losses"] += 1

    with open(STATS_JSON,"w") as f:
        json.dump(stats,f, indent=2)

def submit_order(symbol, qty, side):
    try:
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side),
            time_in_force=TimeInForce.DAY
        )
        order = client.submit_order(order_data)
        logging.info(f"Order submitted: {symbol} {side} {qty}")
        return order
    except Exception as e:
        logging.error(f"Order failed: {e}")
        return None

def fetch_google_sheet_signals():
    try:
        resp = requests.get(SIGNAL_SHEET_CSV_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(resp.text))
        df = df[df.get("enabled", True)==True]  # Only enabled signals
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch Google Sheet: {e}")
        return []

# -------------------- PRIMARY STRATEGY --------------------
def primary_trade(symbol):
    try:
        df = yf.download(symbol, period="3d", interval="5m", progress=False, auto_adjust=True)
        if df.empty:
            logging.info(f"No data for {symbol}")
            return None

        df["VWAP"] = (df["Close"]*df["Volume"]).cumsum() / df["Volume"].cumsum()

        last_row = df.iloc[-1]
        price = float(last_row["Close"])
        vwap = float(last_row["VWAP"])
        volume = float(last_row["Volume"])
        candle_spread = float(last_row["High"]) - float(last_row["Low"])

        if volume < MIN_VOLUME:
            logging.info(f"{symbol} volume too low, skipping.")
            return None
        if candle_spread > MAX_CANDLE_SPREAD:
            logging.info(f"{symbol} candle spread too big, skipping.")
            return None
        if abs(price - vwap)/vwap > VWAP_THRESHOLD:
            logging.info(f"{symbol} price too far from VWAP, skipping.")
            return None

        # Place order (BUY only for simplicity)
        order = submit_order(symbol, qty=1, side="BUY")
        if order:
            trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side":"BUY", "qty":1}
            save_trade(trade)
            return trade
    except Exception as e:
        logging.error(f"Primary trade error for {symbol}: {e}")
    return None

# -------------------- FALLBACK STRATEGY --------------------
def fallback_trade():
    signals = fetch_google_sheet_signals()
    if not signals:
        logging.info("No fallback signals.")
        return None

    for sig in signals:
        try:
            symbol = sig.get("symbol")
            action = sig.get("action","BUY").upper()
            qty = int(sig.get("qty",1))
            order = submit_order(symbol, qty, action)
            if order:
                trade = {
                    "timestamp": now_et().isoformat(),
                    "symbol": symbol,
                    "side": action,
                    "qty": qty,
                    "notes": sig.get("notes","fallback")
                }
                save_trade(trade)
                return trade
        except Exception as e:
            logging.error(f"Fallback trade error: {e}")
    return None

# -------------------- MAIN LOOP --------------------
def main():
    logging.info("Checking primary symbols...")
    for symbol in SYMBOLS:
        trade = primary_trade(symbol)
        if trade:
            logging.info(f"Primary trade executed: {trade}")
            return

    # If past fallback time ET, check Google Sheet
    current_hour = now_et().hour
    if current_hour >= FALLBACK_TIME_ET:
        logging.info("Checking fallback signals...")
        trade = fallback_trade()
        if trade:
            logging.info(f"Fallback trade executed: {trade}")
            return

    logging.info("No trade executed today.")

if __name__ == "__main__":
    main()
