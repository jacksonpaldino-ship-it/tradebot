# hybrid_tradebot_spy_signals.py

import os
import time
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -------------------- CONFIG --------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
VWAP_THRESHOLD = 0.02         # loosened threshold for guaranteed trading
MAX_CANDLE_SPREAD = 1.0       # loosened max candle spread
MIN_VOLUME = 10000
GUARANTEE_TRADE = True        # force 1 trade per day
MONITOR_INTERVAL = 10         # seconds between monitoring stop-loss/TP
TP_PERCENT = 0.5 / 100        # 0.5% take profit
SL_PERCENT = 0.3 / 100        # 0.3% stop loss

# Alpaca API from secrets
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL")
PAPER = True

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

# -------------------- HELPERS --------------------
def now_et():
    return datetime.now(pytz.timezone("US/Eastern"))

def save_trade(trade):
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
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        df = df[df.get("enabled", True) == True]
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
        last = df.iloc[-1]
        price = float(last["Close"])
        vwap = float(last["VWAP"])
        volume = float(last["Volume"])
        candle_spread = float(last["High"]) - float(last["Low"])

        if volume < MIN_VOLUME:
            logging.info(f"{symbol} volume too low, skipping.")
            return None
        if candle_spread > MAX_CANDLE_SPREAD:
            logging.info(f"{symbol} candle spread too big, skipping.")
            return None
        if abs(price - vwap)/vwap > VWAP_THRESHOLD:
            logging.info(f"{symbol} price-vwap {abs(price-vwap)/vwap:.4f} too far, skipping.")
            return None

        order = submit_order(symbol, qty=1, side="BUY")
        if order:
            trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side":"BUY", "qty":1}
            save_trade(trade)
            monitor_trade(symbol, price)
            return trade
    except Exception as e:
        logging.error(f"Primary trade error for {symbol}: {e}")
    return None

# -------------------- MONITOR --------------------
def monitor_trade(symbol, entry_price):
    tp = entry_price*(1+TP_PERCENT)
    sl = entry_price*(1-SL_PERCENT)
    logging.info(f"Monitoring {symbol} entry {entry_price} TP {tp} SL {sl}")
    while True:
        df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            time.sleep(MONITOR_INTERVAL)
            continue
        last_price = float(df["Close"].iloc[-1])
        logging.info(f"{symbol} price {last_price}")
        if last_price >= tp:
            logging.info(f"{symbol} reached TP {tp}, selling.")
            submit_order(symbol, 1, "SELL")
            break
        elif last_price <= sl:
            logging.info(f"{symbol} hit SL {sl}, selling.")
            submit_order(symbol, 1, "SELL")
            break
        time.sleep(MONITOR_INTERVAL)

# -------------------- FALLBACK --------------------
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
            if not symbol:
                continue
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

# -------------------- MAIN --------------------
def main():
    logging.info("Starting hybrid_tradebot_advanced")
    trade_executed = False
    for symbol in SYMBOLS:
        trade = primary_trade(symbol)
        if trade:
            trade_executed = True
            break

    if not trade_executed:
        trade = fallback_trade()
        if trade:
            trade_executed = True

    if not trade_executed and GUARANTEE_TRADE:
        # Force first symbol if nothing triggers
        symbol = SYMBOLS[0]
        logging.info(f"Forcing trade for {symbol} due to guarantee flag.")
        order = submit_order(symbol, 1, "BUY")
        if order:
            trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side":"BUY", "qty":1}
            save_trade(trade)
            monitor_trade(symbol, float(yf.download(symbol, period="1d", interval="1m", progress=False)["Close"].iloc[-1]))

    logging.info("Run complete.")

if __name__ == "__main__":
    main()
