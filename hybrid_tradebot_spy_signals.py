# hybrid_tradebot_spy_signals.py

import os
import time
import json
import logging
from io import StringIO
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
VWAP_THRESHOLD = 0.02       # Looser threshold for realistic trading
MAX_CANDLE_SPREAD = 1.0
MIN_VOLUME = 50000
FALLBACK_TIME_ET = 14       # 2 PM ET fallback
PAPER = True
TP_PCT = 0.005              # 0.5% take-profit
SL_PCT = 0.0025             # 0.25% stop-loss
MONITOR_INTERVAL = 10        # seconds (poll interval)
MONITOR_TIMEOUT = 600        # seconds (10 minutes max per trade)

# Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL")

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# Paths
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"
LOG_FILE = "bot.log"

# -------------------- LOGGING --------------------
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

def submit_order(symbol, qty, side_str):
    side_str = side_str.upper()
    if side_str == "BUY":
        side_enum = OrderSide.BUY
    elif side_str == "SELL":
        side_enum = OrderSide.SELL
    else:
        logging.error(f"Invalid side: {side_str}")
        return None

    if not symbol or symbol.strip() == "":
        logging.error("submit_order called with empty symbol; skipping.")
        return None

    req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=TimeInForce.DAY)

    attempt = 0
    max_attempts = 5
    while attempt < max_attempts:
        attempt += 1
        try:
            order = client.submit_order(req)
            logging.info(f"Submitted {side_str} {qty} {symbol}")
            return order
        except Exception as e:
            msg = str(e)
            if "400" in msg or "symbol is required" in msg:
                logging.error(f"Client error on submit: {e} — skipping retries")
                return None
            delay = 2 ** attempt
            logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s")
            time.sleep(delay)
    logging.error(f"Failed to submit order after {max_attempts} attempts")
    return None

def fetch_sheet_signals():
    if not SIGNAL_SHEET_CSV_URL:
        return []
    try:
        resp = requests.get(SIGNAL_SHEET_CSV_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "enabled" in df.columns:
            df = df[df["enabled"].astype(str).str.lower().isin(["true","1","yes","y"])]
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            df = df[df["symbol"].notna() & (df["symbol"]!="")]
        else:
            return []
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch sheet: {e}")
        return []

def monitor_trade(symbol, qty, entry_price):
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)
    start_time = time.time()
    logging.info(f"Monitoring {symbol} entry {entry_price:.2f} TP {tp_price:.2f} SL {sl_price:.2f}")
    while time.time() - start_time < MONITOR_TIMEOUT:
        df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            time.sleep(MONITOR_INTERVAL)
            continue
        last_price = float(df["Close"].iloc[-1])
        logging.info(f"Monitor {symbol} price {last_price:.2f}")
        if last_price >= tp_price:
            submit_order(symbol, qty, "SELL")
            pnl = (tp_price - entry_price)*qty
            trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side":"LONG", "qty":qty,
                     "entry_price":entry_price,"exit_price":tp_price,"pnl":round(pnl,4),
                     "result":"WIN","notes":"Take-profit hit"}
            save_trade(trade)
            logging.info(f"TP hit {trade}")
            return trade
        if last_price <= sl_price:
            submit_order(symbol, qty, "SELL")
            pnl = (sl_price - entry_price)*qty
            trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side":"LONG", "qty":qty,
                     "entry_price":entry_price,"exit_price":sl_price,"pnl":round(pnl,4),
                     "result":"LOSS","notes":"Stop-loss hit"}
            save_trade(trade)
            logging.info(f"SL hit {trade}")
            return trade
        time.sleep(MONITOR_INTERVAL)
    # Timeout, sell at last price
    submit_order(symbol, qty, "SELL")
    pnl = (last_price - entry_price)*qty
    trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side":"LONG", "qty":qty,
             "entry_price":entry_price,"exit_price":last_price,"pnl":round(pnl,4),
             "result":"TIMEOUT","notes":"Monitor timeout exit"}
    save_trade(trade)
    logging.info(f"Timeout exit {trade}")
    return trade

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
            logging.info(f"{symbol} volume too low")
            return None
        if candle_spread > MAX_CANDLE_SPREAD:
            logging.info(f"{symbol} candle spread too big")
            return None
        if abs(price-vwap)/vwap > VWAP_THRESHOLD:
            logging.info(f"{symbol} price-vwap too far")
            return None
        order = submit_order(symbol, qty=1, side_str="BUY")
        if order:
            trade = monitor_trade(symbol, qty=1, entry_price=price)
            return trade
    except Exception as e:
        logging.error(f"Primary trade error for {symbol}: {e}")
    return None

# -------------------- FALLBACK STRATEGY --------------------
def fallback_trade():
    signals = fetch_sheet_signals()
    if not signals:
        logging.info("No fallback signals")
        return None
    for sig in signals:
        try:
            symbol = sig.get("symbol")
            side = sig.get("action","BUY").upper()
            qty = int(sig.get("qty",1))
            order = submit_order(symbol, qty, side)
            if order:
                df = yf.download(symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
                entry_price = float(df["Close"].iloc[-1]) if not df.empty else None
                trade = monitor_trade(symbol, qty, entry_price)
                return trade
        except Exception as e:
            logging.error(f"Fallback trade error: {e}")
    return None

# -------------------- MAIN --------------------
def main():
    logging.info("Checking primary symbols...")
    top_trade = None
    for symbol in SYMBOLS:
        trade = primary_trade(symbol)
        if trade:
            top_trade = trade
            break
    # If past fallback time ET
    if not top_trade and now_et().hour >= FALLBACK_TIME_ET:
        logging.info("Checking fallback sheet signals...")
        top_trade = fallback_trade()
    # Guarantee at least 1 trade
    if not top_trade:
        logging.info("No primary/fallback trade executed; forcing top candidate")
        # force first symbol
        forced_symbol = SYMBOLS[0]
        order = submit_order(forced_symbol, 1, "BUY")
        if order:
            df = yf.download(forced_symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
            entry_price = float(df["Close"].iloc[-1]) if not df.empty else None
            top_trade = monitor_trade(forced_symbol, 1, entry_price)
    if top_trade:
        logging.info(f"Trade executed: {top_trade}")
    else:
        logging.info("No trade executed today.")

if __name__ == "__main__":
    main()
