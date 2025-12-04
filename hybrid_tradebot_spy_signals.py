# hybrid_tradebot_ready.py

import os
import time
import json
import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# -------------------- CONFIG --------------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MIN_VOLUME = 50000
VWAP_WEIGHT = 0.4
VOL_WEIGHT = 0.4
SPREAD_WEIGHT = 0.2
FALLBACK_TIME_ET = 14  # 2PM ET for fallback
GUARANTEE_TRADE = True
PAPER = True

# Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
SIGNAL_SHEET_CSV_URL = os.getenv("SIGNAL_SHEET_CSV_URL")

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# Logging and paths
logging.basicConfig(filename="bot.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
TRADES_CSV = "trades.csv"
STATS_JSON = "trade_stats.json"

logging.info("Starting hybrid_tradebot_ready")

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
        stats[symbol] = {"wins":0, "losses":0, "trades":0}

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
        logging.error(f"Order failed {symbol} {side}: {e}")
        return None

def fetch_google_sheet_signals():
    try:
        resp = requests.get(SIGNAL_SHEET_CSV_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(resp.text)) if hasattr(pd.compat,'StringIO') else pd.read_csv(pd.io.common.StringIO(resp.text))
        df = df[df.get("enabled", True)==True]
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch Google Sheet: {e}")
        return []

# -------------------- SCORING --------------------
def score_symbol(symbol):
    try:
        df = yf.download(symbol, period="3d", interval="5m", progress=False, auto_adjust=True)
        if df.empty: return None

        df["VWAP"] = (df["Close"]*df["Volume"]).cumsum() / df["Volume"].cumsum()
        last = df.iloc[-1]

        price = float(last["Close"])
        vwap = float(last["VWAP"])
        volume = float(last["Volume"])
        spread = float(last["High"]) - float(last["Low"])

        if volume < MIN_VOLUME:
            return None

        vw_pct = abs(price - vwap)/vwap
        spread_threshold = float(df["High"].max() - df["Low"].min())
        vol_norm = min(volume/100000,1.0)

        score = float((1 - vw_pct)*VWAP_WEIGHT + vol_norm*VOL_WEIGHT + (1 - spread/spread_threshold)*SPREAD_WEIGHT)
        return {"symbol": symbol, "score": score, "price": price, "vwap": vwap, "volume": volume, "spread": spread}
    except Exception as e:
        logging.error(f"Error scoring {symbol}: {e}")
        return None

# -------------------- PRIMARY TRADE --------------------
def primary_trade():
    candidates = []
    for sym in SYMBOLS:
        s = score_symbol(sym)
        if s: candidates.append(s)

    if not candidates: return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]
    logging.info(f"Top candidate: {top['symbol']} score {top['score']:.4f}")

    # Adaptive filter
    if abs(top["price"] - top["vwap"])/top["vwap"] > 0.02:  # 2% max deviation
        logging.info(f"{top['symbol']} price-vwap too far, forcing if guarantee")
        if not GUARANTEE_TRADE:
            return None

    order = submit_order(top["symbol"], 1, "BUY")
    if order:
        trade = {"timestamp": now_et().isoformat(), "symbol": top["symbol"], "side":"BUY", "qty":1}
        save_trade(trade)
        logging.info(f"Primary trade executed: {trade}")
        return trade
    return None

# -------------------- FALLBACK --------------------
def fallback_trade():
    signals = fetch_google_sheet_signals()
    if not signals:
        logging.info("No fallback signals")
        return None

    for sig in signals:
        try:
            symbol = sig.get("symbol")
            action = sig.get("action","BUY").upper()
            qty = int(sig.get("qty",1))
            if not symbol: continue
            order = submit_order(symbol, qty, action)
            if order:
                trade = {"timestamp": now_et().isoformat(), "symbol": symbol, "side": action, "qty": qty, "notes":"fallback"}
                save_trade(trade)
                logging.info(f"Fallback trade executed: {trade}")
                return trade
        except Exception as e:
            logging.error(f"Fallback trade error: {e}")
    return None

# -------------------- MAIN --------------------
def main():
    logging.info(f"Run start ET {now_et().isoformat()}")
    trade = primary_trade()
    if trade: return

    # Fallback after certain time
    if now_et().hour >= FALLBACK_TIME_ET:
        trade = fallback_trade()
        if trade: return

    # Guarantee one trade if enabled
    if GUARANTEE_TRADE:
        logging.info("No primary/fallback trade executed; forcing top candidate")
        candidates = []
        for sym in SYMBOLS:
            s = score_symbol(sym)
            if s: candidates.append(s)
        if candidates:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            top = candidates[0]
            order = submit_order(top["symbol"],1,"BUY")
            if order:
                trade = {"timestamp": now_et().isoformat(), "symbol": top["symbol"], "side":"BUY", "qty":1, "notes":"guaranteed"}
                save_trade(trade)
                logging.info(f"Guaranteed trade executed: {trade}")

if __name__ == "__main__":
    main()
