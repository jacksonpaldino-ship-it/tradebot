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
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]  # Primary symbols to monitor
VWAP_THRESHOLD = 0.01     # initial threshold, will adapt
MAX_CANDLE_SPREAD = 1.0   # initial threshold, will adapt
MIN_VOLUME = 50000        # initial threshold
PAPER = True              # paper trading

# Alpaca API from GitHub secrets
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
        with open(STATS_JSON,"r") as f:
            stats = json.load(f)

    symbol = trade["symbol"]
    if symbol not in stats:
        stats[symbol] = {"wins":0,"losses":0,"trades":0}

    stats[symbol]["trades"] += 1
    if trade.get("result")=="WIN":
        stats[symbol]["wins"] +=1
    elif trade.get("result")=="LOSS":
        stats[symbol]["losses"] +=1

    with open(STATS_JSON,"w") as f:
        json.dump(stats,f,indent=2)

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
        logging.error(f"Order failed {symbol}: {e}")
        return None

def fetch_google_sheet_signals():
    try:
        resp = requests.get(SIGNAL_SHEET_CSV_URL, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        df = df[df.get("enabled", True)==True]
        return df.to_dict("records")
    except Exception as e:
        logging.error(f"Failed to fetch Google Sheet: {e}")
        return []

# -------------------- PRIMARY STRATEGY --------------------
def score_symbol(symbol):
    try:
        df = yf.download(symbol, period="3d", interval="5m", progress=False, auto_adjust=True)
        if df.empty: return None

        df["VWAP"] = (df["Close"]*df["Volume"]).cumsum() / df["Volume"].cumsum()
        last = df.iloc[-1]

        # future-proof float
        price = float(last["Close"].iloc[0]) if hasattr(last["Close"],"iloc") else float(last["Close"])
        vwap = float(last["VWAP"].iloc[0]) if hasattr(last["VWAP"],"iloc") else float(last["VWAP"])
        volume = float(last["Volume"].iloc[0]) if hasattr(last["Volume"],"iloc") else float(last["Volume"])
        spread = (float(last["High"].iloc[0])-float(last["Low"].iloc[0])
                  if hasattr(last["High"],"iloc") else float(last["High"])-float(last["Low"]))

        # adaptive thresholds
        vw_pct = abs(price-vwap)/vwap
        spread_threshold = df["High"].max()-df["Low"].min()
        vol_norm = min(volume/100000,1.0)

        score = (1-vw_pct)*0.4 + vol_norm*0.4 + (1 - spread/spread_threshold)*0.2
        return {"symbol":symbol, "score":score, "price":price, "vwap":vwap, "volume":volume, "spread":spread}
    except Exception as e:
        logging.error(f"Error scoring {symbol}: {e}")
        return None

def primary_trade():
    candidates = []
    for s in SYMBOLS:
        res = score_symbol(s)
        if res: candidates.append(res)
    if not candidates: return None

    # rank by score
    candidates.sort(key=lambda x:x["score"], reverse=True)
    for c in candidates:
        price_vwap = abs(c["price"]-c["vwap"])/c["vwap"]
        if price_vwap>VWAP_THRESHOLD:
            logging.info(f"{c['symbol']} price-vwap {price_vwap:.4f} too far")
            continue
        if c["spread"]>MAX_CANDLE_SPREAD:
            logging.info(f"{c['symbol']} spread {c['spread']:.4f} too big")
            continue
        if c["volume"]<MIN_VOLUME:
            logging.info(f"{c['symbol']} volume {c['volume']} too low")
            continue
        order = submit_order(c["symbol"],1,"BUY")
        if order:
            trade = {"timestamp":now_et().isoformat(),"symbol":c["symbol"],"side":"BUY","qty":1}
            save_trade(trade)
            return trade
    # guarantee top candidate if none passed filters
    top = candidates[0]
    logging.info(f"Forcing trade on top candidate {top['symbol']}")
    order = submit_order(top["symbol"],1,"BUY")
    if order:
        trade = {"timestamp":now_et().isoformat(),"symbol":top["symbol"],"side":"BUY","qty":1,"notes":"guaranteed"}
        save_trade(trade)
        return trade
    return None

# -------------------- FALLBACK STRATEGY --------------------
def fallback_trade():
    signals = fetch_google_sheet_signals()
    if not signals: return None
    for sig in signals:
        try:
            symbol = sig.get("symbol")
            action = sig.get("action","BUY").upper()
            qty = int(sig.get("qty",1))
            order = submit_order(symbol, qty, action)
            if order:
                trade = {
                    "timestamp":now_et().isoformat(),
                    "symbol":symbol,
                    "side":action,
                    "qty":qty,
                    "notes":"fallback"
                }
                save_trade(trade)
                return trade
        except Exception as e:
            logging.error(f"Fallback trade error: {e}")
    return None

# -------------------- MAIN --------------------
def main():
    logging.info(f"Starting hybrid_tradebot_advanced {now_et()}")
    trade = primary_trade()
    if trade:
        logging.info(f"Primary trade executed: {trade}")
        return
    # fallback
    trade = fallback_trade()
    if trade:
        logging.info(f"Fallback trade executed: {trade}")
        return
    logging.info("No trade executed today")

if __name__=="__main__":
    main()
