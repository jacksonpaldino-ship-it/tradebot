#!/usr/bin/env python3

"""
hybrid_tradebot_v3.py
Clean, error-free, Alpaca+YFinance hybrid intraday bot.
"""

import os
import time
import math
import json
import csv
import traceback
from datetime import datetime
import pytz

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_TRADES_PER_DAY = 6
PER_SYMBOL_DAILY_CAP = 4
TP_PCT = 0.0020
SL_PCT = 0.0015
RISK_PER_TRADE = 0.015
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2500
VWAP_BAND = 0.005
GUARANTEE_TRADE = True
FORCE_HOUR = 15
FORCE_MIN = 30
STATE_FILE = "bot_state_v3.json"
TRADES_CSV = "trades_v3.csv"
LOG_FILE = "bot_v3.log"

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in repository secrets")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Helpers ----------------
def now_et():
    return datetime.now(TZ)

def utcnow_iso():
    return datetime.utcnow().isoformat()

def log(s):
    msg = f"{utcnow_iso()} {s}"
    print(msg)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except:
        pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "open_order_id": None}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "open_order_id": None}

def save_state(s):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(s, f)
    except Exception as e:
        log(f"save_state error: {e}")

def append_trade_row(row):
    header = ["utc_ts","symbol","side","qty","entry","exit","pnl","note"]
    exists = os.path.exists(TRADES_CSV)
    try:
        with open(TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        log(f"append_trade_row error: {e}")

# ---------------- Market data ----------------
def fetch_bars_yf(symbol, minutes=200):
    try:
        days = max(1, (minutes // 60) + 1)
        df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        df = df[["open","high","low","close","volume"]]
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"{symbol} yfinance error: {e}")
        return None

def fetch_recent_bars(symbol, minutes=200):
    return fetch_bars_yf(symbol, minutes)

# ---------------- Indicators ----------------
def compute_atr(df, n=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n, min_periods=1).mean().iloc[-1])

def compute_vwap(df):
    pv = (df["close"] * df["volume"]).sum()
    vol = df["volume"].sum()
    if vol <= 0:
        return float(df["close"].iloc[-1])
    return float(pv / vol)

def compute_macd_hist(df):
    close = df["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return float((macd - signal).iloc[-1])

# ---------------- Sizing ----------------
def get_equity():
    try:
        return float(api.get_account().equity)
    except:
        return None

def compute_qty(price, atr):
    eq = get_equity()
    if not eq:
        return 1
    risk_amount = eq * RISK_PER_TRADE
    per_share_risk = max(atr, price * 0.0005)
    qty = int(max(1, risk_amount // per_share_risk))
    max_pos = int((eq * 0.3) // price)
    return max(1, min(qty, max_pos))

# ---------------- Orders ----------------
def submit_bracket(symbol, qty, sl, tp):
    try:
        return api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp)},
            stop_loss={"stop_price": str(sl)}
        )
    except Exception as e:
        log(f"Bracket error {symbol}: {e}")
        return None

def submit_market_buy(symbol, qty):
    try:
        return api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
    except Exception as e:
        log(f"Market-buy error {symbol}: {e}")
        return None

# ---------------- Position check ----------------
def has_open_positions():
    try:
        return len(api.list_positions()) > 0
    except:
        return False

# ---------------- Scoring ----------------
def score_symbol(symbol):
    df = fetch_recent_bars(symbol, minutes=120)
    if df is None or len(df) < 20:
        return None
    vol_last = int(df["volume"].iloc[-1])
    if vol_last < MIN_VOLUME:
        return None

    window = df.tail(60)
    price = float(window["close"].iloc[-1])
    vwap = compute_vwap(window)
    vwgap = abs(price - vwap) / (vwap + EPS)

    macd_hist = compute_macd_hist(window)
    atr = compute_atr(window)

    vol_score = min(1, vol_last / (window["volume"].mean() + EPS))
    vw_score = max(0, 1 - vwgap / VWAP_BAND)
    macd_score = 1 if macd_hist > 0 else 0

    score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * macd_score
    return {"symbol": symbol, "score": score, "price": price, "atr": atr}

# ---------------- Main ----------------
def run_once():
    log(f"Run start ET {now_et().isoformat()}")

    try:
        if not api.get_clock().is_open:
            log("Market closed.")
            return
    except Exception as e:
        log(f"Clock error: {e}")
        return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")

    if state["date"] != today:
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "open_order_id": None}
        save_state(state)

    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log("Daily cap reached.")
        return

    if has_open_positions():
        log("Existing positions; skipping.")
        return

    scored = []
    for s in SYMBOLS:
        info = score_symbol(s)
        if info:
            scored.append(info)

    if scored:
        scored.sort(key=lambda x: x["score"], reverse=True)

        for c in scored:
            if c["score"] < 0.25:
                continue

            sym = c["symbol"]
            used = state["per_symbol"].get(sym, 0)
            if used >= PER_SYMBOL_DAILY_CAP:
                continue

            price = c["price"]
            atr = c["atr"]
            qty = compute_qty(price, atr)
            if qty < 1:
                continue

            sl = price * (1 - SL_PCT)
            tp = price * (1 + TP_PCT)

            o = submit_bracket(sym, qty, sl, tp)
            if o:
                state["daily_trades"] += 1
                state["per_symbol"][sym] = used + 1
                save_state(state)
                append_trade_row([utcnow_iso(), sym, "BUY", qty, price, None, None, f"score={c['score']:.3f}"])
                log(f"Entry: {sym} qty={qty}")
                return

    log("No trades executed.")

if __name__ == "__main__":
    try:
        run_once()
    except Exception:
        log(traceback.format_exc())
