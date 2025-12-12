#!/usr/bin/env python3
"""
hybrid_tradebot_v8.py
- Alpaca + yfinance fallback
- 2-4 trades/day, efficient scoring
- Market-friendly take-profit/stop-loss rounding
- Single-run: scheduled every 10 minutes
"""

import os
import time
import math
import json
import csv
import traceback
from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_TRADES_PER_DAY = 4
PER_SYMBOL_DAILY_CAP = 2
TP_PCT = 0.002       # 0.2%
SL_PCT = 0.0015      # 0.15%
RISK_PER_TRADE = 0.015
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2500
VWAP_BAND = 0.0075
ENTRY_SCORE_THRESHOLD = 0.20  # lower threshold
STATE_FILE = "bot_state_v8.json"
TRADES_CSV = "trades_v8.csv"
LOG_FILE = "bot_v8.log"

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca client ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set Alpaca secrets!")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Logging ----------------
def now_et():
    return datetime.now(TZ)

def utcnow_iso():
    return datetime.utcnow().isoformat()

def log(s):
    line = f"{utcnow_iso()} {s}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades": 0, "per_symbol": {}}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "daily_trades": 0, "per_symbol": {}}

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

# ---------------- Market Data ----------------
def fetch_recent_bars(symbol, minutes=120):
    try:
        period_days = max(1, (minutes // 60) + 1)
        df = yf.download(symbol, period=f"{period_days}d", interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                return None
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"fetch_recent_bars error {symbol}: {e}")
        return None

# ---------------- Indicators ----------------
def compute_atr(df, period=ATR_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return float(max(atr, 1e-6))

def compute_vwap(df):
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    if v <= 0:
        return float(df["close"].iloc[-1])
    return float(pv / v)

def compute_macd_hist(df):
    close = df["close"]
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return float((macd - sig).iloc[-1])

# ---------------- Sizing ----------------
def get_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except:
        return None

def compute_qty(entry_price, atr):
    equity = get_equity()
    if equity is None or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    max_nominal = int(max(1, math.floor((equity*0.3)/entry_price)))
    return min(qty, max_nominal)

# ---------------- Orders ----------------
def round_price(p):
    # round to nearest cent
    return round(p + 1e-6, 2)

def submit_bracket(symbol, qty, sl_price, tp_price):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day',
            order_class='bracket',
            take_profit={'limit_price': str(round_price(tp_price))},
            stop_loss={'stop_price': str(round_price(sl_price))}
        )
        log(f"Bracket submitted: {symbol} qty={qty} tp={tp_price:.2f} sl={sl_price:.2f}")
        return order
    except Exception as e:
        log(f"submit_bracket error: {e}")
        return None

# ---------------- Utility ----------------
def has_open_positions():
    try:
        return len(api.list_positions()) > 0
    except:
        return False

# ---------------- Scoring ----------------
def compute_score(df):
    df = df.dropna(subset=["close","high","low","volume"])
    if df.empty or len(df) < 10:
        return None
    volume = float(df["volume"].iloc[-1])
    if volume < MIN_VOLUME:
        return None
    window = df.tail(60)
    price = float(window["close"].iloc[-1])
    vwap = compute_vwap(window)
    vw_gap = abs(price - vwap)/(vwap+EPS)
    macd_hist = compute_macd_hist(window)
    atr = compute_atr(window)
    vol_score = min(1.0, float(window["volume"].iloc[-1])/(float(window["volume"].mean())+EPS))
    vw_score = max(0.0, 1.0 - vw_gap/VWAP_BAND)
    macd_score = 1.0 if macd_hist > 0 else 0.0
    score = 0.40*vw_score + 0.40*vol_score + 0.20*macd_score
    return {"score": score, "price": price, "atr": atr}

def pick_trade_candidate():
    scored = []
    for sym in SYMBOLS:
        df = fetch_recent_bars(sym)
        if df is None:
            continue
        info = compute_score(df)
        if info is None:
            continue
        if info["score"] >= ENTRY_SCORE_THRESHOLD:
            scored.append((sym, info))
    if not scored:
        return None
    scored.sort(key=lambda x: x[1]["score"], reverse=True)
    return scored[0]  # top candidate

# ---------------- Main ----------------
def main():
    log(f"Run start ET {now_et().isoformat()}")
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            log("Market closed")
            return
    except:
        log("Clock fetch failed")
        return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "daily_trades": 0, "per_symbol": {}}
        save_state(state)

    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log("Daily cap reached")
        return

    if has_open_positions():
        log("Open positions exist, skipping entry")
        return

    candidate = pick_trade_candidate()
    if candidate is None:
        log("No candidate meets score threshold")
        return

    sym, info = candidate
    per_sym = state["per_symbol"].get(sym, 0)
    if per_sym >= PER_SYMBOL_DAILY_CAP:
        log(f"Per-symbol cap reached for {sym}")
        return

    entry_price = info["price"]
    atr = info["atr"]
    qty = compute_qty(entry_price, atr)
    if qty < 1:
        log("Computed qty < 1, skipping")
        return

    tp = entry_price*(1+TP_PCT)
    sl = entry_price*(1-SL_PCT)

    order = submit_bracket(sym, qty, sl, tp)
    if order:
        state["daily_trades"] += 1
        state["per_symbol"][sym] = per_sym + 1
        save_state(state)
        append_trade_row([utcnow_iso(), sym, "BUY_SUBMIT", qty, round_price(entry_price), None, None, f"score:{info['score']:.3f}"])
        log(f"Placed bracket for {sym} qty={qty} score={info['score']:.3f}")
    else:
        log("Failed to submit order")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled exception: {repr(e)}")
        log(traceback.format_exc())
