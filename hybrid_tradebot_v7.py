#!/usr/bin/env python3
"""
hybrid_tradebot_v7.py
- Alpaca for orders/account
- yfinance for minute bars
- 2-4 trades/day, no guaranteed trade
- Robust, error-free, designed to run every 10 minutes
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
MAX_TRADES_PER_DAY = 4
PER_SYMBOL_DAILY_CAP = 2
TP_PCT = 0.0020
SL_PCT = 0.0015
RISK_PER_TRADE = 0.01
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2500
VWAP_BAND = 0.005
STATE_FILE = "bot_state_v7.json"
TRADES_CSV = "trades_v7.csv"
LOG_FILE = "bot_v7.log"

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca client ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in repository secrets")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Logging/State ----------------
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
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "open_order_id": None}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
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

# ---------------- Market Data ----------------
def fetch_bars_yf(symbol, minutes=200):
    try:
        period_days = max(1, (minutes // 60) + 1)
        df = yf.download(symbol, period=f"{period_days}d", interval="1m", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        # Flatten multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        # Ensure required columns exist
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                df[col] = np.nan
        df = df[["open","high","low","close","volume"]].dropna()
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"fetch_bars_yf error {symbol}: {e}")
        return None

def fetch_recent_bars(symbol, minutes=200):
    return fetch_bars_yf(symbol, minutes)

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
    pv = (df["close"]*df["volume"]).sum()
    v = df["volume"].sum()
    return float(pv/v) if v>0 else float(df["close"].iloc[-1])

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
    except Exception as e:
        log(f"get_equity error: {e}")
        return None

def compute_qty(entry_price, atr):
    equity = get_equity()
    if equity is None or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price*0.0005)
    qty = int(max(1, math.floor(risk_amount/(per_share_risk+EPS))))
    max_nominal = int(max(1, math.floor((equity*0.3)/entry_price)))
    return min(qty, max_nominal)

# ---------------- Orders ----------------
def submit_bracket(symbol, qty, sl_price, tp_price):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='day',
            order_class='bracket',
            take_profit={'limit_price': str(round(tp_price,6))},
            stop_loss={'stop_price': str(round(sl_price,6))}
        )
        log(f"Bracket submitted: {symbol} qty={qty} tp={tp_price:.4f} sl={sl_price:.4f}")
        return order
    except Exception as e:
        log(f"submit_bracket error: {e}")
        return None

def has_open_positions():
    try:
        pos = api.list_positions()
        return len(pos) > 0
    except Exception as e:
        log(f"list_positions error: {e}")
        return False

# ---------------- Scoring ----------------
def compute_score(df):
    if df is None or len(df)<20:
        return None
    try:
        df = df.dropna(subset=["open","high","low","close","volume"])
        if df.empty:
            return None
        price = float(df["close"].iloc[-1])
        volume = float(df["volume"].iloc[-1])
        if volume < MIN_VOLUME:
            return None
        vwap_val = compute_vwap(df)
        vw_gap = abs(price-vwap_val)/vwap_val
        macd_hist = compute_macd_hist(df)
        vol_score = min(1.0, volume/(df["volume"].mean()+EPS))
        vw_score = max(0.0, 1.0 - vw_gap/VWAP_BAND)
        macd_score = 1.0 if macd_hist>0 else 0.0
        return 0.45*vw_score + 0.35*vol_score + 0.2*macd_score
    except Exception as e:
        log(f"compute_score error: {e}")
        return None

# ---------------- Pick Candidate ----------------
def pick_trade_candidate():
    scored = []
    for sym in SYMBOLS:
        df = fetch_recent_bars(sym)
        score = compute_score(df)
        if score is not None:
            scored.append({"symbol": sym, "score": score, "df": df})
    if not scored:
        return None
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0]

# ---------------- Run Once ----------------
def run_once():
    log(f"Run start ET {now_et().isoformat()}")
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            log("Market closed; exiting.")
            return
    except Exception as e:
        log(f"get_clock error: {e}")
        return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "daily_trades":0, "per_symbol":{}, "open_order_id":None}
        save_state(state)
    if state["daily_trades"]>=MAX_TRADES_PER_DAY:
        log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return
    if has_open_positions():
        log("Open positions detected; skipping entry.")
        return

    candidate = pick_trade_candidate()
    if candidate is None:
        log("No trade candidates this run.")
        return

    sym = candidate["symbol"]
    df = candidate["df"]
    entry_price = float(df["close"].iloc[-1])
    atr = compute_atr(df)
    qty = compute_qty(entry_price, atr)
    if qty < 1:
        log("Computed quantity <1; skipping.")
        return
    tp = entry_price*(1+TP_PCT)
    sl = entry_price*(1-SL_PCT)

    order = submit_bracket(sym, qty, sl, tp)
    if order:
        state["daily_trades"] += 1
        state["per_symbol"][sym] = state["per_symbol"].get(sym,0)+1
        state["open_order_id"] = getattr(order,'id',None)
        save_state(state)
        append_trade_row([utcnow_iso(), sym, "BUY_SUBMIT", qty, round(entry_price,6), None, None, f"score:{candidate['score']:.3f}"])
        log(f"Placed trade for {sym} qty={qty} score:{candidate['score']:.3f}")

if __name__=="__main__":
    try:
        run_once()
    except Exception as e:
        log("Unhandled exception: "+repr(e))
        log(traceback.format_exc())
