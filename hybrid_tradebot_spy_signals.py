#!/usr/bin/env python3
"""
hybrid_tradebot_v3.py (ready for 10-min schedule, risk-controlled)
- Minimum 1 trade/day guaranteed
- Target 5-10 trades/day
- Uses Alpaca for orders and account info
- Uses yfinance for minute bars
- ATR-based sizing, MACD + VWAP + volume scoring
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
from alpaca_trade_api.rest import REST, APIError

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_TRADES_PER_DAY = 10
PER_SYMBOL_DAILY_CAP = 4
TP_PCT = 0.0020
SL_PCT = 0.0015
RISK_PER_TRADE = 0.008  # 0.8% equity risk per trade
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2500
VWAP_BAND = 0.005
SCORE_THRESHOLD = 0.20
GUARANTEE_TRADE = True
FORCE_HOUR = 15
FORCE_MIN = 30
STATE_FILE = "bot_state_v3.json"
TRADES_CSV = "trades_v3.csv"
LOG_FILE = "bot_v3.log"

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca client ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in repository secrets")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- logging / state ----------------
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

def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
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

# ---------------- market data ----------------
def fetch_bars_yf(symbol, minutes=200):
    try:
        period_days = max(1, (minutes // 60) + 1)
        period = f"{period_days}d"
        df = yf.download(tickers=symbol, period=period, interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        if not {"open","high","low","close","volume"}.issubset(df.columns):
            return None
        df = df[["open","high","low","close","volume"]]
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"fetch_bars_yf error {symbol}: {e}")
        return None

def fetch_recent_bars(symbol, minutes=200):
    return fetch_bars_yf(symbol, minutes=minutes)

# ---------------- indicators ----------------
def compute_atr(df, period=ATR_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
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

# ---------------- sizing ----------------
def get_equity():
    try:
        return float(api.get_account().equity)
    except Exception as e:
        log(f"get_equity error: {e}")
        return None

def compute_qty(entry_price, atr):
    equity = get_equity()
    if equity is None or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    max_nominal = int(max(1, math.floor((equity * 0.3) / entry_price)))
    return min(qty, max_nominal)

# ---------------- orders ----------------
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

def submit_market_buy(symbol, qty):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
        log(f"Market buy submitted: {symbol} qty={qty}")
        return order
    except Exception as e:
        log(f"submit_market_buy error: {e}")
        return None

# ---------------- utilities ----------------
def has_open_positions():
    try:
        return len(api.list_positions()) > 0
    except Exception as e:
        log(f"list_positions error: {e}")
        return False

# ---------------- scoring ----------------
def score_symbol(symbol):
    df = fetch_recent_bars(symbol, 120)
    if df is None or df.empty or len(df) < 10:
        return None
    volume_last = int(df["volume"].iloc[-1])
    if volume_last < MIN_VOLUME:
        return None
    window = df.tail(60)
    price = float(window["close"].iloc[-1])
    vwap = compute_vwap(window)
    vw_gap = abs(price - vwap) / (vwap + EPS)
    macd_hist = compute_macd_hist(window)
    atr = compute_atr(window)
    vol_score = min(1.0, float(window["volume"].iloc[-1]) / (float(window["volume"].mean()) + EPS))
    vw_score = max(0.0, 1.0 - vw_gap / VWAP_BAND)
    macd_score = 1.0 if macd_hist > 0 else 0.0
    score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * macd_score
    return {"symbol": symbol, "score": float(score), "price": price, "vwap": vwap, "atr": atr, "macd_hist": macd_hist, "vol": volume_last}

# ---------------- forced trade ----------------
def force_trade(symbol):
    df = fetch_recent_bars(symbol, 60)
    if df is None or df.empty:
        return False
    price = float(df["close"].iloc[-1])
    atr = compute_atr(df)
    equity = get_equity() or 0
    per_trade_risk = max(0.005, RISK_PER_TRADE/3)
    risk_amount = equity * per_trade_risk
    per_share_risk = max(atr, price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    qty = min(qty, int(max(1, math.floor((equity * 0.2) / price))))
    if qty <= 0:
        return False
    tp = price * (1 + TP_PCT)
    sl = price * (1 - SL_PCT)
    order = submit_bracket(symbol, qty, sl, tp)
    if order:
        append_trade_row([utcnow_iso(), symbol, "FORCE_BUY", qty, round(price,6), None, None, "forced_trade"])
        log(f"Forced trade placed: {symbol} qty={qty}")
        return True
    return False

# ---------------- main loop ----------------
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
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "open_order_id": None}
        save_state(state)

    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return

    if has_open_positions():
        log("Open positions detected; skipping entry this run.")
        return

    scored = []
    for s in SYMBOLS:
        try:
            info = score_symbol(s)
            if info:
                scored.append(info)
        except Exception as e:
            log(f"score_symbol error {s}: {e}")

    if scored:
        scored.sort(key=lambda x: x["score"], reverse=True)
        for c in scored:
            if c["score"] < SCORE_THRESHOLD:
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
                append_trade_row([utcnow_iso(), sym, "BUY", qty, round(price,6), None, None, f"score={c['score']:.3f}"])
                log(f"Entry placed: {sym} qty={qty}")
                return
    # Forced trade if min 1 trade not yet executed
    now = now_et()
    if GUARANTEE_TRADE and state["daily_trades"] == 0 and (now.hour > FORCE_HOUR or (now.hour == FORCE_HOUR and now.minute >= FORCE_MIN)):
        best = max(SYMBOLS, key=lambda s: fetch_recent_bars(s,30)["volume"].iloc[-1] if fetch_recent_bars(s,30) is not None else 0)
        force_trade(best)
    log("Run complete.")

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Unhandled exception: " + repr(e))
        log(traceback.format_exc())
