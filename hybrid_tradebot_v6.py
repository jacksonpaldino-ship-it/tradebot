#!/usr/bin/env python3
"""
hybrid_tradebot_v6.py

- Uses Alpaca REST API for orders and account info
- Uses yfinance for minute bars (reliable without Alpaca data subscription)
- Runs once per call (intended for every 10 min GitHub Action schedule)
- Targets ~2–4 trades/day with risk control
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
from alpaca_trade_api.rest import REST, APIError

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_TRADES_PER_DAY = 4
PER_SYMBOL_DAILY_CAP = 2
TP_PCT = 0.0020
SL_PCT = 0.0015
RISK_PER_TRADE = 0.01  # 1% equity per trade
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2500
VWAP_BAND = 0.005
STATE_FILE = "bot_state_v6.json"
TRADES_CSV = "trades_v6.csv"
LOG_FILE = "bot_v6.log"
TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca client ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in repo secrets")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Logging / state ----------------
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
    except:
        pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades": 0, "per_symbol": {}}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return {"date": None, "daily_trades": 0, "per_symbol": {}}

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

# ---------------- Market data helpers ----------------
def fetch_bars_yf(symbol, minutes=200):
    try:
        period_days = max(1, (minutes // 60) + 1)
        df = yf.download(symbol, period=f"{period_days}d", interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        df = df[["open","high","low","close","volume"]].dropna()
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"fetch_bars_yf error {symbol}: {e}")
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
    return float(pv / v) if v > 0 else float(df["close"].iloc[-1])

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
    if not equity or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price*0.0005)
    qty = int(max(1, math.floor(risk_amount/(per_share_risk+EPS))))
    max_nom = int(max(1, math.floor(equity*0.3/entry_price)))
    return min(qty, max_nom)

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
        log(f"Bracket submitted {symbol} qty={qty} tp={tp_price:.4f} sl={sl_price:.4f}")
        return order
    except Exception as e:
        log(f"submit_bracket error: {e}")
        return None

def submit_market_buy(symbol, qty):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')
        log(f"Market buy submitted {symbol} qty={qty}")
        return order
    except Exception as e:
        log(f"submit_market_buy error: {e}")
        return None

def has_open_positions():
    try:
        return len(api.list_positions()) > 0
    except:
        return False

# ---------------- Scoring ----------------
def compute_score(df):
    if df is None or len(df) < 20:
        return None
    df = df.dropna(subset=["close","high","low","volume"])
    price = float(df["close"].iloc[-1])
    volume = df["volume"].iloc[-1]
    if isinstance(volume, pd.Series):
        volume = float(volume.iloc[-1])
    else:
        volume = float(volume)
    if volume < MIN_VOLUME:
        return None
    vwap_val = compute_vwap(df)
    vw_gap = abs(price - vwap_val)/vwap_val
    macd_hist = compute_macd_hist(df)
    vol_score = min(1.0, volume/(df["volume"].mean()+EPS))
    vw_score = max(0.0, 1.0 - vw_gap/VWAP_BAND)
    macd_score = 1.0 if macd_hist>0 else 0.0
    score = 0.45*vw_score + 0.35*vol_score + 0.20*macd_score
    return float(score)

def pick_trade_candidate():
    candidates = []
    for sym in SYMBOLS:
        df = fetch_bars_yf(sym, 120)
        score = compute_score(df)
        if score and score > 0.25:
            candidates.append({"symbol": sym, "score": score, "price": df["close"].iloc[-1], "atr": compute_atr(df)})
    if not candidates:
        return None
    # sort highest score first
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

# ---------------- Main ----------------
def main():
    log(f"Run start ET {now_et().isoformat()}")
    try:
        clock = api.get_clock()
        if not getattr(clock,"is_open",False):
            log("Market closed, skipping run")
            return
    except:
        log("Error fetching market clock, skipping")
        return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "daily_trades": 0, "per_symbol": {}}
        save_state(state)
    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return
    if has_open_positions():
        log("Open positions exist, skipping entry")
        return

    candidate = pick_trade_candidate()
    if not candidate:
        log("No scored candidates this run")
        return

    sym = candidate["symbol"]
    per_sym = state["per_symbol"].get(sym, 0)
    if per_sym >= PER_SYMBOL_DAILY_CAP:
        log(f"Per-symbol cap reached for {sym}")
        return

    qty = compute_qty(candidate["price"], candidate["atr"])
    if qty < 1:
        log(f"Quantity too low for {sym}")
        return

    tp = candidate["price"]*(1+TP_PCT)
    sl = candidate["price"]*(1-SL_PCT)
    order = submit_bracket(sym, qty, sl, tp)
    if order:
        state["daily_trades"] += 1
        state["per_symbol"][sym] = per_sym + 1
        save_state(state)
        append_trade_row([utcnow_iso(), sym, "BUY_SUBMIT", qty, round(candidate["price"],6), None, None, f"score:{candidate['score']:.3f}"])
        log(f"Placed bracket for {sym} qty={qty} score={candidate['score']:.3f}")
    else:
        log(f"Failed to place order for {sym}")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        log("Unhandled exception: "+repr(e))
        log(traceback.format_exc())
