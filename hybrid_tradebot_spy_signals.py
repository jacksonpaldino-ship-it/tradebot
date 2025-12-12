#!/usr/bin/env python3
"""
hybrid_tradebot_v3.py

Goals:
- Minimum 1 trade/day guaranteed (forced trade at FORCE_HOUR:FORCE_MIN ET)
- Target ~5 trades/day (configurable caps)
- MACD + VWAP + volume scoring, ATR-based sizing
- Bracket orders (TP + SL) preferred, market fallback
- Single-run: exit quickly (designed to be scheduled every 10 minutes)
- Uses alpaca-trade-api (REST) and yfinance fallback for market data
"""

import os
import time
import math
import json
import csv
import traceback
from datetime import datetime, timedelta, time as dtime
import pytz
import requests

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, APIError

# ---------------- CONFIG (tweakable) ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]      # universe
MAX_TRADES_PER_DAY = 6                      # target ~3-6 trades/day
PER_SYMBOL_DAILY_CAP = 4
TP_PCT = 0.0020                             # 0.20% take profit
SL_PCT = 0.0015                             # 0.15% stop loss
RISK_PER_TRADE = 0.015                      # 1.5% equity risk per trade (ATR-based)
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2_500                          # realistic 1-min bar filter
VWAP_BAND = 0.005                           # 0.5% default band for scoring (adaptive will be used)
GUARANTEE_TRADE = True
FORCE_HOUR = 15                             # ET 15:30 fallback if none executed
FORCE_MIN = 30
STATE_FILE = "bot_state_v3.json"
TRADES_CSV = "trades_v3.csv"
LOG_FILE = "bot_v3.log"

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca client (old SDK) ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. "https://paper-api.alpaca.markets"
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

# ---------------- market data helpers ----------------
def fetch_bars_alpaca(symbol, limit=200):
    """
    Try Alpaca bars; fall back to None on any error.
    Returns DataFrame with columns open,high,low,close,volume (index = timestamp)
    """
    try:
        bars = api.get_barset(symbol, timeframe='1Min', limit=limit)
        # alpaca-trade-api returns BarList in a dict
        if symbol not in bars or len(bars[symbol]) == 0:
            return None
        # convert to DataFrame
        rows = []
        for b in bars[symbol]:
            rows.append({'t': b.t, 'open': b.o, 'high': b.h, 'low': b.l, 'close': b.c, 'volume': b.v})
        df = pd.DataFrame(rows).set_index('t')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        # drop to yfinance fallback
        return None

def fetch_bars_yf(symbol, minutes=200):
    try:
        # yfinance returns tz-aware index; remove tz
        period = f"{max(1, (minutes // 60) + 1)}d"
        df = yf.download(symbol, period=period, interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]]
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        return None

def fetch_recent_bars(symbol, minutes=200):
    df = fetch_bars_alpaca(symbol, limit=minutes)
    if df is not None:
        return df
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
    hist = (macd - sig).iloc[-1]
    return float(hist)

# ---------------- sizing ----------------
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
    per_share_risk = max(atr, entry_price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    # safety cap: do not allocate more than 30% of equity on a single position
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

# ---------------- utility ----------------
def has_open_positions():
    try:
        pos = api.list_positions()
        return len(pos) > 0
    except Exception as e:
        log(f"list_positions error: {e}")
        return False

# ---------------- main logic ----------------
def score_symbol(symbol):
    df = fetch_recent_bars(symbol, minutes=120)
    if df is None or df.empty or len(df) < 10:
        return None
    try:
        volume_last = int(df["volume"].iloc[-1])
    except Exception:
        volume_last = 0
    if volume_last < MIN_VOLUME:
        return None

    window = df.tail(60)
    price = float(window["close"].iloc[-1])
    vwap = compute_vwap(window)
    vw_gap = abs(price - vwap) / (vwap + EPS)
    macd_hist = compute_macd_hist(window)
    atr = compute_atr(window)
    # adaptive spread normalization
    vol_score = min(1.0, window["volume"].iloc[-1] / (window["volume"].mean() + EPS))
    vw_score = max(0.0, 1.0 - vw_gap / VWAP_BAND)
    macd_score = 1.0 if macd_hist > 0 else 0.0
    score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * macd_score
    return {"symbol": symbol, "score": float(score), "price": price, "vwap": vwap, "atr": atr, "macd_hist": macd_hist, "vol": volume_last}

def force_trade(symbol):
    # Conservative forced trade: small size (1% equity risk) so force does not blow account
    df = fetch_recent_bars(symbol, minutes=60)
    if df is None or df.empty:
        return False
    price = float(df["close"].iloc[-1])
    atr = compute_atr(df)
    # use smaller risk for forced trade
    equity = get_equity() or 0
    per_trade_risk = max(0.005, RISK_PER_TRADE/3)  # ~0.5% or smaller
    risk_amount = equity * per_trade_risk
    per_share_risk = max(atr, price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    qty = min(qty, int(max(1, math.floor((equity * 0.2) / price))))  # cap nominal
    if qty <= 0:
        return False
    tp = price * (1 + TP_PCT)
    sl = price * (1 - SL_PCT)
    order = submit_bracket(symbol, qty, sl, tp)
    if order:
        append_trade_row([utcnow_iso(), symbol, "FORCE_BUY", qty, round(price,6), None, None, "forced_trade"])
        log(f"Forced trade placed on {symbol} qty={qty}")
        return True
    return False

def run_once():
    log(f"Run start ET {now_et().isoformat()}")

    # check market open
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            log("Market closed; exiting.")
            return
    except Exception as e:
        log(f"get_clock error: {e}")
        # if error fetching clock, exit to be safe
        return

    # load/reset state
    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "open_order_id": None}
        save_state(state)

    # if daily cap reached, exit
    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return

    # if there's any open position, skip entering new trade (we rely on bracket exits)
    if has_open_positions():
        log("Open positions detected; skipping entry until cleared.")
        return

    # score universe
    scored = []
    for s in SYMBOLS:
        try:
            info = score_symbol(s)
            if info is not None:
                scored.append(info)
        except Exception as e:
            log(f"score_symbol error {s}: {e}")

    if scored:
        # sort by score desc
        scored.sort(key=lambda x: x["score"], reverse=True)
        # try top candidates until one placed or none left (but respect per-symbol caps)
        for cand in scored:
            sym = cand["symbol"]
            per_sym = state["per_symbol"].get(sym, 0)
            if per_sym >= PER_SYMBOL_DAILY_CAP:
                log(f"Per-symbol cap reached for {sym}")
                continue
            # acceptance: require minimum score to avoid spam; tuned to target ~5/day
            if cand["score"] < 0.25:
                log(f"{sym} score {cand['score']:.3f} below entry threshold")
                continue
            entry_price = cand["price"]
            atr = cand["atr"]
            qty = compute_qty(entry_price, atr)
            if qty < 1:
                continue
            tp = entry_price * (1 + TP_PCT)
            sl = entry_price * (1 - SL_PCT)
            # try bracket order
            order = submit_bracket(sym, qty, sl, tp)
            if order:
                state["daily_trades"] += 1
                state["per_symbol"][sym] = per_sym + 1
                state["open_order_id"] = getattr(order, 'id', None)
                save_state(state)
                append_trade_row([utcnow_iso(), sym, "BUY_SUBMIT", qty, round(entry_price,6), None, None, f"score:{cand['score']:.3f}"])
                log(f"Placed bracket for {sym} qty={qty} score={cand['score']:.3f}")
                return
            # fallback market buy
            order2 = submit_market_buy(sym, qty)
            if order2:
                # best-effort filled price
                time.sleep(1.2)
                try:
                    o = api.get_order(order2.id)
                    fill_price = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else entry_price
                except Exception:
                    fill_price = entry_price
                state["daily_trades"] += 1
                state["per_symbol"][sym] = per_sym + 1
                save_state(state)
                append_trade_row([utcnow_iso(), sym, "BUY_FILLED", qty, round(fill_price,6), None, None, f"score:{cand['score']:.3f}"])
                log(f"Placed market buy for {sym} qty={qty} filled {fill_price:.4f}")
                return
            # else try next candidate
        log("Scored candidates exhausted; no order placed.")
    else:
        log("No scored candidates this run.")

    # Fallback: forced trade near end-of-day if guarantee on
    now = now_et()
    if GUARANTEE_TRADE and (now.hour > FORCE_HOUR or (now.hour == FORCE_HOUR and now.minute >= FORCE_MIN)):
        log("Force-trade window reached; attempting forced trade.")
        # choose highest-volume symbol
        best = None
        best_vol = 0
        for s in SYMBOLS:
            try:
                df = fetch_recent_bars(s, minutes=30)
                if df is None or df.empty:
                    continue
                vol = int(df["volume"].iloc[-1])
                if vol > best_vol:
                    best_vol = vol
                    best = s
            except Exception:
                continue
        if best:
            ok = force_trade(best)
            if ok:
                state["daily_trades"] += 1
                state["per_symbol"][best] = state["per_symbol"].get(best, 0) + 1
                save_state(state)
                return
        log("Forced trade attempt failed or no symbol available.")
    # no trade executed
    log("Run complete. No trade executed this run.")

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Unhandled exception: " + repr(e))
        log(traceback.format_exc())
