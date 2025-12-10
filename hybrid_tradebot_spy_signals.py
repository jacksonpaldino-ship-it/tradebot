#!/usr/bin/env python3
"""
Hybrid HF scalper (single-run)
- Uses alpaca-trade-api (REST) for orders/account
- Uses yfinance for minute bars (fast fallback, no Alpaca data subscription needed)
- Bracket orders attempted (TP + SL). Falls back to market buy if bracket fails.
- Persistent state to avoid loops and enforce daily caps.
- Single-run (safe to schedule every 10 minutes in GitHub Actions).
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
PER_SYMBOL_DAILY_CAP = 6
TP_PCT = 0.0020    # 0.20%
SL_PCT = 0.0015    # 0.15%
RISK_PER_TRADE = 0.02  # 2% equity risk
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 20_000
STATE_FILE = "hf_state.json"
TRADES_CSV = "hf_trades.csv"
LOG_FILE = "hf_bot.log"
TZ = pytz.timezone("US/Eastern")
EPS = 1e-9
MONITOR_TIMEOUT = 60 * 60 * 3  # 3 hours (safety, not used heavily)

# ---------------- Alpaca credentials (old SDK) ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. https://paper-api.alpaca.markets

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Missing Alpaca credentials. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- utilities ----------------
def now_et():
    return datetime.now(TZ)

def utcnow_str():
    return datetime.utcnow().isoformat()

def log(msg):
    line = f"{utcnow_str()} {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "has_open": False}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "has_open": False}

def save_state(s):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(s, f)
    except Exception:
        pass

def append_trade_row(row):
    header = ["utc_ts", "symbol", "side", "qty", "entry", "exit", "pnl", "note"]
    exists = os.path.exists(TRADES_CSV)
    try:
        with open(TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        log(f"append_trade_row error: {e}")

# ---------------- market data (yfinance minute bars) ----------------
def fetch_minute_bars_yf(symbol, minutes=120):
    """
    Use yfinance to fetch recent minute bars. Returns DataFrame with columns open, high, low, close, volume.
    """
    try:
        period = f"{max(2, minutes // 60 + 1)}d"  # safe period
        df = yf.download(tickers=symbol, period=period, interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        if "close" not in df.columns:
            return None
        # drop rows with zero volume (market closed times)
        df = df[~(df["volume"].isna())]
        df = df.tail(minutes)
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        log(f"fetch_minute_bars_yf error {symbol}: {e}")
        return None

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
def get_account_equity():
    try:
        acct = api.get_account()
        return float(acct.equity)
    except Exception as e:
        log(f"get_account_equity error: {e}")
        return None

def compute_qty(entry_price, atr):
    equity = get_account_equity()
    if equity is None or equity <= 0:
        return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    # cap nominal exposure to 50% equity (safety)
    cap = max(1, int((equity * 0.5) // entry_price))
    return min(qty, cap)

# ---------------- order helpers ----------------
def submit_bracket_order(symbol, qty, sl_price, tp_price):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(round(tp_price, 6))},
            stop_loss={"stop_price": str(round(sl_price, 6))}
        )
        log(f"Bracket submitted {symbol} qty={qty} tp={tp_price} sl={sl_price}")
        return order
    except Exception as e:
        log(f"submit_bracket_order error: {e}")
        return None

def submit_market_buy(symbol, qty):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day"
        )
        log(f"Market buy submitted {symbol} qty={qty}")
        return order
    except Exception as e:
        log(f"submit_market_buy error: {e}")
        return None

# ---------------- reconcile orders/positions ----------------
def has_open_positions():
    try:
        positions = api.list_positions()
        return len(positions) > 0
    except Exception as e:
        log(f"has_open_positions error: {e}")
        return False

# ---------------- main scanning and run_once ----------------
def run_once():
    log(f"Run start ET {now_et().isoformat()}")
    # check market open
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            log(f"Market closed; exiting. ({clock.timestamp})")
            return
    except Exception as e:
        log(f"clock error: {e}")
        # try to continue but safe-guard: if market appears closed, exit
        return

    # load/reset state
    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "has_open": False}
        save_state(state)

    # avoid entering if daily cap reached
    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log(f"Daily cap reached: {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return

    # avoid entering if any open positions (single-open-position rule)
    if has_open_positions():
        log("Existing open position found; skipping entry this run.")
        return

    # gather scored candidates
    candidates = []
    for sym in SYMBOLS:
        try:
            df = fetch_minute_bars_yf(sym, minutes=120)
            if df is None or df.empty or len(df) < 10:
                continue
            if int(df["volume"].iloc[-1]) < MIN_VOLUME:
                continue
            window = df.tail(60)
            price = float(window["close"].iloc[-1])
            vwap = compute_vwap(window)
            vw_gap = abs(price - vwap) / (vwap + EPS)
            macd_hist = compute_macd_hist(window)
            atr = compute_atr(window)
            vw_score = max(0.0, 1.0 - vw_gap / 0.01)  # 1% normalization
            vol_score = min(1.0, window["volume"].iloc[-1] / (window["volume"].mean() + EPS))
            macd_score = 1.0 if macd_hist > 0 else 0.0
            score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * macd_score
            candidates.append({"symbol": sym, "score": score, "price": price, "vwap": vwap, "atr": atr, "macd_hist": macd_hist})
        except Exception as e:
            log(f"score error {sym}: {e}")

    if not candidates:
        log("No candidates identified this run.")
        return

    candidates.sort(key=lambda x: x["score"], reverse=True)
    for cand in candidates:
        sym = cand["symbol"]
        per_sym = state["per_symbol"].get(sym, 0)
        if per_sym >= PER_SYMBOL_DAILY_CAP:
            log(f"Per-symbol cap reached for {sym}")
            continue

        entry_price = cand["price"]
        atr = cand["atr"]
        qty = compute_qty(entry_price, atr)
        if qty < 1:
            continue
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)

        # try bracket first
        br = submit_bracket_order(sym, qty, sl_price, tp_price)
        if br:
            # record and update state
            state["daily_trades"] += 1
            state["per_symbol"][sym] = per_sym + 1
            state["has_open"] = True
            save_state(state)
            append_trade_row([utcnow_str(), sym, "BUY_SUBMIT", qty, round(entry_price,6), None, None, "bracket_submitted"])
            log(f"Bracket order placed for {sym}. Exiting run (one entry per run).")
            return
        # fallback to market buy (less ideal)
        mk = submit_market_buy(sym, qty)
        if mk:
            # best-effort read filled price
            time.sleep(1.2)
            entry = None
            try:
                o = api.get_order(mk.id)
                entry = float(getattr(o, "filled_avg_price", None)) if getattr(o, "filled_avg_price", None) else entry_price
            except Exception:
                entry = entry_price
            state["daily_trades"] += 1
            state["per_symbol"][sym] = per_sym + 1
            state["has_open"] = True
            save_state(state)
            append_trade_row([utcnow_str(), sym, "BUY_FILLED", qty, round(entry,6), None, None, "market_filled"])
            log(f"Market buy executed for {sym} qty {qty} entry {entry}. Exiting run.")
            return

    log("No candidate placed due to caps/failures this run.")

if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Unhandled exception: " + repr(e))
        log(traceback.format_exc())
