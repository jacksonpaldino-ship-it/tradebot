#!/usr/bin/env python3
"""
hybrid_tradebot_v4.py

Ready-to-run single-invocation trade bot:
- yfinance for data (minute bars)
- alpaca-trade-api for orders/account
- ATR-based sizing, MACD+VWAP+volume scoring
- Bracket orders (TP/SL) preferred, market fallback
- Min 1 forced trade/day at FORCE_HOUR:FORCE_MIN ET if no trades yet
- Target ~5-10 trades/day (configurable)
- Allows multiple open positions up to MAX_OPEN_POSITIONS
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
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]   # universe
TARGET_TRADES_PER_DAY = 6                # target (aim for 5-10)
MAX_TRADES_PER_DAY = 10
PER_SYMBOL_DAILY_CAP = 4
MAX_OPEN_POSITIONS = 3                   # allow multiple concurrent positions
TP_PCT = 0.0020                          # 0.20% take profit
SL_PCT = 0.0015                          # 0.15% stop loss
RISK_PER_TRADE = 0.008                   # 0.8% equity risk per trade
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 1500                        # minute-bar realistic floor (lowered)
VWAP_BAND = 0.02                         # 2% VWAP band for scoring (looser)
SCORE_THRESHOLD = 0.12                   # lower threshold -> more trades
GUARANTEE_TRADE = True
FORCE_HOUR = 10                          # ET 10:05 forced trade window
FORCE_MIN = 5
STATE_FILE = "bot_state_v4.json"
TRADES_CSV = "trades_v4.csv"
LOG_FILE = "bot_v4.log"

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca client ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. https://paper-api.alpaca.markets
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Missing Alpaca secrets. Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Logging / state ----------------
def now_et():
    return datetime.now(TZ)

def utcnow_iso():
    return datetime.utcnow().isoformat()

def log(msg):
    line = f"{utcnow_iso()} {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "orders": []}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"date": None, "daily_trades": 0, "per_symbol": {}, "orders": []}

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

# ---------------- Market data (yfinance) ----------------
def fetch_minute_bars(symbol, minutes=200):
    """
    Fetch recent minute bars using yfinance. Returns DataFrame with open,high,low,close,volume.
    """
    try:
        days = max(1, (minutes // 60) + 1)
        period = f"{days}d"
        df = yf.download(tickers=symbol, period=period, interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        if not {"open","high","low","close","volume"}.issubset(df.columns):
            return None
        df = df[["open","high","low","close","volume"]]
        # remove tz info for simplicity
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
        return df.tail(minutes)
    except Exception as e:
        log(f"yfinance fetch error {symbol}: {e}")
        return None

# ---------------- Indicators ----------------
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
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = (macd - signal).iloc[-1]
    return float(hist)

# ---------------- Sizing ----------------
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
    # cap to not exceed a % of equity nominally
    max_nominal = int(max(1, math.floor((equity * 0.25) / entry_price)))
    return min(qty, max_nominal)

# ---------------- Orders ----------------
def submit_bracket_order(symbol, qty, sl_price, tp_price):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(round(tp_price,6))},
            stop_loss={"stop_price": str(round(sl_price,6))}
        )
        log(f"Bracket submitted: {symbol} qty={qty} tp={tp_price:.4f} sl={sl_price:.4f}")
        return order
    except Exception as e:
        log(f"submit_bracket_order error: {e}")
        return None

def submit_market_buy(symbol, qty):
    try:
        order = api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")
        log(f"Market buy submitted: {symbol} qty={qty}")
        return order
    except Exception as e:
        log(f"submit_market_buy error: {e}")
        return None

# ---------------- Utilities ----------------
def count_open_positions():
    try:
        pos = api.list_positions()
        return len(pos)
    except Exception as e:
        log(f"list_positions error: {e}")
        return 0

def has_open_positions():
    return count_open_positions() > 0

# ---------------- Scoring ----------------
def score_symbol(symbol):
    df = fetch_minute_bars(symbol, minutes=120)
    if df is None or df.empty or len(df) < 20:
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

    vol_score = min(1.0, float(window["volume"].iloc[-1]) / (float(window["volume"].mean()) + EPS))
    vw_score = max(0.0, 1.0 - vw_gap / VWAP_BAND)
    macd_score = 1.0 if macd_hist > 0 else 0.0

    score = 0.45 * vw_score + 0.35 * vol_score + 0.20 * macd_score
    return {
        "symbol": symbol,
        "score": float(score),
        "price": price,
        "vwap": vwap,
        "atr": atr,
        "macd_hist": macd_hist,
        "vol": volume_last
    }

# ---------------- Force trade ----------------
def force_trade(symbol):
    df = fetch_minute_bars(symbol, minutes=60)
    if df is None or df.empty:
        return False
    price = float(df["close"].iloc[-1])
    atr = compute_atr(df)
    equity = get_account_equity() or 0
    per_trade_risk = max(0.005, RISK_PER_TRADE / 3.0)
    risk_amount = equity * per_trade_risk
    per_share_risk = max(atr, price * 0.0005)
    qty = int(max(1, math.floor(risk_amount / (per_share_risk + EPS))))
    qty = min(qty, int(max(1, math.floor((equity * 0.15) / price))))
    if qty <= 0:
        return False
    tp = price * (1 + TP_PCT)
    sl = price * (1 - SL_PCT)
    order = submit_bracket_order(symbol, qty, sl, tp)
    if order:
        append_trade_row([utcnow_iso(), symbol, "FORCE_BUY", qty, round(price,6), None, None, "forced"])
        log(f"Forced trade placed on {symbol} qty={qty}")
        return True
    return False

# ---------------- RUN ONCE ----------------
def run_once():
    log(f"Run start ET {now_et().isoformat()}")

    # market open check
    try:
        clock = api.get_clock()
        if not getattr(clock, "is_open", False):
            log("Market closed; exiting.")
            return
    except Exception as e:
        log(f"get_clock error: {e}")
        # If clock can't be fetched, be conservative and exit
        return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date") != today:
        state = {"date": today, "daily_trades": 0, "per_symbol": {}, "orders": []}
        save_state(state)

    # daily cap
    if state["daily_trades"] >= MAX_TRADES_PER_DAY:
        log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return

    # limit concurrent positions
    open_pos = count_open_positions()
    if open_pos >= MAX_OPEN_POSITIONS:
        log(f"Max open positions reached ({open_pos}); skipping new entries.")
        return

    # Score symbols
    scored = []
    for s in SYMBOLS:
        try:
            info = score_symbol(s)
            if info is not None:
                scored.append(info)
        except Exception as e:
            log(f"score_symbol error {s}: {e}")

    # Try to place entries based on ranking
    if scored:
        scored.sort(key=lambda x: x["score"], reverse=True)
        for cand in scored:
            sym = cand["symbol"]
            # per-symbol cap
            used = state["per_symbol"].get(sym, 0)
            if used >= PER_SYMBOL_DAILY_CAP:
                log(f"Per-symbol cap reached for {sym}")
                continue
            # acceptance threshold
            if cand["score"] < SCORE_THRESHOLD:
                log(f"{sym} score {cand['score']:.3f} below threshold {SCORE_THRESHOLD}")
                continue
            # compute size and attempt order
            entry_price = cand["price"]
            atr = cand["atr"]
            qty = compute_qty(entry_price, atr)
            if qty < 1:
                log(f"{sym} qty computed < 1; skipping")
                continue
            tp = entry_price * (1 + TP_PCT)
            sl = entry_price * (1 - SL_PCT)
            order = submit_bracket_order(sym, qty, sl, tp)
            if order:
                state["daily_trades"] += 1
                state["per_symbol"][sym] = used + 1
                state["orders"].append({"id": getattr(order, "id", None), "symbol": sym})
                save_state(state)
                append_trade_row([utcnow_iso(), sym, "BUY_SUBMIT", qty, round(entry_price,6), None, None, f"score:{cand['score']:.3f}"])
                log(f"Placed bracket for {sym} qty={qty} score={cand['score']:.3f}")
                return
            # fallback market buy
            order2 = submit_market_buy(sym, qty)
            if order2:
                # try to read fill price
                time.sleep(1.0)
                try:
                    o = api.get_order(order2.id)
                    fill_price = getattr(o, "filled_avg_price", None) or entry_price
                    fill_price = float(fill_price)
                except Exception:
                    fill_price = entry_price
                state["daily_trades"] += 1
                state["per_symbol"][sym] = used + 1
                save_state(state)
                append_trade_row([utcnow_iso(), sym, "BUY_FILLED", qty, round(fill_price,6), None, None, f"score:{cand['score']:.3f}"])
                log(f"Market buy filled for {sym} qty={qty} price={fill_price:.4f}")
                return
        log("No scored candidate succeeded this run.")
    else:
        log("No scored candidates this run.")

    # Forced trade: ensure min 1 trade/day (run after FORCE_HOUR:FORCE_MIN)
    now = now_et()
    if GUARANTEE_TRADE and state.get("daily_trades", 0) == 0 and (now.hour > FORCE_HOUR or (now.hour == FORCE_HOUR and now.minute >= FORCE_MIN)):
        # choose symbol with highest recent volume
        best = None
        best_vol = 0
        for s in SYMBOLS:
            try:
                df = fetch_minute_bars(s, 30)
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
                log(f"Forced trade executed for {best}")
                return
        log("Forced trade attempt failed or no symbol available.")

    log("Run complete. No trade executed this run.")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        log("Unhandled exception: " + repr(e))
        log(traceback.format_exc())
