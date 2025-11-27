#!/usr/bin/env python3
"""
tradebot_spy_realtime.py

Continuous intraday SPY day-trading bot (Option A improved) — checks continuously during market hours.

- DRY_RUN=True by default (safe).
- Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in environment to submit real orders.
- Uses alpaca-py OrderRequest for broker-side bracket orders.

Save as tradebot_spy_realtime.py and run: python tradebot_spy_realtime.py
"""

import os
import json
import math
import csv
import signal
import sys
from datetime import datetime, date, time as dtime, timedelta
import time as time_mod
import pytz
import traceback

import numpy as np
import pandas as pd
import yfinance as yf

# Attempt to import alpaca-py. If unavailable, we'll run in DRY_RUN.
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
    from alpaca.trading.requests import OrderRequest
    ALPACA_PY_AVAILABLE = True
except Exception:
    ALPACA_PY_AVAILABLE = False

# ---------------- CONFIG ----------------
DRY_RUN = True                 # Set False to submit orders (ONLY after testing)
PAPER = True                   # Alpaca paper account
SYMBOL = "SPY"

# Opening Range
OPEN_START = dtime(9, 30)
OPEN_END   = dtime(9, 45)      # 15 minute opening range

# Timezone
ET = pytz.timezone("US/Eastern")

# Market-close behavior
CLOSE_TIME = dtime(15, 55)

# How often to check signals (seconds)
CHECK_INTERVAL_SECONDS = 60    # default 1 minute; set lower for more active bot but beware rate limits

# Risk / sizing
EQUITY_RISK_PCT = 0.004        # 0.4% equity risk per trade
MAX_ALLOC_PCT = 0.25
MAX_TOTAL_EXPOSURE_PCT = 0.90

# ATR parameters
ATR_PERIOD = 14
MIN_STOP_PCT = 0.0015
MAX_STOP_PCT = 0.06
TP_MULT = 2.0

# Behavioral limits
REENTRY_COOLDOWN_MIN = 25
MAX_ENTRIES_PER_DAY = 6

# Data sources
YF_LOOKBACK_DAYS = 7   # yfinance 1m available ~last 7 days

# Files
STATE_FILE = "realtime_state.json"
LOG_FILE = "realtime_trades.csv"

# Alpaca credentials (env)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# If alpaca-py is installed and keys present, create client
if ALPACA_PY_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
else:
    client = None
    if not ALPACA_PY_AVAILABLE:
        print("alpaca-py not installed; running DRY_RUN only.")
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Alpaca keys missing; running DRY_RUN only.")
    DRY_RUN = True

# ---------------- Utilities ----------------
def now_et():
    return datetime.now(ET)

def is_market_open_now():
    now = now_et().time()
    return OPEN_START <= now <= dtime(16, 0)  # market hours 9:30-16:00 ET

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"date": date.today().isoformat(), "opening_range": None, "entries_today": 0, "last_entry_ts": None, "entries": []}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

def reset_if_new_day(state):
    today = date.today().isoformat()
    if state.get("date") != today:
        return {"date": today, "opening_range": None, "entries_today": 0, "last_entry_ts": None, "entries": []}
    return state

def append_log(row):
    fieldnames = ["timestamp","date","symbol","side","mode","qty","entry_price","stop_price","tp_price","order_id","dry_run"]
    newfile = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if newfile:
            writer.writeheader()
        writer.writerow({k: row.get(k,"") for k in fieldnames})

# ---------------- Data helpers ----------------
def fetch_1m(symbol, days=YF_LOOKBACK_DAYS):
    """Return intraday 1m DataFrame with ET tz-index; robust to multiindex."""
    df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.droplevel(0)
        except Exception:
            pass
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
    except Exception:
        pass
    return df

def vwap(series_df):
    typical = (series_df['High'] + series_df['Low'] + series_df['Close']) / 3
    pv = typical * series_df['Volume']
    return pv.cumsum() / series_df['Volume'].cumsum()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, min_periods=n).mean()
    ma_down = down.ewm(alpha=1/n, min_periods=n).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_atr(df, n=ATR_PERIOD):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def safe_last_float(series, offset=-1):
    """Return float from series safely (offset -1 last, -2 prev)."""
    try:
        return float(series.iloc[offset])
    except Exception:
        s = series.tail(abs(offset))
        if not s.empty:
            # take last element
            return float(s.values[-1])
        raise

def get_account_equity():
    if DRY_RUN or client is None:
        # if you want to simulate, you can hardcode or read a file
        return 100000.0
    try:
        acct = client.get_account()
        return float(acct.equity)
    except Exception as e:
        print("Error fetching account equity:", e)
        return 100000.0

def get_position_qty(symbol):
    if DRY_RUN or client is None:
        return 0
    try:
        pos = client.get_all_positions()
        for p in pos:
            if p.symbol == symbol:
                return int(float(p.qty))
        return 0
    except Exception:
        return 0

# ---------------- Order helpers ----------------
def submit_bracket_long(symbol, qty, stop_price, tp_price):
    if qty <= 0:
        return None
    if DRY_RUN or client is None:
        print(f"[DRY_RUN] BUY {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id": "dry_buy"}
    try:
        order = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": f"{tp_price:.2f}"},
            stop_loss={"stop_price": f"{stop_price:.2f}"}
        )
        resp = client.submit_order(order)
        print("Submitted bracket buy id:", getattr(resp, "id", "unknown"))
        return resp
    except Exception as e:
        print("Error submitting bracket long:", e)
        traceback.print_exc()
        return None

def submit_bracket_short(symbol, qty, stop_price, tp_price):
    if qty <= 0:
        return None
    if DRY_RUN or client is None:
        print(f"[DRY_RUN] SELL {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id": "dry_sell"}
    try:
        order = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": f"{tp_price:.2f}"},
            stop_loss={"stop_price": f"{stop_price:.2f}"}
        )
        resp = client.submit_order(order)
        print("Submitted bracket sell id:", getattr(resp, "id", "unknown"))
        return resp
    except Exception as e:
        print("Error submitting bracket short:", e)
        traceback.print_exc()
        return None

# ---------------- Strategy logic ----------------
def form_opening_range(state, symbol=SYMBOL):
    if state.get("opening_range"):
        return state
    df = fetch_1m(symbol)
    if df.empty:
        print("No data from yfinance; OR can't form.")
        return state
    today = date.today()
    todays = df[df.index.date == today]
    if todays.empty:
        print("No intraday bars for today.")
        return state
    mask = (todays.index.time >= OPEN_START) & (todays.index.time < OPEN_END)
    or_bars = todays.loc[mask]
    if or_bars.empty:
        print("Opening range not complete yet.")
        return state
    high = float(or_bars["High"].max())
    low  = float(or_bars["Low"].min())
    state["opening_range"] = {"high": high, "low": low, "formed_at": or_bars.index[0].isoformat()}
    print(f"Opening range formed: high={high:.4f}, low={low:.4f}")
    save_state(state)
    return state

def compute_qty_stop(price, atr, equity, side="long"):
    if atr and not np.isnan(atr) and atr > 0:
        stop_dist = max(atr, price * MIN_STOP_PCT)
    else:
        stop_dist = max(price * MIN_STOP_PCT, 0.01)
    stop_dist = min(stop_dist, price * MAX_STOP_PCT)
    risk_budget = equity * EQUITY_RISK_PCT
    if stop_dist <= 0:
        return 0, 0, 0
    raw_qty = math.floor(risk_budget / stop_dist)
    cap_qty = math.floor((equity * MAX_ALLOC_PCT) / price)
    qty = int(max(0, min(raw_qty, cap_qty)))
    if qty <= 0:
        return 0, stop_dist, stop_dist
    if side == "long":
        stop_price = round(price - stop_dist, 4)
        tp_price = round(price + stop_dist * TP_MULT, 4)
    else:
        stop_price = round(price + stop_dist, 4)
        tp_price = round(price - stop_dist * TP_MULT, 4)
    return qty, stop_price, tp_price

def should_enter_or(state, price):
    or_high = state["opening_range"]["high"]
    or_low  = state["opening_range"]["low"]
    buf = 0.0006
    if price > or_high * (1 + buf):
        return "long_or"
    if price < or_low * (1 - buf):
        return "short_or"
    return None

def should_enter_vwap(df):
    if len(df) < 3:
        return None
    v = vwap(df)
    price = safe_last_float(df["Close"], -1)
    prev_price = safe_last_float(df["Close"], -2)
    last_v = safe_last_float(v, -1)
    prev_v = safe_last_float(v, -2)
    # long: price above vwap now, prev below vwap and bounce
    if price > last_v and prev_price < prev_v and price > prev_price:
        r = safe_last_float(rsi(df["Close"]), -1)
        if r < 70:
            return "long_vwap"
    # short: price below vwap now, prev above vwap and drop
    if price < last_v and prev_price > prev_v and price < prev_price:
        r = safe_last_float(rsi(df["Close"]), -1)
        if r > 30:
            return "short_vwap"
    return None

def should_enter_ema(df):
    if len(df) < 10:
        return None
    ema9 = ema(df["Close"], 9)
    ema21 = ema(df["Close"], 21)
    price = safe_last_float(df["Close"], -1)
    prev = safe_last_float(df["Close"], -2)
    if safe_last_float(ema9, -1) > safe_last_float(ema21, -1):
        if prev < safe_last_float(ema9, -1) and price > prev:
            r = safe_last_float(rsi(df["Close"]), -1)
            if r < 70:
                return "long_ema"
    if safe_last_float(ema9, -1) < safe_last_float(ema21, -1):
        if prev > safe_last_float(ema9, -1) and price < prev:
            r = safe_last_float(rsi(df["Close"]), -1)
            if r > 30:
                return "short_ema"
    return None

def attempt_entries(state):
    if not state.get("opening_range"):
        # can't trade until OR formed
        return state
    if state.get("entries_today", 0) >= MAX_ENTRIES_PER_DAY:
        return state
    last_ts = state.get("last_entry_ts")
    if last_ts:
        last = datetime.fromisoformat(last_ts)
        if (now_et() - last) < timedelta(minutes=REENTRY_COOLDOWN_MIN):
            return state

    df = fetch_1m(SYMBOL)
    if df.empty:
        return state
    today_df = df[df.index.date == date.today()]
    if today_df.empty:
        return state

    price = safe_last_float(today_df["Close"], -1)
    equity = get_account_equity()
    atr = None
    atr_series = compute_atr(df)
    if not atr_series.dropna().empty:
        atr = float(atr_series.dropna().iloc[-1])

    # 1) OR breakout
    or_sig = should_enter_or(state, price)
    if or_sig == "long_or" and get_position_qty(SYMBOL) <= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
        if qty > 0:
            resp = submit_bracket_long(SYMBOL, qty, stop, tp)
            order_id = getattr(resp, "id", (resp or {}).get("id", ""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side": "LONG", "mode": "OR_BREAKOUT",
                   "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                   "order_id": order_id, "dry_run": DRY_RUN}
            append_log(log)
            state["last_entry_ts"] = now_et().isoformat()
            state["entries_today"] = state.get("entries_today", 0) + 1
            state.setdefault("entries", []).append(log)
            save_state(state)
            return state

    if or_sig == "short_or" and get_position_qty(SYMBOL) >= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
        if qty > 0:
            resp = submit_bracket_short(SYMBOL, qty, stop, tp)
            order_id = getattr(resp, "id", (resp or {}).get("id", ""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side": "SHORT", "mode": "OR_BREAKDOWN",
                   "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                   "order_id": order_id, "dry_run": DRY_RUN}
            append_log(log)
            state["last_entry_ts"] = now_et().isoformat()
            state["entries_today"] = state.get("entries_today", 0) + 1
            state.setdefault("entries", []).append(log)
            save_state(state)
            return state

    # 2) VWAP pullback
    vwap_sig = should_enter_vwap(today_df)
    if vwap_sig == "long_vwap" and get_position_qty(SYMBOL) <= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
        if qty > 0:
            resp = submit_bracket_long(SYMBOL, qty, stop, tp)
            order_id = getattr(resp, "id", (resp or {}).get("id", ""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side": "LONG", "mode": "VWAP_PULL",
                   "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                   "order_id": order_id, "dry_run": DRY_RUN}
            append_log(log)
            state["last_entry_ts"] = now_et().isoformat()
            state["entries_today"] = state.get("entries_today", 0) + 1
            state.setdefault("entries", []).append(log)
            save_state(state)
            return state

    if vwap_sig == "short_vwap" and get_position_qty(SYMBOL) >= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
        if qty > 0:
            resp = submit_bracket_short(SYMBOL, qty, stop, tp)
            order_id = getattr(resp, "id", (resp or {}).get("id", ""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side": "SHORT", "mode": "VWAP_PULL",
                   "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                   "order_id": order_id, "dry_run": DRY_RUN}
            append_log(log)
            state["last_entry_ts"] = now_et().isoformat()
            state["entries_today"] = state.get("entries_today", 0) + 1
            state.setdefault("entries", []).append(log)
            save_state(state)
            return state

    # 3) EMA pullback
    ema_sig = should_enter_ema(today_df)
    if ema_sig == "long_ema" and get_position_qty(SYMBOL) <= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
        if qty > 0:
            resp = submit_bracket_long(SYMBOL, qty, stop, tp)
            order_id = getattr(resp, "id", (resp or {}).get("id", ""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side": "LONG", "mode": "EMA_PULL",
                   "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                   "order_id": order_id, "dry_run": DRY_RUN}
            append_log(log)
            state["last_entry_ts"] = now_et().isoformat()
            state["entries_today"] = state.get("entries_today", 0) + 1
            state.setdefault("entries", []).append(log)
            save_state(state)
            return state

    if ema_sig == "short_ema" and get_position_qty(SYMBOL) >= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
        if qty > 0:
            resp = submit_bracket_short(SYMBOL, qty, stop, tp)
            order_id = getattr(resp, "id", (resp or {}).get("id", ""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side": "SHORT", "mode": "EMA_PULL",
                   "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                   "order_id": order_id, "dry_run": DRY_RUN}
            append_log(log)
            state["last_entry_ts"] = now_et().isoformat()
            state["entries_today"] = state.get("entries_today", 0) + 1
            state.setdefault("entries", []).append(log)
            save_state(state)
            return state

    # nothing to do
    return state

def close_all_positions_if_eod():
    now = now_et()
    if now.time() < CLOSE_TIME:
        return
    qty = get_position_qty(SYMBOL)
    if qty == 0:
        return
    if DRY_RUN or client is None:
        print(f"[DRY_RUN] Would close position qty={qty} at EOD")
        append_log({"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                    "symbol": SYMBOL, "side": "EOD_CLOSE", "mode": "EOD", "qty": qty, "order_id": "dry_eod", "dry_run": True})
        return
    try:
        if qty > 0:
            req = OrderRequest(symbol=SYMBOL, qty=qty, side=OrderSide.SELL, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
        else:
            req = OrderRequest(symbol=SYMBOL, qty=abs(qty), side=OrderSide.BUY, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
        resp = client.submit_order(req)
        append_log({"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                    "symbol": SYMBOL, "side": "EOD_CLOSE", "mode": "EOD", "qty": qty, "order_id": getattr(resp,"id",""), "dry_run": False})
        print("EOD close submitted:", getattr(resp,"id",""))
    except Exception as e:
        print("Error closing positions at EOD:", e)
        traceback.print_exc()

# Graceful shutdown
stop_requested = False
def _signal_handler(sig, frame):
    global stop_requested
    print("Shutdown requested — exiting after current loop.")
    stop_requested = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ---------------- Main loop ----------------
def main_loop():
    state = load_state()
    state = reset_if_new_day(state)
    print(f"Starting real-time bot loop (DRY_RUN={DRY_RUN}) — checking every {CHECK_INTERVAL_SECONDS}s.")
    while not stop_requested:
        try:
            state = reset_if_new_day(state)
            # Only operate during market hours
            if is_market_open_now():
                # Form OR once
                state = form_opening_range(state, SYMBOL)
                # Attempt entries
                state = attempt_entries(state)
                # If close-of-day reached, close positions
                close_all_positions_if_eod()
            else:
                # outside market hours: print sleeping message in morning/evening
                now = now_et()
                print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market closed or not in trading window. Sleeping.")
                # If OR exists but it's a new day, reset next loop
                if state.get("opening_range") and state.get("date") != date.today().isoformat():
                    state = reset_if_new_day(state)

            # persist state
            save_state(state)
        except Exception as e:
            print("Unhandled error in main loop:", e)
            traceback.print_exc()

        # Sleep until next check, but break early if shutdown requested
        for _ in range(int(max(1, CHECK_INTERVAL_SECONDS))):
            if stop_requested:
                break
            time_mod.sleep(1)

    print("Bot stopped. Exiting.")

if __name__ == "__main__":
    main_loop()
