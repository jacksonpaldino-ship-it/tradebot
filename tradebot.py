#!/usr/bin/env python3
"""
SPY 1-minute scalper with broker-side bracket orders (multi-entry intraday).

Features:
- Uses alpaca-py TradingClient (new SDK).
- Uses yfinance for 1m historical data (last 7 days limitation).
- Builds the opening range (9:30-9:45 ET) on 1m bars.
- Long on breakout above OR high, short on breakdown below OR low.
- Attaches broker-side bracket order (TP & SL) when submitting.
- Re-entry cooldown and max entries/day guard.
- Logs each attempted/placed trade to CSV.
- DRY_RUN default True (simulate only).
"""

import os
import json
import math
import csv
from datetime import datetime, date, time as dtime, timedelta
import pytz
import time

import numpy as np
import pandas as pd
import yfinance as yf

# Alpaca-py imports
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import MarketOrderRequest

# ---------------- CONFIG ----------------
DRY_RUN = True                 # True => don't submit real orders
PAPER = True                   # TradingClient(..., paper=True)
SYMBOL = "SPY"

# Opening range window (minutes)
OPEN_START = dtime(9, 30)
OPEN_END   = dtime(9, 45)     # 15-minute opening range

# Trading hours guard
ET = pytz.timezone("US/Eastern")
CLOSE_TIME = dtime(15, 55)    # close positions before 15:55 ET

# Risk sizing
EQUITY_RISK_PCT = 0.005       # 0.5% of equity risk per trade
MAX_ALLOC_PCT = 0.30          # cap allocation per trade
MAX_TOTAL_EXPOSURE_PCT = 0.9

# ATR stop sizing
ATR_PERIOD = 14
MIN_STOP_PCT = 0.0015         # 0.15% min
MAX_STOP_PCT = 0.05           # 5% max
TP_MULTIPLIER = 2.0

# Re-entry controls
REENTRY_COOLDOWN_MIN = 20     # minutes between entries
MAX_ENTRIES_PER_DAY = 8

# Persistence/log paths
STATE_FILE = "scalper_state.json"
LOG_FILE = "scalper_trades.csv"

# YFinance constraints: 1m bars only available for last ~7 days
YF_LOOKBACK_DAYS = 7

# Alpaca credentials from env (use GitHub Secrets)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ---------------- Alpaca client ----------------
if ALPACA_API_KEY is None or ALPACA_SECRET_KEY is None:
    print("ALPACA API keys not found in environment — running in DRY_RUN-only mode.")
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# ---------------- utilities ----------------
def now_et():
    return datetime.now(ET)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "date": date.today().isoformat(),
        "opening_range": None,   # {"high":float,"low":float,"formed_at":str}
        "entries_today": 0,
        "last_entry_ts": None,
        "entries": []
    }

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

def reset_if_new_day(state):
    today = date.today().isoformat()
    if state.get("date") != today:
        return {
            "date": today,
            "opening_range": None,
            "entries_today": 0,
            "last_entry_ts": None,
            "entries": []
        }
    return state

def append_log(row):
    fieldnames = ["timestamp","date","symbol","side","mode","qty","entry_price","stop_price","tp_price","order_id","dry_run"]
    newfile = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if newfile:
            writer.writeheader()
        writer.writerow({k: row.get(k,"") for k in fieldnames})

# ---------------- data helpers ----------------
def fetch_1m(symbol, days=YF_LOOKBACK_DAYS):
    # yfinance 1m bars - limited to last ~7 days
    df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
    if df.empty:
        return df
    # drop MultiIndex if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    # ensure tz-aware index localized to UTC then convert to ET
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
    except Exception:
        pass
    return df

def compute_atr(df, n=ATR_PERIOD):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr

def get_account_equity():
    try:
        acct = trading_client.get_account()
        return float(acct.equity)
    except Exception as e:
        print("Could not fetch account equity:", e)
        return 100000.0

def total_open_exposure():
    try:
        pos = trading_client.get_all_positions()
        return sum(float(p.market_value) for p in pos)
    except Exception:
        return 0.0

def get_position_qty(symbol):
    try:
        pos = trading_client.get_all_positions()
        for p in pos:
            if p.symbol == symbol:
                return int(float(p.qty))
        return 0
    except Exception:
        return 0

# ---------------- order helpers (bracket) ----------------
def submit_bracket_long(symbol, qty, stop_price, tp_price):
    if qty <= 0:
        return None
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] BRACKET LONG: {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id":"dryrun_long","symbol":symbol,"qty":qty}
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": f"{tp_price:.2f}"},
            stop_loss={"stop_price": f"{stop_price:.2f}"}
        )
        order = trading_client.submit_order(market_order_request=req)
        print("Submitted bracket long:", getattr(order, "id", "unknown"))
        return order
    except Exception as e:
        print("Error submitting bracket long:", e)
        return None

def submit_bracket_short(symbol, qty, stop_price, tp_price):
    if qty <= 0:
        return None
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] BRACKET SHORT: {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id":"dryrun_short","symbol":symbol,"qty":qty}
    try:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,   # open short
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": f"{tp_price:.2f}"},
            stop_loss={"stop_price": f"{stop_price:.2f}"}
        )
        order = trading_client.submit_order(market_order_request=req)
        print("Submitted bracket short:", getattr(order, "id", "unknown"))
        return order
    except Exception as e:
        print("Error submitting bracket short:", e)
        return None

# ---------------- core logic ----------------
def form_opening_range(state):
    if state.get("opening_range"):
        return state
    df = fetch_1m(SYMBOL)
    if df.empty:
        print("No 1m bars available yet.")
        return state
    today = date.today()
    todays = df[df.index.date == today]
    if todays.empty:
        print("No intraday bars for today")
        return state
    mask = (todays.index.time >= OPEN_START) & (todays.index.time < OPEN_END)
    or_bars = todays.loc[mask]
    if or_bars.empty or len(or_bars) < 1:
        print("Opening range not yet formed.")
        return state
    high = float(or_bars["High"].max())
    low  = float(or_bars["Low"].min())
    state["opening_range"] = {"high": high, "low": low, "formed_at": or_bars.index[0].isoformat()}
    print(f"Opening range formed: high={high:.4f}, low={low:.4f}")
    save_state(state)
    return state

def compute_qty_stop(price, atr, equity, side="long"):
    # stop distance
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
    qty = max(0, min(raw_qty, cap_qty))
    if qty <= 0:
        return 0, stop_dist, stop_dist
    if side == "long":
        stop_price = round(price - stop_dist, 4)
        tp_price = round(price + stop_dist * TP_MULTIPLIER, 4)
    else:
        stop_price = round(price + stop_dist, 4)
        tp_price = round(price - stop_dist * TP_MULTIPLIER, 4)
    return int(qty), stop_price, tp_price

def attempt_entries(state):
    if not state.get("opening_range"):
        print("OR not formed; skipping entries")
        return state
    if state.get("entries_today", 0) >= MAX_ENTRIES_PER_DAY:
        print("Reached max entries today.")
        return state
    last_ts = state.get("last_entry_ts")
    if last_ts:
        last = datetime.fromisoformat(last_ts)
        if (now_et() - last) < timedelta(minutes=REENTRY_COOLDOWN_MIN):
            print("In cooldown; skip entry.")
            return state

    df = fetch_1m(SYMBOL)
    if df.empty:
        return state
    todays = df[df.index.date == date.today()]
    if todays.empty:
        return state
    latest = todays.iloc[-1]
    price = float(latest["Close"])

    OR_high = state["opening_range"]["high"]
    OR_low  = state["opening_range"]["low"]
    buffer = 0.0006  # small buffer (0.06%)

    equity = get_account_equity()
    atr_series = compute_atr(df)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None

    # Long breakout
    if price > OR_high * (1 + buffer):
        if get_position_qty(SYMBOL) > 0:
            print("Already long; skip new long.")
            return state
        qty, stop_price, tp_price = compute_qty_stop(price, atr, equity, "long")
        if qty <= 0:
            print("Computed qty 0 for long; skip.")
            return state
        order = submit_bracket_long(SYMBOL, qty, stop_price, tp_price)
        order_id = getattr(order, "id", (order or {}).get("id"))
        log = {
            "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
            "symbol": SYMBOL, "side": "LONG", "mode": "BREAKOUT",
            "qty": qty, "entry_price": price, "stop_price": stop_price, "tp_price": tp_price,
            "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)
        }
        append_log(log)
        state["last_entry_ts"] = now_et().isoformat()
        state["entries_today"] = state.get("entries_today",0) + 1
        state.setdefault("entries", []).append(log)
        save_state(state)
        return state

    # Short breakdown
    if price < OR_low * (1 - buffer):
        # ensure no existing short (we check long qty only; alpaca returns positive longs)
        pos_qty = get_position_qty(SYMBOL)
        if pos_qty < 0:
            print("Already short; skip short (pos_qty<0).")
            return state
        qty, stop_price, tp_price = compute_qty_stop(price, atr, equity, "short")
        if qty <= 0:
            print("Computed qty 0 for short; skip.")
            return state
        order = submit_bracket_short(SYMBOL, qty, stop_price, tp_price)
        order_id = getattr(order, "id", (order or {}).get("id"))
        log = {
            "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
            "symbol": SYMBOL, "side": "SHORT", "mode": "BREAKDOWN",
            "qty": qty, "entry_price": price, "stop_price": stop_price, "tp_price": tp_price,
            "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)
        }
        append_log(log)
        state["last_entry_ts"] = now_et().isoformat()
        state["entries_today"] = state.get("entries_today",0) + 1
        state.setdefault("entries", []).append(log)
        save_state(state)
        return state

    # nothing
    print(f"No breakout/breakdown: price={price:.4f} OR_high={OR_high:.4f} OR_low={OR_low:.4f}")
    return state

def close_all_before_close():
    now = now_et()
    if now.time() < CLOSE_TIME:
        return
    pos_qty = get_position_qty(SYMBOL)
    if pos_qty == 0:
        print("No positions at EOD to close.")
        return
    # close using market order
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] Closing position qty={pos_qty} for {SYMBOL}")
        append_log({
            "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
            "symbol": SYMBOL, "side": "CLOSE", "mode": "EOD_CLOSE",
            "qty": pos_qty, "entry_price": "", "stop_price": "", "tp_price": "",
            "order_id": "dryrun_close", "dry_run": True
        })
        return
    try:
        # positive qty => sell to close longs
        if pos_qty > 0:
            req = MarketOrderRequest(symbol=SYMBOL, qty=pos_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        else:
            req = MarketOrderRequest(symbol=SYMBOL, qty=abs(pos_qty), side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        order = trading_client.submit_order(market_order_request=req)
        print("Submitted EOD close order id:", getattr(order,"id",""))
        append_log({
            "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
            "symbol": SYMBOL, "side": "CLOSE", "mode": "EOD_CLOSE",
            "qty": pos_qty, "entry_price": "", "stop_price": "", "tp_price": "",
            "order_id": getattr(order,"id",""), "dry_run": False
        })
    except Exception as e:
        print("Error during EOD close:", e)

# ---------------- main run ----------------
def run_once():
    state = load_state()
    state = reset_if_new_day(state)

    # form opening range once bars exist
    state = form_opening_range(state)

    # attempt to enter (if before close)
    if now_et().time() < CLOSE_TIME:
        state = attempt_entries(state)
    else:
        close_all_before_close()

    save_state(state)
    print("Run complete.")

if __name__ == "__main__":
    run_once()
