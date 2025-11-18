#!/usr/bin/env python3
"""
SPY Opening Range Breakout/Breakdown bot with broker-side bracket orders.
- DRY_RUN True by default (no real orders). Flip to False after testing.
- Uses alpaca-py (TradingClient) and yfinance.
- Opening range 9:30-9:45 ET. Long on breakout above high. Short on breakdown below low.
- Attaches bracket orders (take profit + stop loss) at submit.
- Logs every attempted/filled entry to orb_trades.csv.
"""

import os
import json
import math
from datetime import datetime, date, time as dtime, timedelta
import pytz
import csv
import time

import numpy as np
import pandas as pd
import yfinance as yf

# Alpaca-py
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, TakeProfit, StopLoss

# ---------------- CONFIG ----------------
DRY_RUN = False                 # True = do not submit real orders
PAPER = True                    # use paper account
SYMBOL = "SPY"

OPEN_START = dtime(9, 30)
OPEN_END   = dtime(9, 45)

CLOSE_TIME = dtime(15, 55)     # close before this time

ET = pytz.timezone("US/Eastern")

STATE_FILE = "orb_state_spy.json"
LOG_FILE = "orb_trades.csv"

# sizing / risk
EQUITY_RISK_PCT = 0.005        # risk per trade (0.5% of equity)
MAX_ALLOC_PCT = 0.25
MAX_TOTAL_EXPOSURE_PCT = 0.90

ATR_PERIOD = 14
MIN_STOP_PCT = 0.002           # 0.2%
MAX_STOP_PCT = 0.05            # 5%
TP_MULTIPLIER = 2.0

# re-entry
REENTRY_COOLDOWN_MIN = 30
MAX_ENTRIES_PER_DAY = 6

# Alpaca credentials via env / GitHub secrets
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ---------------- CLIENT ----------------
if ALPACA_API_KEY is None or ALPACA_SECRET_KEY is None:
    print("ALPACA_API_KEY or ALPACA_SECRET_KEY not found in environment — DRY_RUN only if keys missing.")
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# ---------------- UTIL ----------------
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
        "opening_range": None,
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

def append_log(row: dict):
    header = ["timestamp", "date", "symbol", "side", "mode", "qty", "entry_price", "stop_price", "tp_price", "order_id", "dry_run"]
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in header})

# ---------------- DATA helpers ----------------
def fetch_5m(symbol, lookback_days=5):
    df = yf.download(symbol, period=f"{lookback_days}d", interval="5m", progress=False)
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    return df

def tz_convert_index_to_et(df):
    if df.empty:
        return df
    try:
        # if naive, localize to UTC then convert to ET
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
    except Exception:
        # ignore conversion errors
        pass
    return df

def compute_atr_from_df(df, n=ATR_PERIOD):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr

def get_account_equity():
    try:
        acct = trading_client.get_account()
        return float(acct.equity)
    except Exception as e:
        # fallback mock equity for DRY_RUN/no-keys
        print("Warning fetching account equity:", e)
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

# ---------------- ORDER helpers (bracket) ----------------
def submit_bracket_long(symbol, qty, entry_price, stop_price, tp_price):
    """Submit long bracket: buy market + attach stop_loss & take_profit via MarketOrderRequest"""
    if qty <= 0:
        print("qty <= 0; skipping buy")
        return None
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] BRACKET LONG {symbol} qty={qty} entry={entry_price} stop={stop_price} tp={tp_price}")
        return {"id": "dryrun_long", "symbol": symbol, "qty": qty}
    try:
        take_profit = TakeProfit(limit_price=f"{tp_price:.2f}")
        stop_loss = StopLoss(stop_price=f"{stop_price:.2f}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit=take_profit,
            stop_loss=stop_loss
        )
        order = trading_client.submit_order(market_order_request=req)
        print("Submitted bracket long, id:", order.id)
        return order
    except Exception as e:
        print("Bracket long error:", e)
        return None

def submit_bracket_short(symbol, qty, entry_price, stop_price, tp_price):
    """Submit short bracket: sell short market + attach stop_loss & take_profit (stop above entry, tp below)"""
    if qty <= 0:
        print("qty <= 0; skipping short")
        return None
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] BRACKET SHORT {symbol} qty={qty} entry={entry_price} stop={stop_price} tp={tp_price}")
        return {"id": "dryrun_short", "symbol": symbol, "qty": qty}
    try:
        # For a short, take_profit limit price should be lower than entry, stop_loss above entry
        take_profit = TakeProfit(limit_price=f"{tp_price:.2f}")
        stop_loss = StopLoss(stop_price=f"{stop_price:.2f}")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,  # SELL to open short
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            take_profit=take_profit,
            stop_loss=stop_loss
        )
        order = trading_client.submit_order(market_order_request=req)
        print("Submitted bracket short, id:", order.id)
        return order
    except Exception as e:
        print("Bracket short error:", e)
        return None

# ---------------- CORE logic ----------------
def form_opening_range(state):
    if state.get("opening_range"):
        return state
    df = fetch_5m(SYMBOL, lookback_days=2)
    if df.empty:
        print("No 5m bars yet.")
        return state
    df = tz_convert_index_to_et(df)
    today = date.today()
    todays = df[df.index.date == today]
    if todays.empty:
        print("No intraday bars for today.")
        return state
    mask = (todays.index.time >= OPEN_START) & (todays.index.time < OPEN_END)
    opening_bars = todays.loc[mask]
    if opening_bars.empty:
        print("Opening bars not complete yet.")
        return state
    high = float(opening_bars["High"].max())
    low  = float(opening_bars["Low"].min())
    state["opening_range"] = {"high": high, "low": low, "formed_at": opening_bars.index[0].isoformat()}
    print(f"Opening range formed: high={high:.4f}, low={low:.4f}")
    save_state(state)
    return state

def compute_qty_and_stop(price, atr, equity, side="long"):
    # determine stop distance
    if atr and not np.isnan(atr) and atr > 0:
        stop_dist = max(atr, price * MIN_STOP_PCT)
    else:
        stop_dist = price * MIN_STOP_PCT
    stop_dist = min(stop_dist, price * MAX_STOP_PCT)
    risk_budget = equity * EQUITY_RISK_PCT
    if stop_dist <= 0:
        return 0, 0
    raw_qty = math.floor(risk_budget / stop_dist)
    cap_qty = math.floor((equity * MAX_ALLOC_PCT) / price)
    qty = int(max(0, min(raw_qty, cap_qty)))
    if side == "short":
        stop_price = round(price + stop_dist, 4)   # stop above entry
        tp_price = round(price - stop_dist * TP_MULTIPLIER, 4)  # profit below
    else:
        stop_price = round(price - stop_dist, 4)
        tp_price = round(price + stop_dist * TP_MULTIPLIER, 4)
    return qty, stop_price, tp_price

def check_and_trade(state):
    # ensure opening range formed
    if not state.get("opening_range"):
        print("OR not formed; skipping trade check.")
        return state

    # entries/day cap
    if state.get("entries_today", 0) >= MAX_ENTRIES_PER_DAY:
        print("Reached max entries today.")
        return state

    # cooldown
    last_ts = state.get("last_entry_ts")
    if last_ts:
        last = datetime.fromisoformat(last_ts)
        if (now_et() - last) < timedelta(minutes=REENTRY_COOLDOWN_MIN):
            print("In cooldown; skipping.")
            return state

    # latest bar
    df = fetch_5m(SYMBOL, lookback_days=2)
    if df.empty:
        return state
    df = tz_convert_index_to_et(df)
    last_row = df.iloc[-1]
    price = float(last_row["Close"])

    OR_high = state["opening_range"]["high"]
    OR_low  = state["opening_range"]["low"]
    buffer = 0.0005  # small tolerance

    equity = get_account_equity()
    atr_series = compute_atr_from_df(df, ATR_PERIOD)
    atr = float(atr_series.iloc[-1]) if not atr_series.empty else None

    # check breakout long
    if price > OR_high * (1 + buffer):
        # ensure not already long
        if get_position_qty(SYMBOL) > 0:
            print("Already long; skip long entry.")
        else:
            qty, stop_price, tp_price = compute_qty_and_stop(price, atr, equity, side="long")
            if qty <= 0:
                print("Qty 0 for long; skip.")
            else:
                order = submit_bracket_long(SYMBOL, qty, price, stop_price, tp_price)
                order_id = getattr(order, "id", (order or {}).get("id"))
                log_row = {
                    "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                    "symbol": SYMBOL, "side": "LONG", "mode": "BREAKOUT",
                    "qty": qty, "entry_price": price, "stop_price": stop_price, "tp_price": tp_price,
                    "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)
                }
                append_log(log_row)
                state["last_entry_ts"] = now_et().isoformat()
                state["entries_today"] = state.get("entries_today", 0) + 1
                state.setdefault("entries", []).append(log_row)
                save_state(state)
        return state

    # check breakdown short
    if price < OR_low * (1 - buffer):
        # ensure not already short (we check position qty; positive qty means long)
        pos_qty = get_position_qty(SYMBOL)
        if pos_qty < 0:
            print("Already short; skip short entry.")
        else:
            qty, stop_price, tp_price = compute_qty_and_stop(price, atr, equity, side="short")
            if qty <= 0:
                print("Qty 0 for short; skip.")
            else:
                order = submit_bracket_short(SYMBOL, qty, price, stop_price, tp_price)
                order_id = getattr(order, "id", (order or {}).get("id"))
                log_row = {
                    "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                    "symbol": SYMBOL, "side": "SHORT", "mode": "BREAKDOWN",
                    "qty": qty, "entry_price": price, "stop_price": stop_price, "tp_price": tp_price,
                    "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)
                }
                append_log(log_row)
                state["last_entry_ts"] = now_et().isoformat()
                state["entries_today"] = state.get("entries_today", 0) + 1
                state.setdefault("entries", []).append(log_row)
                save_state(state)
        return state

    print(f"No breakout/breakdown (price={price:.4f}, OR_high={OR_high:.4f}, OR_low={OR_low:.4f})")
    return state

def close_all_before_close():
    now = now_et()
    if now.time() < CLOSE_TIME:
        return
    pos_qty = get_position_qty(SYMBOL)
    if pos_qty == 0:
        print("No positions to close.")
        return
    # sell to close longs, buy to cover shorts (Alpaca submit market sell for positive qty, buy for negative?)
    # trading_client.submit_order with side SELL will close longs; to close shorts, submit BUY.
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] Closing all positions for {SYMBOL}, qty={pos_qty}")
        append_log({
            "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
            "symbol": SYMBOL, "side": "CLOSE", "mode": "EOD_CLOSE",
            "qty": pos_qty, "entry_price": "", "stop_price": "", "tp_price": "",
            "order_id": "dryrun_close", "dry_run": True
        })
        return
    try:
        # submit market order to close full position
        if pos_qty > 0:
            req = MarketOrderRequest(symbol=SYMBOL, qty=pos_qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
        else:
            # For negative qty (short), qty value returned will be negative? get_position_qty returns int positive for long only earlier.
            # But to be safe, if pos_qty < 0:
            req = MarketOrderRequest(symbol=SYMBOL, qty=abs(pos_qty), side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        order = trading_client.submit_order(market_order_request=req)
        print("Submitted close order id:", order.id)
        append_log({
            "timestamp": now_et().isoformat(), "date": date.today().isoformat(),
            "symbol": SYMBOL, "side": "CLOSE", "mode": "EOD_CLOSE",
            "qty": pos_qty, "entry_price": "", "stop_price": "", "tp_price": "",
            "order_id": getattr(order, "id", ""), "dry_run": False
        })
    except Exception as e:
        print("Error closing positions:", e)

# ---------------- MAIN RUN ----------------
def run_once():
    state = load_state()
    state = reset_if_new_day(state)

    # form opening range
    state = form_opening_range(state)

    # attempt entries
    now = now_et()
    if now.time() < CLOSE_TIME:
        state = check_and_trade(state)
    else:
        close_all_before_close()

    save_state(state)
    print("Run complete.")

if __name__ == "__main__":
    run_once()
