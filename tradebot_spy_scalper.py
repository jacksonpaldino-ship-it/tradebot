#!/usr/bin/env python3
"""
tradebot_spy_improved.py

Medium-frequency SPY day-trader (Option A):
- Opening-range breakout (9:30-9:45 ET)
- VWAP pullback entries
- 9-EMA pullback entries (trend-follow)
- Broker-side bracket orders (TP & SL) via alpaca-py OrderRequest
- ATR-based sizing & stops
- Cooldowns, daily entry cap, DRY_RUN default
- Logs trades to CSV and persists state to JSON
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

# Alpaca-py
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.trading.requests import OrderRequest

# ------------------ CONFIG ------------------
DRY_RUN = False                 # set False to submit real orders
PAPER = True                   # use paper account in TradingClient(...)
SYMBOL = "SPY"

# Opening range window
OPEN_START = dtime(9, 30)
OPEN_END   = dtime(9, 45)      # 15-minute opening range

# Trading day parameters
ET = pytz.timezone("US/Eastern")
CLOSE_TIME = dtime(15, 55)     # close before 3:55 PM ET

# Risk & sizing
EQUITY_RISK_PCT = 0.004        # risk per trade = 0.4% of equity
MAX_ALLOC_PCT = 0.25           # cap allocation per trade (25% of equity)
MAX_TOTAL_EXPOSURE_PCT = 0.9

# ATR stops
ATR_PERIOD = 14
MIN_STOP_PCT = 0.0015          # 0.15%
MAX_STOP_PCT = 0.06            # 6%
TP_MULT = 2.0

# Entries & cooldown
REENTRY_COOLDOWN_MIN = 25
MAX_ENTRIES_PER_DAY = 6

# Files
STATE_FILE = "improved_state.json"
LOG_FILE = "improved_trades.csv"

# YFinance lookback (1m data limited ~7 days)
YF_LOOKBACK_DAYS = 7

# Alpaca credentials (env / GitHub Secrets)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ------------------ CLIENT ------------------
if ALPACA_API_KEY is None or ALPACA_SECRET_KEY is None:
    print("Warning: Alpaca keys not found in env — DRY_RUN-only mode.")
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# ------------------ UTIL ------------------
def now_et():
    return datetime.now(ET)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
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
        return {"date": today, "opening_range": None, "entries_today": 0, "last_entry_ts": None, "entries": []}
    return state

def append_log(row):
    fields = ["timestamp","date","symbol","side","mode","qty","entry_price","stop_price","tp_price","order_id","dry_run"]
    newf = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if newf:
            writer.writeheader()
        writer.writerow({k: row.get(k,"") for k in fields})

# ------------------ DATA HELPERS ------------------
def fetch_1m(symbol, days=YF_LOOKBACK_DAYS):
    df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    # localize/convert to ET
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
    except Exception:
        pass
    return df

def vwap(df):
    # df expected to have 'Close' and 'Volume' with datetime index intraday
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    pv = typical * df['Volume']
    return pv.cumsum() / df['Volume'].cumsum()

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
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def get_account_equity():
    try:
        acct = client.get_account()
        return float(acct.equity)
    except Exception as e:
        print("Couldn't fetch account equity:", e)
        return 100000.0

def get_position_qty(symbol):
    try:
        pos = client.get_all_positions()
        for p in pos:
            if p.symbol == symbol:
                return int(float(p.qty))
        return 0
    except Exception:
        return 0

# ------------------ ORDER HELPERS (bracket) ------------------
def submit_bracket(symbol, qty, side, stop_price, tp_price):
    if qty <= 0:
        print("Qty 0 — skipping order.")
        return None
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] SUBMIT {side} {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id": f"dry_{side}", "symbol": symbol, "qty": qty}
    try:
        order = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": f"{tp_price:.2f}"},
            stop_loss={"stop_price": f"{stop_price:.2f}"}
        )
        resp = client.submit_order(order)
        print("Submitted bracket order id:", getattr(resp, "id", "unknown"))
        return resp
    except Exception as e:
        print("Error submitting bracket:", e)
        return None

# ------------------ STRATEGY LOGIC ------------------

# Entry rules (priority order)
# 1) OR breakout/breakdown (fast)
# 2) VWAP pullback (price > VWAP, pulls to VWAP, momentum supports)
# 3) EMA9 pullback in trend (price returns to EMA9 while 1m trend above/below)

def form_opening_range(state):
    if state.get("opening_range"):
        return state
    df = fetch_1m(SYMBOL)
    if df.empty:
        print("No 1m bars")
        return state
    today = date.today()
    todays = df[df.index.date == today]
    if todays.empty:
        print("No intraday bars for today")
        return state
    mask = (todays.index.time >= OPEN_START) & (todays.index.time < OPEN_END)
    or_bars = todays.loc[mask]
    if or_bars.empty:
        print("Opening range not complete yet")
        return state
    high = float(or_bars['High'].max())
    low = float(or_bars['Low'].min())
    state['opening_range'] = {"high": high, "low": low, "formed_at": or_bars.index[0].isoformat()}
    print(f"OR formed: high={high:.4f}, low={low:.4f}")
    save_state(state)
    return state

def compute_qty_stop(price, atr, equity, side="long"):
    # Determine stop distance
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

def should_enter_or_breakout(state, df_latest_row):
    # check OR breakout/breakdown
    or_high = state['opening_range']['high']
    or_low  = state['opening_range']['low']
    price = float(df_latest_row['Close'])
    buf = 0.0006
    if price > or_high * (1 + buf):
        return "long_or"
    if price < or_low * (1 - buf):
        return "short_or"
    return None

def should_enter_vwap(df):
    # vwap on intraday df
    v = vwap(df)
    last = v.iloc[-1]
    price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else price
    # long case: price > vwap, then pullback close to vwap and bounce with RSI not overbought
    if price > last:
        # price above vwap now; check prior pullback: if prev_price < last and now price > prev -> bounce
        if prev_price < last and price > prev_price:
            r = rsi(df['Close']).iloc[-1]
            if r < 70:
                return "long_vwap"
    # short case: price < vwap and bounce down from vwap
    if price < last:
        if prev_price > last and price < prev_price:
            r = rsi(df['Close']).iloc[-1]
            if r > 30:
                return "short_vwap"
    return None

def should_enter_ema_pull(df):
    # 9-EMA pullback in direction of short-term trend
    ema9 = ema(df['Close'], 9)
    ema21 = ema(df['Close'], 21)
    price = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2] if len(df) > 1 else price
    # long condition: 9>21 (uptrend) and price pulls to ema9 and bounces
    if ema9.iloc[-1] > ema21.iloc[-1]:
        # pull to ema9 then bounce: prev < ema9 and price > prev
        if prev < ema9.iloc[-1] and price > prev:
            r = rsi(df['Close']).iloc[-1]
            if r < 70:
                return "long_ema"
    # short condition
    if ema9.iloc[-1] < ema21.iloc[-1]:
        if prev > ema9.iloc[-1] and price < prev:
            r = rsi(df['Close']).iloc[-1]
            if r > 30:
                return "short_ema"
    return None

def attempt_entry(state):
    # must have OR formed
    if not state.get('opening_range'):
        print("No OR -> skip entries")
        return state
    if state.get('entries_today', 0) >= MAX_ENTRIES_PER_DAY:
        print("Max entries today reached")
        return state
    last_ts = state.get('last_entry_ts')
    if last_ts:
        last = datetime.fromisoformat(last_ts)
        if (now_et() - last) < timedelta(minutes=REENTRY_COOLDOWN_MIN):
            print("Cooldown active -> skip entry")
            return state

    df = fetch_1m(SYMBOL)
    if df.empty:
        return state
    today_df = df[df.index.date == date.today()]
    if today_df.empty:
        return state
    latest = today_df.iloc[-1]
    price = float(latest['Close'])

    equity = get_account_equity()
    atr_series = compute_atr(df)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None

    # Priority 1: OR breakout
    or_signal = should_enter_or_breakout(state, latest)
    if or_signal:
        if or_signal == "long_or" and get_position_qty(SYMBOL) <= 0:
            qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
            if qty > 0:
                resp = submit_bracket(SYMBOL, qty, "buy", stop, tp)
                order_id = getattr(resp,"id",(resp or {}).get("id",""))
                log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                       "symbol": SYMBOL, "side": "LONG", "mode": "OR_BREAKOUT",
                       "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                       "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)}
                append_log(log)
                state['last_entry_ts'] = now_et().isoformat()
                state['entries_today'] = state.get('entries_today',0)+1
                state.setdefault('entries',[]).append(log)
                save_state(state)
                return state
        if or_signal == "short_or" and get_position_qty(SYMBOL) >= 0:
            qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
            if qty > 0:
                resp = submit_bracket(SYMBOL, qty, "sell", stop, tp)
                order_id = getattr(resp,"id",(resp or {}).get("id",""))
                log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                       "symbol": SYMBOL, "side": "SHORT", "mode": "OR_BREAKDOWN",
                       "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                       "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)}
                append_log(log)
                state['last_entry_ts'] = now_et().isoformat()
                state['entries_today'] = state.get('entries_today',0)+1
                state.setdefault('entries',[]).append(log)
                save_state(state)
                return state

    # Priority 2: VWAP pullbacks
    vwap_signal = should_enter_vwap(today_df)
    if vwap_signal:
        if vwap_signal == "long_vwap" and get_position_qty(SYMBOL) <= 0:
            qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
            if qty > 0:
                resp = submit_bracket(SYMBOL, qty, "buy", stop, tp)
                order_id = getattr(resp,"id",(resp or {}).get("id",""))
                log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                       "symbol": SYMBOL, "side": "LONG", "mode": "VWAP_PULL",
                       "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                       "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)}
                append_log(log)
                state['last_entry_ts'] = now_et().isoformat()
                state['entries_today'] = state.get('entries_today',0)+1
                state.setdefault('entries',[]).append(log)
                save_state(state)
                return state
        if vwap_signal == "short_vwap" and get_position_qty(SYMBOL) >= 0:
            qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
            if qty > 0:
                resp = submit_bracket(SYMBOL, qty, "sell", stop, tp)
                order_id = getattr(resp,"id",(resp or {}).get("id",""))
                log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                       "symbol": SYMBOL, "side": "SHORT", "mode": "VWAP_PULL",
                       "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                       "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)}
                append_log(log)
                state['last_entry_ts'] = now_et().isoformat()
                state['entries_today'] = state.get('entries_today',0)+1
                state.setdefault('entries',[]).append(log)
                save_state(state)
                return state

    # Priority 3: EMA9 pullback entries
    ema_signal = should_enter_ema_pull(today_df)
    if ema_signal:
        if ema_signal == "long_ema" and get_position_qty(SYMBOL) <= 0:
            qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
            if qty > 0:
                resp = submit_bracket(SYMBOL, qty, "buy", stop, tp)
                order_id = getattr(resp,"id",(resp or {}).get("id",""))
                log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                       "symbol": SYMBOL, "side": "LONG", "mode": "EMA_PULL",
                       "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                       "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)}
                append_log(log)
                state['last_entry_ts'] = now_et().isoformat()
                state['entries_today'] = state.get('entries_today',0)+1
                state.setdefault('entries',[]).append(log)
                save_state(state)
                return state
        if ema_signal == "short_ema" and get_position_qty(SYMBOL) >= 0:
            qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
            if qty > 0:
                resp = submit_bracket(SYMBOL, qty, "sell", stop, tp)
                order_id = getattr(resp,"id",(resp or {}).get("id",""))
                log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                       "symbol": SYMBOL, "side": "SHORT", "mode": "EMA_PULL",
                       "qty": qty, "entry_price": price, "stop_price": stop, "tp_price": tp,
                       "order_id": order_id, "dry_run": DRY_RUN or (ALPACA_API_KEY is None)}
                append_log(log)
                state['last_entry_ts'] = now_et().isoformat()
                state['entries_today'] = state.get('entries_today',0)+1
                state.setdefault('entries',[]).append(log)
                save_state(state)
                return state

    print("No entry conditions met this run.")
    return state

def close_all_before_close():
    now = now_et()
    if now.time() < CLOSE_TIME:
        return
    pos_qty = get_position_qty(SYMBOL)
    if pos_qty == 0:
        print("No positions to close.")
        return
    # close with market order via OrderRequest
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] Closing {pos_qty} of {SYMBOL}")
        append_log({"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                    "symbol": SYMBOL, "side": "CLOSE", "mode": "EOD", "qty": pos_qty, "entry_price": "", "stop_price": "", "tp_price": "", "order_id": "dry_close", "dry_run": True})
        return
    try:
        if pos_qty > 0:
            req = OrderRequest(symbol=SYMBOL, qty=pos_qty, side=OrderSide.SELL, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
        else:
            req = OrderRequest(symbol=SYMBOL, qty=abs(pos_qty), side=OrderSide.BUY, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
        resp = client.submit_order(req)
        append_log({"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                    "symbol": SYMBOL, "side": "CLOSE", "mode": "EOD", "qty": pos_qty, "order_id": getattr(resp,"id",""), "dry_run": False})
        print("EOD close submitted:", getattr(resp,"id",""))
    except Exception as e:
        print("Error closing positions:", e)

# ------------------ MAIN RUN ------------------
def run_once():
    state = load_state()
    state = reset_if_new_day(state)

    # form OR first
    state = form_opening_range(state)

    # only trade during market hours (until close_time)
    if now_et().time() < CLOSE_TIME:
        state = attempt_entry(state)
    else:
        close_all_before_close()

    save_state(state)
    print("Run done.")

if __name__ == "__main__":
    run_once()
