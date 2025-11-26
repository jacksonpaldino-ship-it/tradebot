#!/usr/bin/env python3
"""
tradebot_spy_improved_alpacapy.py
Improved SPY day-trader using alpaca-py (new SDK). DRY_RUN=True by default.
Fixed Series->float issues and robust to yfinance quirks.
"""

import os, json, math, csv
from datetime import datetime, date, time as dtime, timedelta
import pytz, time

import numpy as np
import pandas as pd
import yfinance as yf

# alpaca-py imports
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, OrderClass
from alpaca.trading.requests import OrderRequest

# ---------------- CONFIG ----------------
DRY_RUN = False
PAPER = True
SYMBOL = "SPY"

OPEN_START = dtime(9,30)
OPEN_END   = dtime(9,45)
ET = pytz.timezone("US/Eastern")
CLOSE_TIME = dtime(15,55)

EQUITY_RISK_PCT = 0.004
MAX_ALLOC_PCT = 0.25
MAX_TOTAL_EXPOSURE_PCT = 0.9

ATR_PERIOD = 14
MIN_STOP_PCT = 0.0015
MAX_STOP_PCT = 0.06
TP_MULT = 2.0

REENTRY_COOLDOWN_MIN = 25
MAX_ENTRIES_PER_DAY = 6

STATE_FILE = "improved_state_alpacapy.json"
LOG_FILE = "improved_trades_alpacapy.csv"
YF_LOOKBACK_DAYS = 7

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ---------------- Client ----------------
if ALPACA_API_KEY is None or ALPACA_SECRET_KEY is None:
    print("ALPACA keys missing — running DRY_RUN-only.")
client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)

# ---------------- Utilities ----------------
def now_et():
    return datetime.now(ET)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,"r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"date": date.today().isoformat(), "opening_range": None, "entries_today":0, "last_entry_ts": None, "entries":[]}

def save_state(s):
    with open(STATE_FILE,"w") as f:
        json.dump(s,f,indent=2)

def reset_if_new_day(state):
    today = date.today().isoformat()
    if state.get("date") != today:
        return {"date": today, "opening_range": None, "entries_today":0, "last_entry_ts": None, "entries":[]}
    return state

def append_log(row):
    fields = ["timestamp","date","symbol","side","mode","qty","entry_price","stop_price","tp_price","order_id","dry_run"]
    newf = not os.path.exists(LOG_FILE)
    with open(LOG_FILE,"a",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if newf:
            writer.writeheader()
        writer.writerow({k: row.get(k,"") for k in fields})

# ---------------- Data helpers ----------------
def fetch_1m(symbol, days=YF_LOOKBACK_DAYS):
    df = yf.download(symbol, period=f"{days}d", interval="1m", progress=False)
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
    except Exception:
        pass
    return df

def vwap(df):
    # returns Series aligned with df index (float per row)
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    pv = typical * df['Volume']
    return pv.cumsum() / df['Volume'].cumsum()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
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

def safe_last_float(series, offset=0):
    """Return float of series.iloc[offset] where offset= -1 for last, -2 prev, etc."""
    try:
        return float(series.iloc[offset])
    except Exception:
        # fallback: convert to scalar if possible
        val = series.tail(1)
        if not val.empty:
            return float(val.values[0])
        raise

def get_account_equity():
    try:
        acct = client.get_account()
        return float(acct.equity)
    except Exception:
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

# ---------------- Order helpers ----------------
def submit_bracket(symbol, qty, side, stop_price, tp_price):
    if qty <= 0:
        return None
    if DRY_RUN or ALPACA_API_KEY is None:
        print(f"[DRY_RUN] {side.upper()} {symbol} qty={qty} stop={stop_price} tp={tp_price}")
        return {"id":f"dry_{side}"}
    try:
        req = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit={"limit_price": f"{tp_price:.2f}"},
            stop_loss={"stop_price": f"{stop_price:.2f}"}
        )
        resp = client.submit_order(req)
        return resp
    except Exception as e:
        print("Submit error:", e)
        return None

# ---------------- Strategy logic ----------------
def form_opening_range(state):
    if state.get("opening_range"):
        return state
    df = fetch_1m(SYMBOL)
    if df.empty:
        return state
    today = date.today()
    todays = df[df.index.date == today]
    if todays.empty:
        return state
    mask = (todays.index.time >= OPEN_START) & (todays.index.time < OPEN_END)
    or_bars = todays.loc[mask]
    if or_bars.empty:
        return state
    high = float(or_bars['High'].max())
    low  = float(or_bars['Low'].min())
    state["opening_range"] = {"high": high, "low": low, "formed_at": or_bars.index[0].isoformat()}
    print(f"OR formed: high={high}, low={low}")
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
        return 0,0,0
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
    or_h = state["opening_range"]["high"]
    or_l = state["opening_range"]["low"]
    buf = 0.0006
    if price > or_h * (1 + buf):
        return "long_or"
    if price < or_l * (1 - buf):
        return "short_or"
    return None

def should_enter_vwap(df):
    v = vwap(df)
    # convert to scalars
    price = safe_last_float(df['Close'], -1)
    prev_price = safe_last_float(df['Close'], -2) if len(df) > 1 else price
    last_v = safe_last_float(v, -1)
    prev_v = safe_last_float(v, -2) if len(v) > 1 else last_v

    # long: price currently above vwap, previous below vwap (pull then bounce)
    if price > last_v and prev_price < prev_v and price > prev_price:
        r = safe_last_float(rsi(df['Close']), -1)
        if r < 70:
            return "long_vwap"
    # short: price currently below vwap, previous above vwap
    if price < last_v and prev_price > prev_v and price < prev_price:
        r = safe_last_float(rsi(df['Close']), -1)
        if r > 30:
            return "short_vwap"
    return None

def should_enter_ema(df):
    ema9 = ema(df['Close'], 9)
    ema21 = ema(df['Close'],21)
    price = safe_last_float(df['Close'], -1)
    prev_price = safe_last_float(df['Close'], -2) if len(df) > 1 else price
    # long trend
    if safe_last_float(ema9,-1) > safe_last_float(ema21,-1):
        if prev_price < safe_last_float(ema9,-1) and price > prev_price:
            r = safe_last_float(rsi(df['Close']), -1)
            if r < 70:
                return "long_ema"
    # short trend
    if safe_last_float(ema9,-1) < safe_last_float(ema21,-1):
        if prev_price > safe_last_float(ema9,-1) and price < prev_price:
            r = safe_last_float(rsi(df['Close']), -1)
            if r > 30:
                return "short_ema"
    return None

def attempt_entry(state):
    if not state.get("opening_range"):
        return state
    if state.get("entries_today",0) >= MAX_ENTRIES_PER_DAY:
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

    price = safe_last_float(today_df['Close'], -1)
    equity = get_account_equity()
    atr_series = compute_atr(df)
    atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None

    # OR primary
    or_sig = should_enter_or(state, price)
    if or_sig == "long_or" and get_position_qty(SYMBOL) <= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
        if qty>0:
            resp = submit_bracket(SYMBOL, qty, "buy", stop, tp)
            order_id = getattr(resp,"id",(resp or {}).get("id",""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side":"LONG","mode":"OR_BREAKOUT","qty":qty,"entry_price":price,"stop_price":stop,"tp_price":tp,"order_id":order_id,"dry_run":DRY_RUN or (ALPACA_API_KEY is None)}
            append_log(log)
            state['last_entry_ts'] = now_et().isoformat()
            state['entries_today'] = state.get('entries_today',0)+1
            state.setdefault('entries',[]).append(log)
            save_state(state)
            return state
    if or_sig == "short_or" and get_position_qty(SYMBOL) >= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
        if qty>0:
            resp = submit_bracket(SYMBOL, qty, "sell", stop, tp)
            order_id = getattr(resp,"id",(resp or {}).get("id",""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side":"SHORT","mode":"OR_BREAKDOWN","qty":qty,"entry_price":price,"stop_price":stop,"tp_price":tp,"order_id":order_id,"dry_run":DRY_RUN or (ALPACA_API_KEY is None)}
            append_log(log)
            state['last_entry_ts'] = now_et().isoformat()
            state['entries_today'] = state.get('entries_today',0)+1
            state.setdefault('entries',[]).append(log)
            save_state(state)
            return state

    # VWAP
    vwap_sig = should_enter_vwap(today_df)
    if vwap_sig == "long_vwap" and get_position_qty(SYMBOL) <= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
        if qty>0:
            resp = submit_bracket(SYMBOL, qty, "buy", stop, tp)
            order_id = getattr(resp,"id",(resp or {}).get("id",""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side":"LONG","mode":"VWAP_PULL","qty":qty,"entry_price":price,"stop_price":stop,"tp_price":tp,"order_id":order_id,"dry_run":DRY_RUN or (ALPACA_API_KEY is None)}
            append_log(log)
            state['last_entry_ts'] = now_et().isoformat()
            state['entries_today'] = state.get('entries_today',0)+1
            state.setdefault('entries',[]).append(log)
            save_state(state)
            return state
    if vwap_sig == "short_vwap" and get_position_qty(SYMBOL) >= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
        if qty>0:
            resp = submit_bracket(SYMBOL, qty, "sell", stop, tp)
            order_id = getattr(resp,"id",(resp or {}).get("id",""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side":"SHORT","mode":"VWAP_PULL","qty":qty,"entry_price":price,"stop_price":stop,"tp_price":tp,"order_id":order_id,"dry_run":DRY_RUN or (ALPACA_API_KEY is None)}
            append_log(log)
            state['last_entry_ts'] = now_et().isoformat()
            state['entries_today'] = state.get('entries_today',0)+1
            state.setdefault('entries',[]).append(log)
            save_state(state)
            return state

    # EMA pullback
    ema_sig = should_enter_ema(today_df)
    if ema_sig == "long_ema" and get_position_qty(SYMBOL) <= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "long")
        if qty>0:
            resp = submit_bracket(SYMBOL, qty, "buy", stop, tp)
            order_id = getattr(resp,"id",(resp or {}).get("id",""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side":"LONG","mode":"EMA_PULL","qty":qty,"entry_price":price,"stop_price":stop,"tp_price":tp,"order_id":order_id,"dry_run":DRY_RUN or (ALPACA_API_KEY is None)}
            append_log(log)
            state['last_entry_ts'] = now_et().isoformat()
            state['entries_today'] = state.get('entries_today',0)+1
            state.setdefault('entries',[]).append(log)
            save_state(state)
            return state
    if ema_sig == "short_ema" and get_position_qty(SYMBOL) >= 0:
        qty, stop, tp = compute_qty_stop(price, atr, equity, "short")
        if qty>0:
            resp = submit_bracket(SYMBOL, qty, "sell", stop, tp)
            order_id = getattr(resp,"id",(resp or {}).get("id",""))
            log = {"timestamp": now_et().isoformat(), "date": date.today().isoformat(),
                   "symbol": SYMBOL, "side":"SHORT","mode":"EMA_PULL","qty":qty,"entry_price":price,"stop_price":stop,"tp_price":tp,"order_id":order_id,"dry_run":DRY_RUN or (ALPACA_API_KEY is None)}
            append_log(log)
            state['last_entry_ts'] = now_et().isoformat()
            state['entries_today'] = state.get('entries_today',0)+1
            state.setdefault('entries',[]).append(log)
            save_state(state)
            return state

    print("No entry conditions met.")
    return state

def close_all_before_close():
    now = now_et()
    if now.time() < CLOSE_TIME:
        return
    pos_qty = get_position_qty(SYMBOL)
    if pos_qty == 0:
        return
    if DRY_RUN or ALPACA_API_KEY is None:
        append_log({"timestamp": now_et().isoformat(), "date": date.today().isoformat(), "symbol":SYMBOL, "side":"CLOSE","mode":"EOD","qty":pos_qty,"order_id":"dry_close","dry_run":True})
        print(f"[DRY_RUN] EOD close {pos_qty}")
        return
    try:
        req = OrderRequest(symbol=SYMBOL, qty=pos_qty if pos_qty>0 else abs(pos_qty), side=OrderSide.SELL if pos_qty>0 else OrderSide.BUY, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
        resp = client.submit_order(req)
        append_log({"timestamp": now_et().isoformat(), "date": date.today().isoformat(), "symbol":SYMBOL, "side":"CLOSE","mode":"EOD","qty":pos_qty,"order_id":getattr(resp,"id",""),"dry_run":False})
        print("EOD close submitted")
    except Exception as e:
        print("EOD close error:", e)

# Main
def run_once():
    state = load_state()
    state = reset_if_new_day(state)
    state = form_opening_range(state)
    if now_et().time() < CLOSE_TIME:
        state = attempt_entry(state)
    else:
        close_all_before_close()
    save_state(state)
    print("Run complete.")

if __name__ == "__main__":
    run_once()
