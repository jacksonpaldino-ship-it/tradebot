#!/usr/bin/env python3
"""
hybrid_tradebot_v9.py
- Uses alpaca-trade-api for orders/account info
- Uses yfinance for minute bars (reliable without Alpaca subscription)
- Single-run script intended to be scheduled every 10 minutes
- ~2–4 trades/day, risk-controlled
- Fully error-proof, handles missing columns and empty data
"""

import os
import time
import math
import json
import csv
import traceback
from datetime import datetime
import pytz
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY","QQQ","IWM","DIA"]
MAX_TRADES_PER_DAY = 4
PER_SYMBOL_DAILY_CAP = 2
TP_PCT = 0.002
SL_PCT = 0.0015
RISK_PER_TRADE = 0.01   # 1% equity per trade
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2000
VWAP_BAND = 0.005
STATE_FILE = "bot_state_v9.json"
TRADES_CSV = "trades_v9.csv"
LOG_FILE = "bot_v9.log"

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
        with open(LOG_FILE,"a") as f:
            f.write(line+"\n")
    except:
        pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "daily_trades":0, "per_symbol":{}}
    try:
        with open(STATE_FILE,"r") as f:
            return json.load(f)
    except:
        return {"date": None, "daily_trades":0, "per_symbol":{}}

def save_state(state):
    try:
        with open(STATE_FILE,"w") as f:
            json.dump(state,f)
    except Exception as e:
        log(f"save_state error: {e}")

def append_trade_row(row):
    header = ["utc_ts","symbol","side","qty","entry","exit","pnl","note"]
    exists = os.path.exists(TRADES_CSV)
    try:
        with open(TRADES_CSV,"a",newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(header)
            w.writerow(row)
    except Exception as e:
        log(f"append_trade_row error: {e}")

# ---------------- market data ----------------
def fetch_recent_bars(symbol, minutes=120):
    try:
        period_days = max(1,(minutes//60)+1)
        df = yf.download(symbol, period=f"{period_days}d", interval="1m", progress=False)
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)
        required_cols = ["open","high","low","close","volume"]
        if not all(col in df.columns for col in required_cols):
            log(f"{symbol} missing columns: {df.columns.tolist()}")
            return None
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"fetch_recent_bars error {symbol}: {e}")
        return None

# ---------------- indicators ----------------
def compute_atr(df, period=ATR_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev = close.shift(1)
    tr = pd.concat([high-low,(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    atr = tr.rolling(period,min_periods=1).mean().iloc[-1]
    return float(max(atr,1e-6))

def compute_vwap(df):
    pv = (df["close"]*df["volume"]).sum()
    v = df["volume"].sum()
    return float(pv/v) if v>0 else float(df["close"].iloc[-1])

def compute_macd_hist(df):
    close = df["close"]
    ema_fast = close.ewm(span=MACD_FAST,adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW,adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=MACD_SIGNAL,adjust=False).mean()
    return float((macd-sig).iloc[-1])

# ---------------- sizing ----------------
def get_equity():
    try:
        return float(api.get_account().equity)
    except:
        return None

def compute_qty(price, atr):
    equity = get_equity()
    if equity is None or equity<=0:
        return 1
    risk_amount = equity*RISK_PER_TRADE
    per_share_risk = max(atr, price*0.0005)
    qty = int(max(1, math.floor(risk_amount/(per_share_risk+EPS))))
    max_nominal = int(max(1, math.floor((equity*0.3)/price)))
    return min(qty,max_nominal)

# ---------------- orders ----------------
def submit_bracket(symbol, qty, sl_price, tp_price):
    # round prices to nearest cent
    sl_price = round(sl_price,2)
    tp_price = round(tp_price,2)
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price":str(tp_price)},
            stop_loss={"stop_price":str(sl_price)}
        )
        log(f"Bracket submitted: {symbol} qty={qty} tp={tp_price} sl={sl_price}")
        return order
    except Exception as e:
        log(f"submit_bracket error: {e}")
        return None

def has_open_positions():
    try:
        return len(api.list_positions())>0
    except:
        return False

# ---------------- scoring ----------------
def compute_score(df):
    if df is None or df.empty:
        return None
    required_cols = ["close","high","low","volume"]
    if not all(col in df.columns for col in required_cols):
        return None
    df = df.dropna(subset=required_cols)
    if df.empty or len(df)<10:
        return None
    volume = float(df["volume"].iloc[-1])
    if volume<MIN_VOLUME:
        return None
    window = df.tail(60)
    price = float(window["close"].iloc[-1])
    vwap = compute_vwap(window)
    vw_gap = abs(price-vwap)/(vwap+EPS)
    macd_hist = compute_macd_hist(window)
    atr = compute_atr(window)
    vol_score = min(1.0,float(window["volume"].iloc[-1])/(float(window["volume"].mean())+EPS))
    vw_score = max(0.0,1.0-vw_gap/VWAP_BAND)
    macd_score = 1.0 if macd_hist>0 else 0.0
    score = 0.40*vw_score + 0.40*vol_score + 0.20*macd_score
    return {"score":score,"price":price,"atr":atr}

def pick_trade_candidate():
    candidates = []
    for sym in SYMBOLS:
        try:
            df = fetch_recent_bars(sym, minutes=120)
            info = compute_score(df)
            if info:
                info["symbol"]=sym
                candidates.append(info)
        except Exception as e:
            log(f"pick_trade_candidate error {sym}: {e}")
    if not candidates:
        return None
    candidates.sort(key=lambda x:x["score"], reverse=True)
    for cand in candidates:
        if cand["score"]>=0.25:
            return cand
    return None

# ---------------- main ----------------
def main():
    log(f"Run start ET {now_et().isoformat()}")
    try:
        state = load_state()
        today = now_et().strftime("%Y-%m-%d")
        if state.get("date") != today:
            state = {"date":today,"daily_trades":0,"per_symbol":{}}
            save_state(state)
        if state["daily_trades"]>=MAX_TRADES_PER_DAY:
            log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
            return
        if has_open_positions():
            log("Open positions present; skipping trade.")
            return

        candidate = pick_trade_candidate()
        if candidate:
            sym = candidate["symbol"]
            per_sym = state["per_symbol"].get(sym,0)
            if per_sym>=PER_SYMBOL_DAILY_CAP:
                log(f"Per-symbol cap reached for {sym}")
                return
            qty = compute_qty(candidate["price"], candidate["atr"])
            if qty<1:
                return
            tp = candidate["price"]*(1+TP_PCT)
            sl = candidate["price"]*(1-SL_PCT)
            order = submit_bracket(sym, qty, sl, tp)
            if order:
                state["daily_trades"]+=1
                state["per_symbol"][sym]=per_sym+1
                save_state(state)
                append_trade_row([utcnow_iso(), sym, "BUY", qty, round(candidate["price"],2), None, None, f"score:{candidate['score']:.3f}"])
                log(f"Trade executed for {sym} qty={qty}")
            else:
                log("Order submission failed")
        else:
            log("No candidate meets score threshold")
    except Exception as e:
        log("Unhandled exception: "+repr(e))
        log(traceback.format_exc())
    log("Run complete.")

if __name__=="__main__":
    main()
