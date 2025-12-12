#!/usr/bin/env python3
"""
hybrid_tradebot_v10.py

- Aggressive 2–4 trades/day
- ATR-based sizing, risk per trade 1.5% equity
- No guaranteed trades
- YFinance fallback with safe column handling
- Bracket orders with proper rounding
- Single-run for 10-min schedule
"""

import os, math, json, csv, traceback
from datetime import datetime
import pytz
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
MAX_TRADES_PER_DAY = 4
PER_SYMBOL_DAILY_CAP = 2
TP_PCT = 0.0025  # slightly more aggressive take profit
SL_PCT = 0.0015
RISK_PER_TRADE = 0.015  # 1.5% equity per trade
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2000
VWAP_BAND = 0.007  # slightly looser for aggressive entry
STATE_FILE = "bot_state_v10.json"
TRADES_CSV = "trades_v10.csv"
LOG_FILE = "bot_v10.log"
TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set Alpaca secrets")

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Logging / State ----------------
def now_et():
    return datetime.now(TZ)

def utcnow_iso():
    return datetime.utcnow().isoformat()

def log(msg):
    line = f"{utcnow_iso()} {msg}"
    print(line)
    try:
        with open(LOG_FILE,"a") as f:
            f.write(line+"\n")
    except: pass

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None,"daily_trades":0,"per_symbol":{}}
    try:
        with open(STATE_FILE,"r") as f:
            return json.load(f)
    except:
        return {"date": None,"daily_trades":0,"per_symbol":{}}

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
            if not exists: w.writerow(header)
            w.writerow(row)
    except Exception as e:
        log(f"append_trade_row error: {e}")

# ---------------- Market Data ----------------
def fetch_recent_bars(symbol, minutes=120):
    try:
        period_days = max(1,(minutes//60)+1)
        df = yf.download(symbol, period=f"{period_days}d", interval="1m", progress=False, auto_adjust=False)
        for col in ["Open","High","Low","Close","Volume"]:
            if col not in df.columns: df[col] = float('nan')
        df = df.rename(columns=lambda x: x.lower())
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e:
        log(f"fetch_recent_bars error {symbol}: {e}")
        return None

# ---------------- Indicators ----------------
def compute_atr(df, period=ATR_PERIOD):
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high-low,(high-prev).abs(),(low-prev).abs()], axis=1).max(axis=1)
    return float(max(tr.rolling(period,min_periods=1).mean().iloc[-1],1e-6))

def compute_vwap(df):
    v = df["volume"].sum()
    if v <=0: return float(df["close"].iloc[-1])
    return float((df["close"]*df["volume"]).sum()/v)

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
        return float(api.get_account().equity)
    except:
        return None

def compute_qty(entry_price, atr):
    equity = get_equity()
    if equity is None or equity <= 0: return 1
    risk_amount = equity * RISK_PER_TRADE
    per_share_risk = max(atr, entry_price*0.0005)
    qty = int(max(1, math.floor(risk_amount/(per_share_risk+EPS))))
    max_nominal = int(max(1, math.floor(equity*0.3/entry_price)))
    return min(qty,max_nominal)

# ---------------- Orders ----------------
def submit_bracket(symbol, qty, sl_price, tp_price):
    tp_price = round(tp_price,2)
    sl_price = round(sl_price,2)
    try:
        order = api.submit_order(
            symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day',
            order_class='bracket',
            take_profit={'limit_price':tp_price},
            stop_loss={'stop_price':sl_price}
        )
        log(f"Bracket submitted: {symbol} qty={qty} tp={tp_price} sl={sl_price}")
        return order
    except Exception as e:
        log(f"submit_bracket error: {e}")
        return None

# ---------------- Scoring ----------------
def compute_score(df):
    df = df.dropna(subset=["close","high","low","volume"])
    if df.empty: return None
    price = float(df["close"].iloc[-1])
    volume = int(df["volume"].iloc[-1])
    if volume < MIN_VOLUME: return None
    vwap = compute_vwap(df)
    vw_gap = abs(price-vwap)/(vwap+EPS)
    macd_hist = compute_macd_hist(df)
    atr = compute_atr(df)
    vol_score = min(1.0, volume/(df["volume"].mean()+EPS))
    vw_score = max(0.0, 1.0 - vw_gap/VWAP_BAND)
    macd_score = 1.0 if macd_hist>0 else 0.0
    return {"price":price,"vwap":vwap,"atr":atr,"macd_hist":macd_hist,"score":0.4*vw_score+0.35*vol_score+0.25*macd_score}

def pick_trade_candidate():
    candidates=[]
    for s in SYMBOLS:
        df = fetch_recent_bars(s)
        if df is None or df.empty: continue
        info = compute_score(df)
        if info is None or info["score"]<0.2: continue  # lower threshold for aggression
        info["symbol"]=s
        candidates.append(info)
    if not candidates: return None
    candidates.sort(key=lambda x:x["score"],reverse=True)
    return candidates[0]

# ---------------- Main ----------------
def main():
    log(f"Run start ET {now_et().isoformat()}")
    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date")!=today:
        state={"date":today,"daily_trades":0,"per_symbol":{}}
        save_state(state)
    if state["daily_trades"]>=MAX_TRADES_PER_DAY:
        log(f"Daily cap reached {state['daily_trades']}/{MAX_TRADES_PER_DAY}")
        return

    candidate = pick_trade_candidate()
    if not candidate:
        log("No candidate meets score threshold")
        return

    sym = candidate["symbol"]
    per_sym = state["per_symbol"].get(sym,0)
    if per_sym>=PER_SYMBOL_DAILY_CAP:
        log(f"Per-symbol cap reached {sym}")
        return

    entry_price = candidate["price"]
    atr = candidate["atr"]
    qty = compute_qty(entry_price,atr)
    if qty<1: return

    tp = entry_price*(1+TP_PCT)
    sl = entry_price*(1-SL_PCT)
    order = submit_bracket(sym,qty,sl,tp)
    if order:
        state["daily_trades"]+=1
        state["per_symbol"][sym]=per_sym+1
        save_state(state)
        append_trade_row([utcnow_iso(),sym,"BUY_SUBMIT",qty,round(entry_price,2),None,None,f"score:{candidate['score']:.3f}"])
        log(f"Placed bracket for {sym} qty={qty} score={candidate['score']:.3f}")

if __name__=="__main__":
    try: main()
    except Exception as e:
        log(f"Unhandled exception: {repr(e)}")
        log(traceback.format_exc())
