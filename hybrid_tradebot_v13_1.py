#!/usr/bin/env python3
"""
hybrid_tradebot_v13_1.py
- Aggressive, ATR-adaptive stop/take profit
- 2-4 trades/day
- Position sizing ~1–2% equity risk
- No guaranteed trades
"""

import os, math, json, csv, traceback
from datetime import datetime
import pytz, numpy as np, pandas as pd, yfinance as yf
from alpaca_trade_api.rest import REST

# ---------------- CONFIG ----------------
SYMBOLS = ["SPY","QQQ","IWM","DIA"]
MAX_TRADES_PER_DAY = 4
PER_SYMBOL_DAILY_CAP = 2
RISK_PER_TRADE = 0.015
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MIN_VOLUME = 2500
VWAP_BAND = 0.005
STATE_FILE = "bot_state_v13_1.json"
TRADES_CSV = "trades_v13_1.csv"
LOG_FILE = "bot_v13_1.log"
SCORE_THRESHOLD = 0.2

TZ = pytz.timezone("US/Eastern")
EPS = 1e-9

# ---------------- Alpaca ----------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY and ALPACA_BASE_URL):
    raise RuntimeError("Set ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL in repository secrets")
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# ---------------- Logging/State ----------------
def now_et(): return datetime.now(TZ)
def utcnow_iso(): return datetime.utcnow().isoformat()
def log(s):
    line = f"{utcnow_iso()} {s}"
    print(line)
    try: open(LOG_FILE,"a").write(line+"\n")
    except: pass
def load_state():
    if not os.path.exists(STATE_FILE): return {"date":None,"daily_trades":0,"per_symbol":{},"open_order_id":None}
    try: return json.load(open(STATE_FILE,"r"))
    except: return {"date":None,"daily_trades":0,"per_symbol":{},"open_order_id":None}
def save_state(s):
    try: json.dump(s,open(STATE_FILE,"w"))
    except Exception as e: log(f"save_state error: {e}")
def append_trade_row(row):
    header = ["utc_ts","symbol","side","qty","entry","exit","pnl","note"]
    exists = os.path.exists(TRADES_CSV)
    try:
        with open(TRADES_CSV,"a",newline="") as f:
            w = csv.writer(f)
            if not exists: w.writerow(header)
            w.writerow(row)
    except Exception as e: log(f"append_trade_row error: {e}")

# ---------------- Market Data ----------------
def fetch_bars_yf(symbol,minutes=200):
    try:
        period_days = max(1,(minutes//60)+1)
        df = yf.download(symbol,period=f"{period_days}d",interval="1m",progress=False)
        if df is None or df.empty: return None
        df = df.rename(columns=str.lower)
        if not {"open","high","low","close","volume"}.issubset(df.columns): return None
        df = df[["open","high","low","close","volume"]]
        df.index = df.index.tz_localize(None)
        return df.tail(minutes)
    except Exception as e: log(f"fetch_bars_yf error {symbol}: {e}"); return None
def fetch_recent_bars(symbol,minutes=200): return fetch_bars_yf(symbol,minutes)

# ---------------- Indicators ----------------
def compute_atr(df,period=ATR_PERIOD):
    high,low,close = df["high"],df["low"],df["close"]
    prev = close.shift(1)
    tr = pd.concat([high-low,(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    return float(max(tr.rolling(period,min_periods=1).mean().iloc[-1],1e-6))
def compute_vwap(df):
    v = df["volume"].sum()
    if v<=0: return float(df["close"].iloc[-1])
    return float((df["close"]*df["volume"]).sum()/v)
def compute_macd_hist(df):
    close = df["close"]
    ema_fast = close.ewm(span=MACD_FAST,adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW,adjust=False).mean()
    macd = ema_fast-ema_slow
    sig = macd.ewm(span=MACD_SIGNAL,adjust=False).mean()
    return float((macd-sig).iloc[-1])

# ---------------- Sizing ----------------
def get_equity():
    try: return float(api.get_account().equity)
    except Exception as e: log(f"get_equity error: {e}"); return None
def compute_qty(entry_price,atr):
    equity = get_equity()
    if equity is None or equity<=0: return 1
    risk_amount = equity*RISK_PER_TRADE
    per_share_risk = max(atr,entry_price*0.0005)
    qty = int(max(1,math.floor(risk_amount/(per_share_risk+EPS))))
    max_nom = int(max(1,math.floor((equity*0.3)/entry_price)))
    return min(qty,max_nom)

# ---------------- Orders ----------------
def submit_bracket(symbol,qty,sl_price,tp_price):
    try:
        tp_price = round(tp_price,2)
        sl_price = round(sl_price,2)
        order = api.submit_order(
            symbol=symbol,qty=qty,side='buy',type='market',time_in_force='day',
            order_class='bracket',
            take_profit={'limit_price': str(tp_price)},
            stop_loss={'stop_price': str(sl_price)}
        )
        log(f"Bracket submitted: {symbol} qty={qty} TP={tp_price} SL={sl_price}")
        return order
    except Exception as e: log(f"submit_bracket error: {e}"); return None

def has_open_positions():
    try: return len(api.list_positions())>0
    except Exception as e: log(f"list_positions error: {e}"); return False

# ---------------- Scoring ----------------
def compute_score(df):
    if df is None or df.empty: return None
    for col in ["close","high","low","volume"]:
        if col not in df.columns: return None
    df = df.dropna(subset=["close","high","low","volume"])
    if df.empty: return None
    price = float(df["close"].iloc[-1])
    volume = int(df["volume"].iloc[-1])
    if volume<MIN_VOLUME: return None
    vwap = compute_vwap(df)
    vw_gap = abs(price-vwap)/(vwap+EPS)
    macd_hist = compute_macd_hist(df)
    atr = compute_atr(df)
    vol_score = min(1.0,volume/(df["volume"].mean()+EPS))
    vw_score = max(0.0,1.0-vw_gap/(VWAP_BAND*0.8))
    macd_score = 1.0 if macd_hist>0 else 0.2
    score = 0.4*vw_score + 0.35*vol_score + 0.25*macd_score
    return {"price":price,"vwap":vwap,"atr":atr,"macd_hist":macd_hist,"score":score}

def pick_trade_candidate():
    best = None; best_score=0
    for s in SYMBOLS:
        try:
            df = fetch_recent_bars(s,minutes=120)
            info = compute_score(df)
            if info and info["score"]>best_score:
                best=(s,info); best_score=info["score"]
        except Exception as e: log(f"pick_trade_candidate error {s}: {e}")
    if best and best_score>=SCORE_THRESHOLD: return best
    return None

# ---------------- Main ----------------
def run_once():
    log(f"Run start ET {now_et().isoformat()}")
    try:
        clock = api.get_clock()
        if not getattr(clock,"is_open",False): log("Market closed"); return
    except Exception as e: log(f"get_clock error: {e}"); return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")
    if state.get("date")!=today: state={"date":today,"daily_trades":0,"per_symbol":{},"open_order_id":None}; save_state(state)
    if state["daily_trades"]>=MAX_TRADES_PER_DAY: log("Daily cap reached"); return
    if has_open_positions(): log("Open positions detected; skipping entry"); return

    candidate = pick_trade_candidate()
    if candidate:
        sym,info = candidate
        per_sym = state["per_symbol"].get(sym,0)
        if per_sym>=PER_SYMBOL_DAILY_CAP: log(f"Per-symbol cap reached for {sym}"); return
        entry_price = info["price"]
        atr = info["atr"]
        qty = compute_qty(entry_price,atr)
        if qty<1: return
        # ATR-based stop and take-profit
        sl = entry_price - atr
        tp = entry_price + 1.5*atr
        order = submit_bracket(sym,qty,sl,tp)
        if order:
            state["daily_trades"]+=1
            state["per_symbol"][sym]=per_sym+1
            state["open_order_id"]=getattr(order,'id',None)
            save_state(state)
            append_trade_row([utcnow_iso(),sym,"BUY_SUBMIT",qty,round(entry_price,2),None,None,f"score:{info['score']:.3f}"])
            log(f"Placed bracket for {sym} qty={qty} score={info['score']:.3f}")
            return
    log("No candidate meets score threshold or caps reached")
    log("Run complete")

if __name__=="__main__":
    try: run_once()
    except Exception as e:
        log("Unhandled exception: "+repr(e))
        import traceback; log(traceback.format_exc())
