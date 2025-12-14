#!/usr/bin/env python3
"""
hybrid_tradebot_v14.py
INTENTIONALLY AGGRESSIVE
- High trade frequency
- Minimal filters
- Tight risk control
"""

import os, json, csv, traceback
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

MAX_TRADES_PER_DAY = 10
TP_PCT = 0.0012      # 0.12%
SL_PCT = 0.0008      # 0.08%
RISK_PER_TRADE = 0.015
ATR_PERIOD = 14

STATE_FILE = "bot_state_v14.json"
TRADES_CSV = "trades_v14.csv"
LOG_FILE = "bot_v14.log"
TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

# ================= UTIL =================
def now_et():
    return datetime.now(TZ)

def log(msg):
    ts = datetime.utcnow().isoformat()
    print(ts, msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts} {msg}\n")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "trades": 0}
    return json.load(open(STATE_FILE))

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"))

def record_trade(row):
    header = ["utc","symbol","qty","entry","tp","sl"]
    exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)

# ================= DATA =================
def fetch(symbol):
    df = yf.download(symbol, interval="1m", period="1d", progress=False)
    if df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    return df.tail(60)

def atr(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(ATR_PERIOD).mean().iloc[-1])

# ================= STRATEGY =================
def should_trade(df):
    """Extremely loose conditions"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    momentum = abs(last.close - prev.close) / prev.close
    range_expansion = (last.high - last.low) / last.close

    # Trade if price is MOVING
    return momentum > 0.0003 or range_expansion > 0.001

def position_size(price, atr_val):
    equity = float(api.get_account().equity)
    risk = equity * RISK_PER_TRADE
    per_share = max(atr_val, price * 0.0005)
    qty = int(risk / per_share)
    max_qty = int((equity * 0.3) / price)
    return max(1, min(qty, max_qty))

# ================= EXECUTION =================
def submit(symbol, price, atr_val):
    qty = position_size(price, atr_val)
    tp = round(price * (1 + TP_PCT), 2)
    sl = round(price * (1 - SL_PCT), 2)

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side="buy",
        type="market",
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": str(tp)},
        stop_loss={"stop_price": str(sl)}
    )

    record_trade([datetime.utcnow().isoformat(), symbol, qty, price, tp, sl])
    log(f"TRADE {symbol} qty={qty} entry={price:.2f} tp={tp} sl={sl}")

# ================= MAIN =================
def main():
    log(f"Run start ET {now_et()}")

    if not api.get_clock().is_open:
        log("Market closed")
        return

    state = load_state()
    today = now_et().strftime("%Y-%m-%d")

    if state["date"] != today:
        state = {"date": today, "trades": 0}

    if state["trades"] >= MAX_TRADES_PER_DAY:
        log("Daily cap reached")
        return

    for sym in SYMBOLS:
        if state["trades"] >= MAX_TRADES_PER_DAY:
            break

        df = fetch(sym)
        if df is None or len(df) < 20:
            continue

        if should_trade(df):
            price = float(df["close"].iloc[-1])
            atr_val = atr(df)
            submit(sym, price, atr_val)
            state["trades"] += 1
            save_state(state)
            break  # ONE TRADE PER RUN (important)

    log("Run complete")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR " + repr(e))
        log(traceback.format_exc())
