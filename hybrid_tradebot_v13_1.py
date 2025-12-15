#!/usr/bin/env python3

import os
import math
import json
import pytz
import yfinance as yf
import pandas as pd
from datetime import datetime
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

TP_PCT = 0.003      # 0.30%
SL_PCT = 0.002      # 0.20%

RISK_PCT = 0.005    # 0.5% equity per trade
MAX_TRADES_PER_DAY = 6

STATE_FILE = "bot_state.json"
TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

# ================= UTIL =================
def log(msg):
    print(f"{datetime.now(TZ)} {msg}")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"date": None, "trades": 0}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def market_open():
    return api.get_clock().is_open

def has_position():
    return len(api.list_positions()) > 0

def round_price(p):
    return round(p, 2)  # ETF-safe

# ================= DATA =================
def fetch(symbol):
    df = yf.download(
        symbol,
        interval="1m",
        period="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        return None

    # Handle MultiIndex safely
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    return df.tail(30)

def vwap(df):
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    return pv / v if v > 0 else df["close"].iloc[-1]

# ================= STRATEGY =================
def should_enter(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    vw = vwap(df)
    vol_avg = df["volume"].rolling(10).mean().iloc[-1]

    return (
        last["close"] > vw and
        last["close"] > last["open"] and
        last["volume"] > vol_avg * 0.7 and
        last["close"] > prev["close"]
    )

def position_size(price):
    equity = float(api.get_account().equity)
    risk_dollars = equity * RISK_PCT
    stop_dist = price * SL_PCT
    qty = int(risk_dollars / stop_dist)
    return max(1, qty)

# ================= ORDER =================
def submit_trade(symbol, price):
    qty = position_size(price)

    tp = round_price(price * (1 + TP_PCT))
    sl = round_price(price * (1 - SL_PCT))

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side="buy",
        type="market",
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": tp},
        stop_loss={"stop_price": sl}
    )

    log(f"TRADE {symbol} qty={qty} tp={tp} sl={sl}")

# ================= MAIN =================
def main():
    log("Run start")

    if not market_open():
        log("Market closed")
        return

    state = load_state()
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    if state["date"] != today:
        state = {"date": today, "trades": 0}
        save_state(state)

    if state["trades"] >= MAX_TRADES_PER_DAY:
        log("Daily cap reached")
        return

    if has_position():
        log("Position open, skipping")
        return

    for sym in SYMBOLS:
        df = fetch(sym)
        if df is None or len(df) < 5:
            continue

        if should_enter(df):
            price = df["close"].iloc[-1]
            submit_trade(sym, price)

            state["trades"] += 1
            save_state(state)
            return  # ONE TRADE PER RUN

    log("No entries")

if __name__ == "__main__":
    main()
