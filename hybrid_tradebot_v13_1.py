#!/usr/bin/env python3

import os
import math
import time
from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

LOOKBACK_MINUTES = 20        # momentum window
MIN_MOVE_PCT = 0.0015        # 0.15% net move (easy to hit)
RISK_PER_TRADE = 0.01        # 1% equity risk
TAKE_PROFIT_PCT = 0.003      # 0.30%
STOP_LOSS_PCT = 0.002        # 0.20%

MIN_VOLUME = 2000            # very permissive
MAX_TRADES_PER_DAY = 5

TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL)

# ================= UTILS =================
def now_et():
    return datetime.now(TZ)

def log(msg):
    print(f"{now_et()} {msg}")

def market_open():
    try:
        return api.get_clock().is_open
    except:
        return False

def has_position():
    try:
        return len(api.list_positions()) > 0
    except:
        return False

def equity():
    try:
        return float(api.get_account().equity)
    except:
        return 0

# ================= DATA =================
def fetch(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False
    )

    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.tail(LOOKBACK_MINUTES + 5)

    if not {"open","high","low","close","volume"}.issubset(df.columns):
        return None

    return df.dropna()

def vwap(df):
    pv = (df["close"] * df["volume"]).sum()
    v = df["volume"].sum()
    return pv / v if v > 0 else df["close"].iloc[-1]

# ================= SIGNAL =================
def score_symbol(symbol):
    df = fetch(symbol)
    if df is None or len(df) < LOOKBACK_MINUTES:
        return None

    recent = df.tail(LOOKBACK_MINUTES)

    price_now = recent["close"].iloc[-1]
    price_then = recent["close"].iloc[0]
    move_pct = (price_now - price_then) / price_then

    if move_pct < MIN_MOVE_PCT:
        return None

    if recent["volume"].iloc[-1] < MIN_VOLUME:
        return None

    vw = vwap(recent)
    if price_now < vw:
        return None

    score = move_pct * 100  # simple, aggressive

    return {
        "symbol": symbol,
        "price": float(price_now),
        "score": float(score)
    }

# ================= ORDER =================
def position_size(price):
    eq = equity()
    risk_dollars = eq * RISK_PER_TRADE
    per_share_risk = price * STOP_LOSS_PCT
    qty = int(risk_dollars / per_share_risk)
    return max(1, qty)

def submit_trade(symbol, price):
    qty = position_size(price)

    tp = round(price * (1 + TAKE_PROFIT_PCT), 2)
    sl = round(price * (1 - STOP_LOSS_PCT), 2)

    try:
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
        log(f"TRADE {symbol} qty={qty} tp={tp} sl={sl}")
        return True
    except Exception as e:
        log(f"ORDER ERROR {symbol}: {e}")
        return False

# ================= MAIN =================
def main():
    log("Run start")

    if not market_open():
        log("Market closed")
        return

    if has_position():
        log("Position open – skip")
        return

    candidates = []

    for sym in SYMBOLS:
        try:
            s = score_symbol(sym)
            if s:
                candidates.append(s)
        except Exception as e:
            log(f"{sym} error {e}")

    if not candidates:
        log("No entries")
        return

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    submit_trade(best["symbol"], best["price"])

if __name__ == "__main__":
    main()
