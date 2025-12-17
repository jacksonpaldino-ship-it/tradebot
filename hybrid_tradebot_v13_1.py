#!/usr/bin/env python3

import os
import math
from datetime import datetime
import pytz

import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

LOOKBACK_MIN = 15            # very short momentum
MIN_MOVE_PCT = 0.0008        # 0.08% move (EXTREMELY EASY)
RISK_PER_TRADE = 0.015       # 1.5%
TP_PCT = 0.0025              # 0.25%
SL_PCT = 0.0018              # 0.18%

MIN_VOLUME = 1000
TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL)

# ================= UTILS =================
def log(msg):
    print(f"{datetime.now(TZ)} {msg}")

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
        return 0.0

# ================= DATA =================
def fetch(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=True
    )

    if df is None or df.empty:
        return None

    # 🔧 FIX: flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    df = df.dropna()
    return df.tail(LOOKBACK_MIN + 5)

# ================= SIGNAL =================
def get_signal(symbol):
    df = fetch(symbol)
    if df is None or len(df) < LOOKBACK_MIN:
        return None

    recent = df.tail(LOOKBACK_MIN)

    start = recent["close"].iloc[0]
    end = recent["close"].iloc[-1]
    move = (end - start) / start

    if move < MIN_MOVE_PCT:
        return None

    if recent["volume"].iloc[-1] < MIN_VOLUME:
        return None

    return {
        "symbol": symbol,
        "price": float(end),
        "score": float(move)
    }

# ================= ORDER =================
def calc_qty(price):
    eq = equity()
    risk = eq * RISK_PER_TRADE
    per_share = price * SL_PCT
    qty = int(risk / per_share)
    return max(1, qty)

def submit_trade(symbol, price):
    qty = calc_qty(price)

    tp = round(price * (1 + TP_PCT), 2)
    sl = round(price * (1 - SL_PCT), 2)

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
        log(f"ENTER {symbol} qty={qty} tp={tp} sl={sl}")
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
        log("Position open")
        return

    signals = []

    for sym in SYMBOLS:
        try:
            sig = get_signal(sym)
            if sig:
                signals.append(sig)
        except Exception as e:
            log(f"{sym} error {e}")

    if not signals:
        log("No entries")
        return

    # Most aggressive momentum
    signals.sort(key=lambda x: x["score"], reverse=True)
    best = signals[0]

    submit_trade(best["symbol"], best["price"])

if __name__ == "__main__":
    main()
