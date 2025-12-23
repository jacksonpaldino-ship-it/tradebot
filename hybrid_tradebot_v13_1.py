#!/usr/bin/env python3

import os
import math
from datetime import datetime, time
import pytz
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["SPY", "QQQ", "IWM"]

OPEN_RANGE_START = time(9, 30)
OPEN_RANGE_END   = time(9, 45)
SESSION_END      = time(15, 30)

RISK_PER_TRADE = 0.005   # 0.5% per trade
R_MULTIPLE = 2.5          # Target multiple of risk
MAX_TRADES_PER_SYMBOL = 1  # one trade per symbol per day in cron

TZ = pytz.timezone("US/Eastern")

# ================= ALPACA =================
api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL")
)

# ================= UTILS =================
def log(msg):
    print(f"{datetime.now(TZ)} {msg}", flush=True)

def market_open():
    try:
        return api.get_clock().is_open
    except:
        return False

def equity():
    try:
        return float(api.get_account().equity)
    except:
        return 0.0

def positions():
    return {p.symbol: p for p in api.list_positions()}

def trades_today(symbol):
    today = datetime.now(TZ).date()
    orders = api.list_orders(status="closed", limit=100)
    return [o for o in orders if o.symbol == symbol and o.filled_at and o.filled_at.date() == today]

# ================= DATA =================
def fetch_intraday(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False
    )
    if df is None or df.empty:
        return None

    df = df.rename(columns=str.lower)
    df.index = df.index.tz_localize("UTC").tz_convert(TZ)

    # VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    return df.dropna()

def opening_range(df):
    or_df = df.between_time(OPEN_RANGE_START, OPEN_RANGE_END)
    if or_df.empty:
        return None, None
    return or_df["high"].max(), or_df["low"].min()

# ================= SIGNAL =================
def get_signal(symbol):
    now = datetime.now(TZ).time()
    if now < OPEN_RANGE_END or now > SESSION_END:
        return None

    df = fetch_intraday(symbol)
    if df is None:
        return None

    orb_high, orb_low = opening_range(df)
    if orb_high is None:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = last["close"]
    vwap = last["vwap"]

    # check if symbol already traded today
    if len(trades_today(symbol)) >= MAX_TRADES_PER_SYMBOL:
        return None

    # check if symbol already has open position
    if symbol in positions():
        return None

    # ORB breakout
    if price > orb_high:
        return {"symbol": symbol, "entry": price, "stop": orb_low}

    # VWAP continuation (price crosses VWAP upward)
    if price > vwap and prev["close"] <= vwap:
        return {"symbol": symbol, "entry": price, "stop": vwap}

    return None

# ================= POSITION SIZING =================
def calc_qty(entry, stop):
    eq = equity()
    if eq <= 0:
        return 0
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    risk_dollars = eq * RISK_PER_TRADE
    qty = int(risk_dollars / risk_per_share)
    return max(1, qty)

def round_price(p):
    return round(p, 2)

# ================= ORDER =================
def submit_trade(signal):
    entry = signal["entry"]
    stop = signal["stop"]
    qty = calc_qty(entry, stop)
    if qty <= 0:
        log(f"Skip {signal['symbol']}: qty=0")
        return False

    target = round_price(entry + (entry - stop) * R_MULTIPLE)

    try:
        api.submit_order(
            symbol=signal["symbol"],
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(target)},
            stop_loss={"stop_price": str(round_price(stop))}
        )
        log(f"ENTER {signal['symbol']} qty={qty} target={target} stop={stop}")
        return True
    except Exception as e:
        log(f"ORDER ERROR {signal['symbol']}: {e}")
        return False

# ================= MAIN =================
def main():
    log("GitHub Actions run start")

    if not market_open():
        log("Market closed")
        return

    for sym in SYMBOLS:
        try:
            sig = get_signal(sym)
            if sig:
                submit_trade(sig)
                # one trade per run to avoid duplicate entries in cron
                break
        except Exception as e:
            log(f"{sym} error: {e}")

if __name__ == "__main__":
    main()
