#!/usr/bin/env python3

import os
import math
import logging
from datetime import datetime, time
import pytz
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST

# ================= CONFIG =================
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

LOOKBACK_MIN = 1
MIN_MOVE_PCT = 0.00015     # VERY aggressive (0.015%)
MIN_VOLUME = 100

TAKE_PROFIT_PCT = 0.0015
STOP_LOSS_PCT = 0.0012

RISK_PER_TRADE = 0.0025
MAX_POSITION_PCT = 0.15

TZ = pytz.timezone("US/Eastern")
TRADE_START = time(9, 30)
TRADE_END   = time(15, 55)

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

# ================= ALPACA =================
api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),
)

# ================= UTILS =================
def now_et():
    return datetime.now(TZ)

def market_open():
    try:
        clock = api.get_clock()
        return clock.is_open
    except:
        return False

def in_trade_window():
    t = now_et().time()
    return TRADE_START <= t <= TRADE_END

def equity():
    return float(api.get_account().equity)

def positions():
    return {p.symbol: p for p in api.list_positions()}

# ================= DATA =================
def fetch(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    return df.dropna().tail(LOOKBACK_MIN + 1)

# ================= SIGNAL =================
def get_signal(symbol):
    df = fetch(symbol)
    if df is None or len(df) < 2:
        return None

    start = float(df["close"].iloc[0])
    end = float(df["close"].iloc[-1])
    vol = int(df["volume"].iloc[-1])

    move = (end - start) / start
    log.info(f"{symbol}: move {move*100:.4f}%")

    if abs(move) < MIN_MOVE_PCT:
        log.info(f"{symbol}: move too small — skip")
        return None

    if vol < MIN_VOLUME:
        log.info(f"{symbol}: volume too low — skip")
        return None

    side = "buy" if move > 0 else "sell"
    return {"symbol": symbol, "side": side, "price": end}

# ================= ORDER =================
def calc_qty(price):
    eq = equity()
    risk_dollars = eq * RISK_PER_TRADE
    per_share_risk = price * STOP_LOSS_PCT

    qty_risk = math.floor(risk_dollars / per_share_risk)
    qty_cap = math.floor((eq * MAX_POSITION_PCT) / price)

    return max(1, min(qty_risk, qty_cap))

def submit_long(symbol, qty, price):
    tp = round(price * (1 + TAKE_PROFIT_PCT), 2)
    sl = round(price * (1 - STOP_LOSS_PCT), 2)

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side="buy",
        type="market",
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": tp},
        stop_loss={"stop_price": sl},
    )

def submit_short(symbol, qty, price):
    entry = round(price + 0.01, 2)
    tp = round(price * (1 - TAKE_PROFIT_PCT), 2)
    sl = round(price * (1 + STOP_LOSS_PCT), 2)

    if tp >= entry - 0.01:
        tp = round(entry - 0.01, 2)
    if sl <= entry + 0.01:
        sl = round(entry + 0.01, 2)

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side="sell",
        type="limit",
        limit_price=entry,
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": tp},
        stop_loss={"stop_price": sl},
    )

# ================= MAIN =================
def main():
    log.info("BOT START")

    if not market_open() or not in_trade_window():
        log.info("Market closed or outside trade window")
        return

    log.info(f"Equity: {equity():.2f}")
    pos = positions()

    for sym in SYMBOLS:
        if sym in pos:
            log.info(f"{sym}: already in position — skip")
            continue

        try:
            sig = get_signal(sym)
            if not sig:
                continue

            qty = calc_qty(sig["price"])
            if qty <= 0:
                continue

            if sig["side"] == "buy":
                submit_long(sym, qty, sig["price"])
            else:
                submit_short(sym, qty, sig["price"])

            log.info(f"ORDER SENT {sym} {sig['side'].upper()}")
            break  # ONE trade per run

        except Exception as e:
            log.error(f"{sym}: ERROR — {e}")

    log.info("BOT END")

if __name__ == "__main__":
    main()
