#!/usr/bin/env python3

import os
import math
import time
import logging
from datetime import datetime, timedelta
import pytz

import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

LOOKBACK_MIN = 5
MIN_MOVE_PCT = 0.001        # 0.10% (aggressive but sane)
MIN_VOLUME = 500_000

RISK_PER_TRADE = 0.003      # 0.3% equity
MAX_POSITION_PCT = 0.20

TAKE_PROFIT_PCT = 0.0025    # 0.25%
STOP_LOSS_PCT = 0.0018      # 0.18%

MAX_DAILY_LOSS_PCT = 0.02   # 2% daily loss kill switch
SYMBOL_COOLDOWN_MIN = 10    # minutes

FORCE_FLAT_TIME = (15, 55)  # 3:55 PM ET

TZ = pytz.timezone("US/Eastern")

# ================= LOGGING =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger(__name__)

# ================= ALPACA =================

API_KEY = os.environ["APCA_API_KEY_ID"]
API_SECRET = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

api = REST(API_KEY, API_SECRET, BASE_URL)

# ================= STATE =================

last_trade_time = {}
start_of_day_equity = None

# ================= UTILS =================

def now_et():
    return datetime.now(TZ)

def market_open():
    return api.get_clock().is_open

def equity():
    return float(api.get_account().equity)

def positions_by_symbol():
    return {p.symbol: p for p in api.list_positions()}

def daily_loss_exceeded():
    global start_of_day_equity
    eq = equity()
    if start_of_day_equity is None:
        start_of_day_equity = eq
        return False
    loss_pct = (start_of_day_equity - eq) / start_of_day_equity
    return loss_pct >= MAX_DAILY_LOSS_PCT

def in_cooldown(symbol):
    if symbol not in last_trade_time:
        return False
    return (now_et() - last_trade_time[symbol]) < timedelta(minutes=SYMBOL_COOLDOWN_MIN)

# ================= DATA =================

def fetch_data(symbol):
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    return df.dropna().tail(LOOKBACK_MIN + 1)

# ================= SIGNAL =================

def get_signal(symbol):
    df = fetch_data(symbol)
    if df is None or len(df) < LOOKBACK_MIN:
        return None

    start = df["close"].iloc[0]
    end = df["close"].iloc[-1]
    move = (end - start) / start

    vol = df["volume"].iloc[-1]

    log.info(f"{symbol}: move {move*100:.4f}%")

    if abs(move) < MIN_MOVE_PCT:
        log.info(f"{symbol}: move too small — skip")
        return None

    if vol < MIN_VOLUME:
        log.info(f"{symbol}: volume too low — skip")
        return None

    side = "buy" if move > 0 else "sell"
    return {"symbol": symbol, "side": side, "price": end}

# ================= SIZING =================

def calc_qty(price):
    eq = equity()
    risk_dollars = eq * RISK_PER_TRADE
    per_share_risk = price * STOP_LOSS_PCT
    qty_risk = risk_dollars / per_share_risk

    cap_qty = (eq * MAX_POSITION_PCT) / price
    return max(1, int(min(qty_risk, cap_qty)))

# ================= ORDERS =================

def submit_trade(signal):
    symbol = signal["symbol"]
    side = signal["side"]
    price = signal["price"]

    qty = calc_qty(price)
    if qty <= 0:
        return

    if side == "buy":
        tp = round(price * (1 + TAKE_PROFIT_PCT), 2)
        sl = round(price * (1 - STOP_LOSS_PCT), 2)
    else:
        tp = round(price * (1 - TAKE_PROFIT_PCT), 2)
        sl = round(price * (1 + STOP_LOSS_PCT), 2)

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day",
        order_class="bracket",
        take_profit={"limit_price": tp},
        stop_loss={"stop_price": sl},
    )

    last_trade_time[symbol] = now_et()
    log.info(f"ORDER SENT {symbol} {side.upper()} qty={qty}")

# ================= EXITS =================

def manage_positions():
    positions = positions_by_symbol()

    for sym, pos in positions.items():
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        side = pos.side

        if side == "long":
            pnl_pct = (current - entry) / entry
        else:
            pnl_pct = (entry - current) / entry

        if pnl_pct <= -STOP_LOSS_PCT:
            log.info(f"{sym}: STOP LOSS EXIT")
            api.close_position(sym)

# ================= MAIN =================

def main():
    log.info("BOT START")

    if not market_open():
        log.info("Market closed")
        return

    if daily_loss_exceeded():
        log.error("DAILY LOSS LIMIT HIT — FLATTENING")
        api.close_all_positions()
        return

    positions = positions_by_symbol()

    for symbol in SYMBOLS:
        if symbol in positions:
            log.info(f"{symbol}: already in position — skip")
            continue

        if in_cooldown(symbol):
            log.info(f"{symbol}: cooldown — skip")
            continue

        signal = get_signal(symbol)
        if signal:
            submit_trade(signal)

    manage_positions()

    # FORCE FLAT BEFORE CLOSE
    now = now_et()
    if (now.hour, now.minute) >= FORCE_FLAT_TIME:
        log.info("FORCE FLAT — END OF DAY")
        api.close_all_positions()

    log.info("BOT END")

if __name__ == "__main__":
    main()
