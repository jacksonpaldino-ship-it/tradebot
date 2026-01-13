import os
import time
import logging
from datetime import datetime, timezone

import pandas as pd
from alpaca_trade_api import REST

# =========================
# CONFIG (CHANGE THESE ONLY)
# =========================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

RISK_PER_TRADE_PCT = 0.002      # 0.2% per trade (SAFE)
MAX_POSITIONS = 2
FLATTEN_MINUTES_BEFORE_CLOSE = 5

BAR_TIMEFRAME = "1Min"
BAR_LIMIT = 100

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# =========================
# API INIT
# =========================

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca environment variables")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# =========================
# UTILITIES
# =========================

def market_is_open():
    clock = api.get_clock()
    return clock.is_open, clock

def minutes_to_close(clock):
    delta = clock.next_close - clock.timestamp
    return delta.total_seconds() / 60

def get_equity():
    return float(api.get_account().equity)

def get_positions_dict():
    positions = api.list_positions()
    return {p.symbol: p for p in positions}

def safe_qty(value, price):
    if price <= 0:
        return 0
    return int(value // price)

# =========================
# DATA
# =========================

def get_bars(symbol):
    bars = api.get_bars(symbol, BAR_TIMEFRAME, limit=BAR_LIMIT).df
    if bars.empty:
        return None
    bars = bars.reset_index()
    return bars

def compute_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

# =========================
# STRATEGY
# =========================

def generate_signal(df):
    df["atr"] = compute_atr(df)
    if df["atr"].isna().iloc[-1]:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Simple momentum + volatility filter
    if last["close"] > prev["high"] and last["atr"] > df["atr"].mean():
        return "buy"

    if last["close"] < prev["low"] and last["atr"] > df["atr"].mean():
        return "sell"

    return None

# =========================
# ORDER EXECUTION
# =========================

def place_order(symbol, side):
    price = float(api.get_latest_trade(symbol).price)
    equity = get_equity()

    risk_dollars = equity * RISK_PER_TRADE_PCT
    qty = safe_qty(risk_dollars, price)

    if qty <= 0:
        logger.info(f"{symbol} | qty=0 — skip")
        return

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )
        logger.info(f"{symbol} | {side.upper()} | qty={qty}")
    except Exception as e:
        logger.error(f"{symbol} ORDER FAILED — {e}")

# =========================
# FLATTEN (SAFE)
# =========================

def flatten_all():
    positions = api.list_positions()
    if not positions:
        logger.info("No positions to flatten")
        return

    for p in positions:
        qty = abs(int(float(p.qty)))
        if qty == 0:
            continue

        side = "sell" if p.side == "long" else "buy"

        try:
            api.submit_order(
                symbol=p.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            logger.info(f"FLATTEN {p.symbol} qty={qty}")
        except Exception as e:
            logger.error(f"FLATTEN FAILED {p.symbol} — {e}")

# =========================
# MAIN RUN (NO LOOPS)
# =========================

def run():
    logger.info("BOT START")

    is_open, clock = market_is_open()

    if not is_open:
        logger.info("Market closed — exiting")
        logger.info("BOT END")
        return

    mins_to_close = minutes_to_close(clock)

    if mins_to_close <= FLATTEN_MINUTES_BEFORE_CLOSE:
        logger.info("Market near close — flattening")
        flatten_all()
        logger.info("BOT END")
        return

    equity = get_equity()
    logger.info(f"Equity: {equity:.2f}")

    positions = get_positions_dict()

    if len(positions) >= MAX_POSITIONS:
        logger.info("Max positions reached — skip entries")
        logger.info("BOT END")
        return

    for symbol in SYMBOLS:
        if symbol in positions:
            continue

        df = get_bars(symbol)
        if df is None or len(df) < 20:
            continue

        signal = generate_signal(df)
        if signal:
            place_order(symbol, signal)
            break  # one trade per run

    logger.info("BOT END")

# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    run()
