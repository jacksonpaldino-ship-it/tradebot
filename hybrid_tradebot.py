import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from alpaca_trade_api import REST

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]
TIMEFRAME = "5Min"
LOOKBACK = 50

RISK_PER_TRADE = 0.0025      # 0.25%
STOP_ATR_MULT = 1.2
MAX_POSITION_PCT = 0.15      # 15% per symbol
MAX_TOTAL_EXPOSURE = 0.5     # 50% total
MIN_ATR = 0.1

# ================= LOGGING =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= API =================

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca environment variables")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ================= INDICATORS =================

def compute_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()

# ================= DATA =================

def get_bars(symbol):
    bars = api.get_bars(
        symbol,
        TIMEFRAME,
        limit=LOOKBACK
    ).df

    if bars.empty:
        return None

    return bars

# ================= RISK =================

def total_exposure():
    positions = api.list_positions()
    return sum(abs(float(p.market_value)) for p in positions)

# ================= TRADING =================

def place_trade(symbol, side, atr):
    if atr is None or atr < MIN_ATR:
        return

    account = api.get_account()
    equity = float(account.equity)
    price = float(api.get_latest_trade(symbol).price)

    risk_dollars = equity * RISK_PER_TRADE
    stop_distance = atr * STOP_ATR_MULT

    qty = int(risk_dollars / stop_distance)
    if qty <= 0:
        return

    # Cap per-symbol exposure
    max_qty = int((equity * MAX_POSITION_PCT) / price)
    qty = min(qty, max_qty)

    if qty <= 0:
        return

    # Portfolio exposure check
    if total_exposure() + qty * price > equity * MAX_TOTAL_EXPOSURE:
        logging.info(f"{symbol} | exposure cap hit — skip")
        return

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    except Exception as e:
        logging.error(f"{symbol} ORDER FAILED — {e}")

# ================= STRATEGY =================

def run_cycle():
    logging.info("BOT START")

    account = api.get_account()
    equity = float(account.equity)
    pnl = float(account.equity) / float(account.last_equity) - 1 if account.last_equity else 0

    logging.info(f"Equity: {equity:.2f} | Daily PnL: {pnl:.2%}")

    for symbol in SYMBOLS:
        try:
            df = get_bars(symbol)
            if df is None or len(df) < 20:
                continue

            df["atr"] = compute_atr(df)
            last = df.iloc[-1]

            # Simple momentum filter (non-stupid)
            if last["close"] > df["close"].rolling(20).mean().iloc[-1]:
                place_trade(symbol, "buy", last["atr"])
            else:
                place_trade(symbol, "sell", last["atr"])

        except Exception as e:
            logging.error(f"{symbol} ERROR — {e}")

    logging.info("BOT END")

# ================= RUN =================

if __name__ == "__main__":
    run_cycle()
