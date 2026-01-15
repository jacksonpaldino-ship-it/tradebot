import os
import time
import pytz
import logging
from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
import pandas as pd

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

ACCOUNT_RISK_PER_TRADE = 0.002      # 0.2% of equity
MAX_POSITIONS = 3
COOLDOWN_MINUTES = 5

MIN_ATR_DOLLARS = 0.05
VWAP_REVERSION_ATR = 0.25
MOMENTUM_ATR = 0.05

FLATTEN_TIME = (15, 55)  # 3:55 PM ET
MARKET_TZ = pytz.timezone("America/New_York")

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

api = tradeapi.REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    base_url="https://paper-api.alpaca.markets"
)

last_trade_time = {}

# =============== HELPERS ==================

def market_open():
    clock = api.get_clock()
    return clock.is_open

def minutes_to_close():
    clock = api.get_clock()
    return (clock.next_close - clock.timestamp).total_seconds() / 60

def get_equity():
    return float(api.get_account().equity)

def get_positions():
    return {p.symbol: int(p.qty) for p in api.list_positions()}

def flatten_all():
    positions = api.list_positions()
    if not positions:
        logging.info("No positions to flatten")
        return

    for p in positions:
        qty = abs(int(p.qty))
        side = "sell" if int(p.qty) > 0 else "buy"
        if qty == 0:
            continue

        logging.info(f"FORCE FLATTEN {p.symbol} qty={qty}")
        try:
            api.submit_order(
                symbol=p.symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day"
            )
        except Exception as e:
            logging.error(f"Flatten error {p.symbol}: {e}")

def cooldown_ok(symbol):
    if symbol not in last_trade_time:
        return True
    return datetime.now(MARKET_TZ) - last_trade_time[symbol] > timedelta(minutes=COOLDOWN_MINUTES)

# ============== STRATEGY ==================

def get_bars(symbol):
    bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=50).df
    if bars.empty:
        return None
    return bars

def should_enter(symbol):
    bars = get_bars(symbol)
    if bars is None or len(bars) < 20:
        return None

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]

    atr = (high - low).rolling(14).mean().iloc[-1]
    if atr < MIN_ATR_DOLLARS:
        return None

    vwap = (close * volume).sum() / volume.sum()
    price = close.iloc[-1]
    vwap_dist = price - vwap
    price_change = price - close.iloc[-2]

    # --- HIGH FREQUENCY FIRST ---
    vwap_reversion = abs(vwap_dist) > VWAP_REVERSION_ATR * atr and abs(price_change) < 0.05 * atr
    if vwap_reversion:
        return "sell" if vwap_dist > 0 else "buy"

    # --- MOMENTUM SECOND ---
    momentum = abs(price_change) > MOMENTUM_ATR * atr
    if momentum:
        return "buy" if price_change > 0 else "sell"

    return None

# ============= EXECUTION ==================

def position_size(symbol, atr):
    equity = get_equity()
    risk_dollars = equity * ACCOUNT_RISK_PER_TRADE
    size = int(risk_dollars / atr)
    return max(size, 1)

def enter_trade(symbol, side):
    bars = get_bars(symbol)
    atr = (bars["high"] - bars["low"]).rolling(14).mean().iloc[-1]
    qty = position_size(symbol, atr)

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        last_trade_time[symbol] = datetime.now(MARKET_TZ)
    except Exception as e:
        logging.error(f"Order failed {symbol}: {e}")

# ================ MAIN ====================

def run():
    logging.info("BOT START")

    if not market_open():
        logging.info("Market closed — exit clean")
        return

    if minutes_to_close() < 10:
        logging.info("Market near close — flattening")
        flatten_all()
        return

    positions = get_positions()

    for symbol in SYMBOLS:
        if symbol in positions:
            continue
        if len(positions) >= MAX_POSITIONS:
            break
        if not cooldown_ok(symbol):
            continue

        signal = should_enter(symbol)
        if signal:
            enter_trade(symbol, signal)

    logging.info("BOT END")

if __name__ == "__main__":
    run()
