import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pytz

from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError

# =========================
# ENV + API SETUP
# =========================

REQUIRED_VARS = [
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL"
]

missing = [v for v in REQUIRED_VARS if v not in os.environ]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

API_KEY = os.environ["APCA_API_KEY_ID"]
API_SECRET = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger()

# =========================
# CONFIG
# =========================

SYMBOLS = ["XLE", "XLV", "XLF", "XLY", "IWM"]

MAX_POSITION_PCT = 0.20        # 20% of equity per position
STOP_LOSS_PCT = 0.004          # 0.4%
TAKE_PROFIT_PCT = 0.006        # 0.6%
MIN_MOVE_PCT = 0.001           # 0.1%
MAX_DAILY_LOSS = 0.01          # 1% equity
COOLDOWN_MINUTES = 15

MARKET_TZ = pytz.timezone("America/New_York")
TRADE_START = datetime.strptime("09:30", "%H:%M").time()
TRADE_END = datetime.strptime("15:45", "%H:%M").time()
FORCE_EXIT_TIME = datetime.strptime("15:55", "%H:%M").time()

# =========================
# STATE
# =========================

cooldowns = {}

# =========================
# HELPERS
# =========================

def now_et():
    return datetime.now(MARKET_TZ)

def market_open_now():
    clock = api.get_clock()
    return clock.is_open

def within_trade_window():
    t = now_et().time()
    return TRADE_START <= t <= TRADE_END

def force_exit_window():
    return now_et().time() >= FORCE_EXIT_TIME

def equity():
    return float(api.get_account().equity)

def daily_pnl_pct():
    acct = api.get_account()
    start = float(acct.last_equity) - float(acct.equity)
    return start / float(acct.last_equity)

def in_cooldown(symbol):
    if symbol not in cooldowns:
        return False
    return now_et() < cooldowns[symbol]

def set_cooldown(symbol):
    cooldowns[symbol] = now_et() + timedelta(minutes=COOLDOWN_MINUTES)

# =========================
# EXITS
# =========================

def manage_open_positions():
    positions = api.list_positions()

    for p in positions:
        symbol = p.symbol
        qty = float(p.qty)
        entry = float(p.avg_entry_price)
        price = float(api.get_latest_trade(symbol).price)

        if qty == 0:
            continue

        pnl_pct = (price - entry) / entry
        if qty < 0:
            pnl_pct = -pnl_pct

        if pnl_pct <= -STOP_LOSS_PCT:
            log.info(f"{symbol}: STOP LOSS hit")
            api.close_position(symbol)
            set_cooldown(symbol)

        elif pnl_pct >= TAKE_PROFIT_PCT:
            log.info(f"{symbol}: TAKE PROFIT hit")
            api.close_position(symbol)
            set_cooldown(symbol)

# =========================
# ENTRY LOGIC
# =========================

def try_enter(symbol):
    if in_cooldown(symbol):
        return

    bars = api.get_bars(symbol, "1Min", limit=3)
    if len(bars) < 3:
        return

    prev = bars[-2].c
    last = bars[-1].c
    move = (last - prev) / prev

    if abs(move) < MIN_MOVE_PCT:
        return

    side = "buy" if move > 0 else "sell"

    acct_equity = equity()
    notional = acct_equity * MAX_POSITION_PCT
    qty = round(notional / last, 2)

    if qty <= 0:
        return

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        log.info(f"{symbol}: ENTER {side.upper()} qty={qty}")
        set_cooldown(symbol)

    except APIError as e:
        log.error(f"{symbol}: order error {e}")

# =========================
# MAIN
# =========================

def main():
    log.info("BOT START")

    if not market_open_now():
        log.info("Market closed")
        return

    if daily_pnl_pct() <= -MAX_DAILY_LOSS:
        log.error("DAILY LOSS LIMIT HIT — LIQUIDATING")
        api.close_all_positions()
        return

    manage_open_positions()

    if force_exit_window():
        log.info("FORCE EXIT WINDOW — closing all positions")
        api.close_all_positions()
        return

    if not within_trade_window():
        log.info("Outside trade window")
        return

    open_symbols = {p.symbol for p in api.list_positions()}

    for symbol in SYMBOLS:
        if symbol in open_symbols:
            continue
        try_enter(symbol)

    log.info("BOT END")

# =========================

if __name__ == "__main__":
    main()
