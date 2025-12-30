import os
import sys
import logging
from datetime import datetime, time, timedelta
import pytz

from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame

# ======================
# LOGGING
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger()

# ======================
# ENV VARS (MATCH YOUR SECRETS)
# ======================
REQUIRED_ENV = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
]

missing = [k for k in REQUIRED_ENV if k not in os.environ]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

API_KEY = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_SECRET_KEY"]
BASE_URL = os.environ["ALPACA_BASE_URL"]

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ======================
# CONFIG
# ======================
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

RISK_PER_TRADE = 0.01        # 1% equity
STOP_LOSS_PCT = 0.003        # 0.3%
TAKE_PROFIT_PCT = 0.006      # 0.6%
MAX_DAILY_LOSS = 0.02        # 2% kill switch
MIN_MOVE_PCT = 0.0015        # 0.15%
COOLDOWN_MINUTES = 15

NY = pytz.timezone("America/New_York")

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
FORCE_EXIT_TIME = time(15, 55)

last_trade_time = {}

# ======================
# HELPERS
# ======================
def now_ny():
    return datetime.now(NY)

def market_is_open():
    t = now_ny().time()
    return MARKET_OPEN <= t <= MARKET_CLOSE

def near_close():
    return now_ny().time() >= FORCE_EXIT_TIME

def in_cooldown(symbol):
    if symbol not in last_trade_time:
        return False
    return now_ny() - last_trade_time[symbol] < timedelta(minutes=COOLDOWN_MINUTES)

# ======================
# MAIN
# ======================
def main():
    log.info("BOT START")

    if not market_is_open():
        log.info("Market closed — exit")
        return

    account = api.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    daily_pnl_pct = (equity - last_equity) / last_equity

    log.info(f"Equity: {equity:.2f} | Daily PnL: {daily_pnl_pct:.2%}")

    if daily_pnl_pct <= -MAX_DAILY_LOSS:
        log.error("MAX DAILY LOSS HIT — LIQUIDATING")
        api.close_all_positions()
        return

    positions = {p.symbol: p for p in api.list_positions()}

    # ======================
    # EXIT LOGIC (ALWAYS FIRST)
    # ======================
    for sym, pos in positions.items():
        entry = float(pos.avg_entry_price)
        price = float(pos.current_price)
        side = pos.side

        if side == "long":
            if price <= entry * (1 - STOP_LOSS_PCT):
                log.warning(f"{sym}: STOP LOSS HIT")
                api.close_position(sym)
                continue
            if price >= entry * (1 + TAKE_PROFIT_PCT):
                log.info(f"{sym}: TAKE PROFIT HIT")
                api.close_position(sym)
                continue

        if side == "short":
            if price >= entry * (1 + STOP_LOSS_PCT):
                log.warning(f"{sym}: STOP LOSS HIT (SHORT)")
                api.close_position(sym)
                continue
            if price <= entry * (1 - TAKE_PROFIT_PCT):
                log.info(f"{sym}: TAKE PROFIT HIT (SHORT)")
                api.close_position(sym)
                continue

    # Force flat before close
    if near_close():
        log.info("Near market close — flattening all positions")
        api.close_all_positions()
        return

    # ======================
    # ENTRY LOGIC
    # ======================
    for symbol in SYMBOLS:
        if symbol in positions:
            continue

        if in_cooldown(symbol):
            continue

        bars = api.get_bars(symbol, TimeFrame.Minute, limit=2)
        if len(bars) < 2:
            continue

        prev = bars[-2].c
        curr = bars[-1].c
        move_pct = (curr - prev) / prev

        if abs(move_pct) < MIN_MOVE_PCT:
            continue

        qty = int((equity * RISK_PER_TRADE) / curr)
        if qty <= 0:
            continue

        if move_pct > 0:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="day",
            )
            log.info(f"{symbol}: LONG ENTRY {qty}")

        else:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
            log.info(f"{symbol}: SHORT ENTRY {qty}")

        last_trade_time[symbol] = now_ny()

    log.info("BOT END")

# ======================
if __name__ == "__main__":
    main()
