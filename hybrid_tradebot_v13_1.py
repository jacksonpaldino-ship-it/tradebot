import os
import time
import logging
from datetime import datetime, time as dtime
import pytz

from alpaca_trade_api import REST

# =========================
# CONFIG
# =========================

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

NOTIONAL_PER_TRADE = 1000        # very small, aggressive testing
MAX_TRADES_PER_DAY = 50
MAX_DAILY_LOSS_PCT = 0.01        # 1%
STOP_LOSS_PCT = 0.003            # 0.3%
TAKE_PROFIT_PCT = 0.002          # 0.2%

MIN_MOVE_PCT = 0.00015           # VERY aggressive
LOOKBACK_MINUTES = 1

TRADE_START = dtime(9, 30)
TRADE_END = dtime(15, 55)

SLEEP_SECONDS = 20               # aggressive polling

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger()

# =========================
# API
# =========================

api = REST(API_KEY, API_SECRET, BASE_URL)

# =========================
# STATE
# =========================

trade_count = 0
starting_equity = None

# =========================
# HELPERS
# =========================

def ny_time():
    return datetime.now(pytz.timezone("America/New_York"))

def market_open():
    now = ny_time().time()
    return TRADE_START <= now <= TRADE_END

def get_equity():
    return float(api.get_account().equity)

def daily_loss_exceeded():
    global starting_equity
    equity = get_equity()
    loss_pct = (starting_equity - equity) / starting_equity
    if loss_pct >= MAX_DAILY_LOSS_PCT:
        log.error(f"DAILY LOSS LIMIT HIT: {loss_pct:.2%}")
        return True
    return False

def get_recent_move(symbol):
    bars = api.get_bars(
        symbol,
        timeframe="1Min",
        limit=LOOKBACK_MINUTES + 1
    )

    if len(bars) < 2:
        return None

    old = bars[0].c
    new = bars[-1].c
    return (new - old) / old

def flatten(symbol):
    try:
        api.close_position(symbol)
        log.info(f"FLATTENED {symbol}")
    except Exception:
        pass

# =========================
# MAIN LOOP
# =========================

def run():
    global trade_count, starting_equity

    starting_equity = get_equity()
    log.info(f"STARTING EQUITY: {starting_equity}")

    while True:
        try:
            if not market_open():
                log.info("Market closed — sleeping")
                time.sleep(60)
                continue

            if trade_count >= MAX_TRADES_PER_DAY:
                log.warning("Max trades reached for day")
                break

            if daily_loss_exceeded():
                break

            for symbol in SYMBOLS:
                move = get_recent_move(symbol)

                if move is None:
                    log.info(f"{symbol}: not enough data")
                    continue

                log.info(f"{symbol}: move {move:.4%}")

                if abs(move) < MIN_MOVE_PCT:
                    log.info(f"{symbol}: move too small — skip")
                    continue

                side = "buy" if move > 0 else "sell"

                try:
                    api.submit_order(
                        symbol=symbol,
                        notional=NOTIONAL_PER_TRADE,
                        side=side,
                        type="market",
                        time_in_force="day",
                    )
                    trade_count += 1
                    log.info(f"ORDER SENT {symbol} {side.upper()}")

                    time.sleep(2)

                    position = api.get_position(symbol)
                    entry_price = float(position.avg_entry_price)

                    if side == "buy":
                        stop = entry_price * (1 - STOP_LOSS_PCT)
                        target = entry_price * (1 + TAKE_PROFIT_PCT)
                    else:
                        stop = entry_price * (1 + STOP_LOSS_PCT)
                        target = entry_price * (1 - TAKE_PROFIT_PCT)

                    # simple manual exit loop
                    while True:
                        last = api.get_last_trade(symbol).price

                        if side == "buy" and (last <= stop or last >= target):
                            flatten(symbol)
                            break

                        if side == "sell" and (last >= stop or last <= target):
                            flatten(symbol)
                            break

                        time.sleep(3)

                except Exception as e:
                    log.error(f"{symbol}: ORDER FAILED — {e}")

            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            log.warning("MANUAL STOP")
            break

        except Exception as e:
            log.error(f"FATAL LOOP ERROR: {e}")
            time.sleep(30)

    log.info("BOT STOPPED")

# =========================
# RUN
# =========================

if __name__ == "__main__":
    run()
