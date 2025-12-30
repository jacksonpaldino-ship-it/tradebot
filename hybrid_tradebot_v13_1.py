import os
import time
import logging
from datetime import datetime, timedelta
import pytz

from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame

# ================= CONFIG =================

SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]

RISK_PER_TRADE = 0.005          # 0.5% equity
STOP_R = 1.0
TARGET_R = 2.0
TRAIL_AFTER_R = 1.0

TIME_STOP_MINUTES = 20
MAX_DAILY_LOSS = -0.02          # -2% equity
COOLDOWN_MINUTES = 10

MIN_MOVE_PCT = 0.0015           # 0.15%
MIN_VOLUME = 50000

MARKET_CLOSE_BUFFER_MIN = 10    # flatten before close

# ================= LOGGING =================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= API =================

api = REST(
    os.environ["APCA_API_KEY_ID"],
    os.environ["APCA_API_SECRET_KEY"],
    base_url="https://paper-api.alpaca.markets"
)

NY = pytz.timezone("America/New_York")

# ================= STATE =================

cooldowns = {}
trade_entry_time = {}
trade_entry_price = {}
daily_start_equity = None

# ================= HELPERS =================

def now():
    return datetime.now(NY)

def market_open():
    clock = api.get_clock()
    return clock.is_open

def minutes_to_close():
    clock = api.get_clock()
    return (clock.next_close - clock.timestamp).total_seconds() / 60

def get_equity():
    return float(api.get_account().equity)

def get_position(symbol):
    try:
        return api.get_position(symbol)
    except:
        return None

def in_cooldown(symbol):
    if symbol not in cooldowns:
        return False
    return now() < cooldowns[symbol]

# ================= DAILY KILL SWITCH =================

def check_daily_loss():
    global daily_start_equity
    if daily_start_equity is None:
        daily_start_equity = get_equity()
        return False

    pnl = (get_equity() - daily_start_equity) / daily_start_equity
    if pnl <= MAX_DAILY_LOSS:
        logging.error("MAX DAILY LOSS HIT — FLATTENING")
        close_all()
        return True
    return False

# ================= EXIT LOGIC =================

def exit_conditions(symbol, position, last_price):
    entry = trade_entry_price[symbol]
    age = (now() - trade_entry_time[symbol]).total_seconds() / 60

    qty = abs(float(position.qty))
    side = position.side

    risk = entry * STOP_R * MIN_MOVE_PCT
    reward = entry * TARGET_R * MIN_MOVE_PCT

    # Directional math
    pnl = (last_price - entry) if side == "long" else (entry - last_price)

    # STOP LOSS
    if pnl <= -risk:
        return "STOP"

    # PROFIT TARGET
    if pnl >= reward:
        return "TARGET"

    # TIME STOP
    if age >= TIME_STOP_MINUTES and pnl < reward * 0.25:
        return "TIME"

    # MOMENTUM FAILURE
    if pnl < 0 and age > 5:
        return "FAIL"

    # TRAIL STOP
    if pnl >= risk * TRAIL_AFTER_R:
        trail_level = entry
        if (side == "long" and last_price <= trail_level) or \
           (side == "short" and last_price >= trail_level):
            return "TRAIL"

    return None

def close_position(symbol, reason):
    try:
        api.close_position(symbol)
        cooldowns[symbol] = now() + timedelta(minutes=COOLDOWN_MINUTES)
        trade_entry_time.pop(symbol, None)
        trade_entry_price.pop(symbol, None)
        logging.info(f"{symbol}: EXIT — {reason}")
    except Exception as e:
        logging.error(f"{symbol}: EXIT FAILED — {e}")

def close_all():
    for pos in api.list_positions():
        close_position(pos.symbol, "FLATTEN")

# ================= ENTRY LOGIC =================

def try_entry(symbol):
    if in_cooldown(symbol):
        return

    if get_position(symbol):
        return

    bars = api.get_bars(symbol, TimeFrame.Minute, limit=6).df
    if len(bars) < 6:
        return

    move = (bars.close.iloc[-1] - bars.open.iloc[0]) / bars.open.iloc[0]
    volume = bars.volume.sum()

    if abs(move) < MIN_MOVE_PCT or volume < MIN_VOLUME:
        return

    side = "buy" if move > 0 else "sell"
    equity = get_equity()
    risk_dollars = equity * RISK_PER_TRADE
    qty = int(risk_dollars / (bars.close.iloc[-1] * MIN_MOVE_PCT))
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

        trade_entry_time[symbol] = now()
        trade_entry_price[symbol] = bars.close.iloc[-1]

        logging.info(f"{symbol}: ENTER {side.upper()} {qty}")

    except Exception as e:
        logging.error(f"{symbol}: ENTRY FAILED — {e}")

# ================= MAIN =================

def main():
    logging.info("BOT START")

    if not market_open():
        logging.info("Market closed")
        return

    if check_daily_loss():
        return

    if minutes_to_close() <= MARKET_CLOSE_BUFFER_MIN:
        logging.info("Flattening before close")
        close_all()
        return

    # Manage open positions
    for pos in api.list_positions():
        last = api.get_latest_trade(pos.symbol).price
        reason = exit_conditions(pos.symbol, pos, last)
        if reason:
            close_position(pos.symbol, reason)

    # Look for new entries
    for symbol in SYMBOLS:
        try_entry(symbol)

    logging.info("BOT END")

if __name__ == "__main__":
    main()
