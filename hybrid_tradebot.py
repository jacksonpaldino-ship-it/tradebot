import os
import time
import pytz
from datetime import datetime, timedelta

from alpaca_trade_api import REST, TimeFrame

# =============================
# ENV / API SETUP (FIXED)
# =============================
API_KEY = os.environ["ALPACA_API_KEY"]
API_SECRET = os.environ["ALPACA_SECRET_KEY"]
BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

NY = pytz.timezone("America/New_York")

# =============================
# CONFIG (SCALE SAFE)
# =============================
ACCOUNT_SIZE = 100_000       # change to 2_000 later
RISK_PER_TRADE_PCT = 0.002   # 0.2%
TAKE_PROFIT_PCT = 0.004      # 0.4%
TRAIL_AFTER_PCT = 0.003
TRAIL_GIVEBACK_PCT = 0.0015

MAX_POSITIONS = 2
SYMBOLS = ["XLF", "XLE", "XLV"]

# =============================
# TIME HELPERS
# =============================
def now_ny():
    return datetime.now(NY)

def market_close_soon():
    t = now_ny()
    close = t.replace(hour=16, minute=0, second=0)
    return close - timedelta(minutes=5) <= t <= close

def market_open():
    clock = api.get_clock()
    return clock.is_open

# =============================
# POSITION / SAFETY
# =============================
def flatten_all():
    positions = api.list_positions()
    for p in positions:
        qty = abs(int(p.qty))
        side = "sell" if p.side == "long" else "buy"
        api.submit_order(
            symbol=p.symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        print(f"FORCE FLATTEN {p.symbol}")

def manage_positions():
    positions = api.list_positions()

    for p in positions:
        qty = abs(int(p.qty))
        entry = float(p.avg_entry_price)
        current = float(p.current_price)

        pnl_pct = (
            (current - entry) / entry
            if p.side == "long"
            else (entry - current) / entry
        )

        # HARD TAKE PROFIT
        if pnl_pct >= TAKE_PROFIT_PCT:
            api.submit_order(
                symbol=p.symbol,
                qty=qty,
                side="sell" if p.side == "long" else "buy",
                type="market",
                time_in_force="day"
            )
            print(f"TAKE PROFIT {p.symbol} {pnl_pct:.2%}")
            continue

        # TRAILING STOP
        if pnl_pct >= TRAIL_AFTER_PCT:
            trail_price = (
                current * (1 - TRAIL_GIVEBACK_PCT)
                if p.side == "long"
                else current * (1 + TRAIL_GIVEBACK_PCT)
            )

            api.submit_order(
                symbol=p.symbol,
                qty=qty,
                side="sell" if p.side == "long" else "buy",
                type="trailing_stop",
                trail_price=round(trail_price, 2),
                time_in_force="day"
            )
            print(f"TRAIL SET {p.symbol}")

# =============================
# HIGH-FREQ-FIRST ENTRY LOGIC
# =============================
def entry_signal(symbol):
    bars = api.get_bars(symbol, TimeFrame.Minute, limit=20).df
    if len(bars) < 20:
        return None

    last = bars.iloc[-1]
    prev = bars.iloc[-2]

    # HIGH-FREQ FIRST: momentum impulse
    if last.close > prev.high * 1.001:
        return "long"

    if last.close < prev.low * 0.999:
        return "short"

    return None

def position_size(price):
    dollar_risk = ACCOUNT_SIZE * RISK_PER_TRADE_PCT
    return max(1, int(dollar_risk / price))

def open_positions_count():
    return len(api.list_positions())

# =============================
# MAIN LOOP
# =============================
def run():
    print("BOT START")

    while True:
        try:
            if not market_open():
                print("Market closed")
                time.sleep(60)
                continue

            # HARD FLATTEN BEFORE CLOSE
            if market_close_soon():
                print("Near close — flattening")
                flatten_all()
                time.sleep(300)
                continue

            manage_positions()

            if open_positions_count() >= MAX_POSITIONS:
                time.sleep(30)
                continue

            for symbol in SYMBOLS:
                if open_positions_count() >= MAX_POSITIONS:
                    break

                signal = entry_signal(symbol)
                if not signal:
                    continue

                price = float(api.get_last_trade(symbol).price)
                qty = position_size(price)

                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy" if signal == "long" else "sell",
                    type="market",
                    time_in_force="day"
                )

                print(f"ENTER {signal.upper()} {symbol} qty={qty}")
                time.sleep(2)

            time.sleep(15)

        except Exception as e:
            print("ERROR:", e)
            time.sleep(30)

if __name__ == "__main__":
    run()
