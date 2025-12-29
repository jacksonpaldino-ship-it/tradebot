import os
import logging
from datetime import datetime, time
import pytz
import alpaca_trade_api as tradeapi
import pandas as pd

# =====================
# CONFIG
# =====================
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]
RISK_PER_TRADE = 0.01          # 1% equity
TAKE_PROFIT_PCT = 0.003        # 0.30%
STOP_LOSS_PCT = 0.002          # 0.20%
MIN_MOVE_PCT = 0.001           # 0.10%
BAR_LOOKBACK = 5               # 5 minutes
MAX_TRADES_PER_RUN = 3

# =====================
# LOGGING
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger()

# =====================
# API
# =====================
api = tradeapi.REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    os.environ["ALPACA_BASE_URL"],
    api_version="v2",
)

# =====================
# HELPERS
# =====================
def market_open():
    ny = pytz.timezone("America/New_York")
    now = datetime.now(ny).time()
    return time(9, 30) <= now <= time(15, 55)

def already_in_position(symbol):
    return symbol in [p.symbol for p in api.list_positions()]

def get_move(symbol):
    bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=BAR_LOOKBACK).df
    if bars.empty or len(bars) < BAR_LOOKBACK:
        return None
    open_price = bars.iloc[0]["open"]
    close_price = bars.iloc[-1]["close"]
    return (close_price - open_price) / open_price

def submit_bracket(symbol, side, qty, price):
    if side == "buy":
        tp = round(price * (1 + TAKE_PROFIT_PCT), 2)
        sl = round(price * (1 - STOP_LOSS_PCT), 2)

        # Enforce minimum distance
        if tp <= price:
            tp = round(price + 0.01, 2)
        if sl >= price:
            sl = round(price - 0.01, 2)

    else:  # SHORT
        tp = round(price * (1 - TAKE_PROFIT_PCT), 2)
        sl = round(price * (1 + STOP_LOSS_PCT), 2)

        # Enforce Alpaca short rules
        if tp >= price - 0.01:
            tp = round(price - 0.01, 2)
        if sl <= price + 0.01:
            sl = round(price + 0.01, 2)

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

# =====================
# MAIN
# =====================
def main():
    log.info("BOT START")

    if not market_open():
        log.info("Market closed — exit")
        return

    account = api.get_account()
    equity = float(account.equity)
    log.info(f"Equity: {equity}")

    trades = 0

    for symbol in SYMBOLS:
        if trades >= MAX_TRADES_PER_RUN:
            break

        try:
            if already_in_position(symbol):
                log.info(f"{symbol}: already in position — skip")
                continue

            move = get_move(symbol)
            if move is None:
                log.info(f"{symbol}: insufficient data")
                continue

            log.info(f"{symbol}: move {move:.4%}")

            if abs(move) < MIN_MOVE_PCT:
                log.info(f"{symbol}: move too small — skip")
                continue

            side = "buy" if move > 0 else "sell"

            last_bar = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=1).df.iloc[-1]
            price = last_bar["close"]

            risk_dollars = equity * RISK_PER_TRADE
            qty = int(risk_dollars / (price * STOP_LOSS_PCT))

            if qty <= 0:
                log.info(f"{symbol}: qty zero — skip")
                continue

            submit_bracket(symbol, side, qty, price)
            log.info(f"{symbol}: ORDER SENT {side.upper()} {qty}")

            trades += 1

        except Exception as e:
            log.error(f"{symbol}: ERROR — {e}")

    log.info("BOT END")

if __name__ == "__main__":
    main()
