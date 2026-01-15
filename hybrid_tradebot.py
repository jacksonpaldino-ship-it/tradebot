import os
import logging
from datetime import datetime, time
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ---------------- CONFIG ----------------
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]
RISK_PER_TRADE = 0.003        # 0.3% per trade (safe at 100k AND 2k)
MAX_POSITIONS = 2
FLATTEN_TIME = time(15, 55)   # 3:55 PM ET
BAR_LIMIT = 100

TZ = pytz.timezone("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

api = REST(api_version="v2")

# ---------------- HELPERS ----------------
def now_et():
    return datetime.now(TZ)

def market_open():
    clock = api.get_clock()
    return clock.is_open

def near_close():
    return now_et().time() >= FLATTEN_TIME

def get_bars(symbol):
    bars = api.get_bars(symbol, "1Min", limit=BAR_LIMIT).df
    bars.index = bars.index.tz_convert(TZ)
    return bars

def indicators(df):
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    return df.dropna()

def position_count():
    return len(api.list_positions())

# ---------------- RISK ----------------
def calc_qty(price, atr):
    acct = api.get_account()
    equity = float(acct.equity)
    risk_dollars = equity * RISK_PER_TRADE
    stop_dist = max(atr, price * 0.002)  # volatility-aware
    qty = int(risk_dollars / stop_dist)
    return max(qty, 1)

# ---------------- EXECUTION ----------------
def place_trade(symbol, side, atr):
    price = float(api.get_latest_trade(symbol).price)
    qty = calc_qty(price, atr)

    if position_count() >= MAX_POSITIONS:
        logging.info(f"{symbol} | max positions hit — skip")
        return

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day"
    )

def flatten_all():
    for pos in api.list_positions():
        qty = abs(int(float(pos.qty)))
        if qty == 0:
            continue
        side = "sell" if pos.side == "long" else "buy"
        logging.info(f"FORCE FLATTEN {pos.symbol} qty={qty}")
        api.submit_order(
            symbol=pos.symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )

# ---------------- STRATEGY ----------------
def run():
    logging.info("BOT START")

    if not market_open():
        logging.info("Market closed — exit")
        return

    if near_close():
        logging.info("Market near close — flattening")
        flatten_all()
        return

    acct = api.get_account()
    logging.info(f"Equity: {acct.equity}")

    for symbol in SYMBOLS:
        try:
            df = indicators(get_bars(symbol))
            last = df.iloc[-1]

            # HIGHER-FREQUENCY LOGIC
            if last.ema9 > last.ema21:
                place_trade(symbol, "buy", last.atr)
            elif last.ema9 < last.ema21:
                place_trade(symbol, "sell", last.atr)

        except Exception as e:
            logging.error(f"{symbol} ERROR — {e}")

    logging.info("BOT END")

if __name__ == "__main__":
    run()
