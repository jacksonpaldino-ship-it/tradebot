import os
import logging
from datetime import datetime, time
import pytz
import pandas as pd
from alpaca_trade_api import REST

# ================== ENV ==================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")

if not API_KEY or not API_SECRET or not BASE_URL:
    raise RuntimeError("Missing Alpaca credentials")

api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ================= CONFIG =================
SYMBOLS = ["XLE", "XLF", "XLV", "XLY", "IWM"]
RISK_PER_TRADE = 0.002      # 0.2% per trade (scales to $2k safely)
MAX_POSITIONS = 2
FLATTEN_TIME = time(15, 55)
BAR_LIMIT = 120

TZ = pytz.timezone("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ================= HELPERS =================
def now_et():
    return datetime.now(TZ)

def market_open():
    return api.get_clock().is_open

def near_close():
    return now_et().time() >= FLATTEN_TIME

def get_bars(symbol):
    df = api.get_bars(symbol, "1Min", limit=BAR_LIMIT).df
    df.index = df.index.tz_convert(TZ)
    return df

def indicators(df):
    df["ema8"] = df["close"].ewm(span=8).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    return df.dropna()

def open_positions():
    return {p.symbol: p for p in api.list_positions()}

# ================= RISK =================
def calc_qty(price, atr):
    equity = float(api.get_account().equity)
    risk_dollars = equity * RISK_PER_TRADE
    stop_dist = max(atr, price * 0.0025)
    qty = int(risk_dollars / stop_dist)
    return max(qty, 1)

# ================= EXECUTION =================
def place_trade(symbol, side, atr):
    positions = open_positions()
    if symbol in positions:
        return

    if len(positions) >= MAX_POSITIONS:
        logging.info(f"{symbol} | exposure cap hit — skip")
        return

    price = float(api.get_latest_trade(symbol).price)
    qty = calc_qty(price, atr)

    logging.info(f"{symbol} | {side.upper()} | qty={qty}")

    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day"
    )

def flatten_all():
    for p in api.list_positions():
        qty = abs(int(float(p.qty)))
        if qty == 0:
            continue
        side = "sell" if p.side == "long" else "buy"
        logging.info(f"FORCE FLATTEN {p.symbol} qty={qty}")
        api.submit_order(
            symbol=p.symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )

# ================= STRATEGY =================
def run():
    logging.info("BOT START")

    if not market_open():
        logging.info("Market closed — exit clean")
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

            # HIGHER-FREQUENCY CORE LOGIC
            if last.ema8 > last.ema21:
                place_trade(symbol, "buy", last.atr)
            elif last.ema8 < last.ema21:
                place_trade(symbol, "sell", last.atr)

        except Exception as e:
            logging.error(f"{symbol} ERROR — {e}")

    logging.info("BOT END")

if __name__ == "__main__":
    run()
